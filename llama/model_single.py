from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
import hiq


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 1
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # shape of x : (1,8,32,64)
    ndim = x.ndim # 4
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # (8,64) == (8,64)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # (1,8,1,64)
    return freqs_cis.view(*shape) # (1,8,1,64)


def apply_rotary_emb(
    xq: torch.Tensor, # (1,8,32,128)
    xk: torch.Tensor, # (1,8,32,128)
    freqs_cis: torch.Tensor, # (8,64)
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # (1,8,32,128) -> (1,8,32,64,2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # (1,8,32,128) -> (1,8,32,64,2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # (1,8,1,64)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # (1,8,32,64,2) -> (1,8,32,128) 
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # (1,8,32,64,2) -> (1,8,32,128)
    return xq_out.type_as(xq), xk_out.type_as(xk) # (1,8,32,128), (1,8,32,128)



class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // 1 # 32 // 1 = 32
        self.head_dim = args.dim // args.n_heads # 4096 // 32 = 128

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        ) # (4096, 4096)
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        ) # (4096, 4096)
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )  # (4096, 4096)
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        ) # (4096, 4096)
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            # (1,1024,32,128)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
            # (1,1024,32,128)
        )
        if hiq.get_env_bool("KV_CAHCHE_IN_GPU", True):
            self.cache_k = self.cache_k.cuda()
            self.cache_v = self.cache_v.cuda()

    def forward(
        self,
        x: torch.Tensor, # (1,8,4096)
        start_pos: int, # 0 (initially)
        freqs_cis: torch.Tensor,  # (8, 64)
        mask: Optional[torch.Tensor],  # (1,1,8,8)
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # all of shape (1,8,4096)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim) # (1,8,32,128)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) # (1,8,32,128), (1,8,32,128)

        self.cache_k = self.cache_k.to(xq) # (1,1024,32,128)
        self.cache_v = self.cache_v.to(xq) # (1,1024,32,128)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk # (1,1024,32,128)
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv # (1,1024,32,128)

        keys = self.cache_k[:bsz, : start_pos + seqlen] # (1,1024,32,128)
        values = self.cache_v[:bsz, : start_pos + seqlen] # (1,1024,32,128)

        xq = xq.transpose(1, 2) # (1,32,8,128)
        keys = keys.transpose(1, 2) # (1,32,1024,128)
        values = values.transpose(1, 2) # (1,32,1024,128)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) # (1,32,8,1024)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq) # (1,32,8,1024)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim) # (1,32,8,128)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1) # (1,8,4096)

        return self.wo(output) # (1,8,4096)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int, # 4096
        hidden_dim: int, # 4 * 4096 = 16384
        multiple_of: int, # 256
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3) # 2 * 16384 / 3 = 10922
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # 256 * (10922 + 256 - 1) // 256 = 11177

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # (4096, 11177)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # (11177, 4096)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # (4096, 11177)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # (1,8,4096)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
            # 4096, 4 * 4096, 256
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor, # (1,8,4096)
        start_pos: int, # 0 (initially)
        freqs_cis: torch.Tensor, # (8, 64)
        mask: Optional[torch.Tensor], # (1,1,8,8)
    ):
        # this is a skip connection
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
            # (1,8,4096), 0, (1024, 64), (1,1,8,8)
        ) # (1,8,4096)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out # (1,8,4096)


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params # ModelArgs
        self.vocab_size = params.vocab_size # 32_000
        self.n_layers = params.n_layers # 32

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim) # (32_000, 4096)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps) # shape of output is same as input
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False) # (4096, 32_000)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
            # 4096 // 32 = 128, 1024 * 2
        ) # torch.Size([2048, 64])

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape # (1,8)
        h = self.tok_embeddings(tokens) # (1,8,4096)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen] # torch.Size([8, 64])

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            ) # (1,1,8,8) , filled with -inf
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
            # (1,1,8,8) , filled with -inf, but only the upper triangle, lower triangle is 0
            # diagnol = start_pos + 1, so the first 8 tokens are not masked, it basically pushes the diagonola above


        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h) # (1,8,4096)
        output = self.output(h[:, -1, :])  # only compute last logits # (1, 4096) * (4096, 32_000) = (1, 32_000)
        return output.float() # (1, 32_000)
