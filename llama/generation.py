# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer


class LLaMA:
    def __init__(self, model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def _should_stop(self, tokens, prompt_tokens, stop_ids, stop_words):
        """credits go to: https://github.com/galatolofederico/vanilla-llama
        If a particular token is in the stop_ids list, then we stop generation.
        Thats it.
        """

        if stop_ids is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                g = t[len(p):].tolist()
                for stop_id in stop_ids:
                    if stop_id in g:
                        do_stop[i] = True

            if all(do_stop):
                return True

        if stop_words is not None:
            do_stop = [False for _ in range(len(tokens))]
            for i, (t, p) in enumerate(zip(tokens, prompt_tokens)):
                t = t.clone()
                g = t[len(p):]
                g[g == self.tokenizer.pad_id] = self.tokenizer.eos_id
                g = g.tolist()
                d = self.tokenizer.decode(g)
                for stop_word in stop_words:
                    if stop_word in d:
                        do_stop[i] = True

            if all(do_stop):
                return True

        return False

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int, # 256
        temperature: float = 0.8, # 0.8
        top_p: float = 0.95, # 0.95
        stop_ids: List[int] = None,
        stop_words: List[str] = None,
    ) -> List[str]:
        bsz = len(prompts) # 1
        params = self.model.params # those same ModelArgs
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # [[1, 306, 4658, 278, 6593, 310, 2834, 338]]

        min_prompt_size = min([len(t) for t in prompt_tokens]) # 8
        max_prompt_size = max([len(t) for t in prompt_tokens]) # 8

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # 264

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # a tensor of size (1, 264) filled with -1's

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
            # fill the first 8(length of prompt_tokens[0]) tokens of tokens with the prompt tokens
        input_text_mask = tokens != self.tokenizer.pad_id # a tensor of size (1, 264) filled with True's ,
        # where tokens is not -1, other wise False
        start_pos = min_prompt_size # 8
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            i = tokens[:, prev_pos:cur_pos]
            logits = self.model(i, prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            
            if self._should_stop(tokens, prompt_tokens, stop_ids, stop_words):
                break

        tokens[tokens == self.tokenizer.pad_id] = self.tokenizer.eos_id
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        #print(decoded)
        return [postprocessing(i, stop_words) for i in decoded]


def postprocessing(output_text, stop_words=None, threshold=10):
    """ 
    The purpose of this post-processing function is to clean up the generated 
    text and ensure that it contains valid and properly formatted sentences. 
    It also provides the flexibility to handle specific cases such as removing
    stop words and enforcing proper sentence endings.
    """
    sentences = output_text.split(".")
    filtered_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > threshold and sentence[-1] == ".":
            filtered_sentences.append(sentence)
    r = '.'.join(sentences).strip()
    if stop_words:
        for w in stop_words:
            if r.endswith(w):
                r = r[0:-len(w)].strip()
    if r[-1] != '.':
        r += '...'
    return r


def sample_top_p(probs, # a tensor of size (1, 32_000) with softmax already applied
                 p # 0.95
                 ):
    """ 
    In summary, the sample_top_p function performs top-p sampling on a given probability
    distribution. It sorts the probabilities, computes the cumulative sum, applies a 
    threshold to remove tokens with cumulative probabilities exceeding the threshold, 
    normalizes the probabilities, samples a token using multinomial sampling, and returns 
    the sampled token index.

    If the sum of probalities coming before a token is greater than p, then the token is
    not considered for sampling. This is done by setting the probability of the token to 0.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
