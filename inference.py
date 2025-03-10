import torch

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    """
    Used in run function to load the model
    Args:
        ckpt_dir: path to the checkpoint directory
        tokenizer_path: path to the tokenizer
        local_rank: local rank of the process
        world_size: world size of the process
        max_seq_len: maximum sequence length
        max_batch_size: maximum batch size
    Returns:
        generator: LLaMA model
    """

    # gives a sorted list of file that match the .pth pattern from the ckpt_dir
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    # The code below does the following, explained in English:
    # 1. Assert that the number of checkpoints (len(checkpoints)) is equal to the world size. This is because we are loading a checkpoint for MP, and we need to make sure that the number of checkpoints is equal to the number of processes. 
    # 2. Set the ckpt_path to the checkpoint corresponding to the local rank. 
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    # load the checkpoint to cpu. Why CPU though?
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load the params from the checkpoint directory which were in a json file
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    # create a model args object
    # ModelArgs is a a simple dataclass that contains the parameters for the model
    # file in llama/model_single.py
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )


    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


def run(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
):
    local_rank = 0
    world_size = 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",  # removed: keep only one prompt
    ]
    while True:
        print("Prompt:", prompts)
        results = generator.generate(
            prompts, max_gen_len=256, temperature=temperature, top_p=top_p
        )
        for result in results:
            print("🦙LLaMA:", result.strip())

        user_input = input("please enter your prompts (Ctrl+C to exit): ")
        prompts = [user_input]


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument(
        "--tokenizer_path", type=str, default="/llama_data/tokenizer.model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1,
    )
