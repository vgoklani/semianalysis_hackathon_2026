"""
Inference script for Qwen 3.5-0.8B (pure PyTorch, no HuggingFace).

Usage:
    # First, download and convert:
    huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./Qwen3.5-0.8B
    python -m qwen3_5.convert_weights --hf-dir ./Qwen3.5-0.8B --output qwen35_weights.pt

    # Then generate:
    python -m qwen3_5.generate --weights qwen35_weights.pt --tokenizer-dir ./Qwen3.5-0.8B --prompt "Hello, how are you?"

    # Or with chat format:
    python -m qwen3_5.generate --weights qwen35_weights.pt --tokenizer-dir ./Qwen3.5-0.8B --chat --prompt "What is the capital of France?"
"""

import argparse
import time

import torch

from .model import Qwen35Model, Qwen35Config
from .tokenizer import Qwen35Tokenizer


def load_model(weights_path: str, device: str = "cpu", dtype=torch.bfloat16) -> Qwen35Model:
    config = Qwen35Config()
    model = Qwen35Model(config)

    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Handle tied weights: if lm_head.weight is missing, it's tied to embed_tokens
    if "lm_head.weight" not in state_dict and config.tie_word_embeddings:
        print("lm_head.weight not found, using tied embeddings")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing keys (tied lm_head)
        real_missing = [k for k in missing if k != "lm_head.weight"]
        if real_missing:
            print(f"Warning: missing keys: {real_missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:10]}")

    model = model.to(dtype=dtype, device=device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M parameters on {device} ({dtype})")
    return model


def generate_text(
    model: Qwen35Model,
    tokenizer: Qwen35Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    chat: bool = False,
    device: str = "cpu",
):
    if chat:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.encode_chat(messages)
    else:
        input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor([input_ids], device=device)

    print(f"Prompt tokens: {len(input_ids)}")
    print(f"Generating up to {max_new_tokens} tokens...")
    print("-" * 60)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(generated_text)
    print("-" * 60)
    print(f"Generated {len(generated_ids)} tokens in {elapsed:.2f}s "
          f"({len(generated_ids) / elapsed:.1f} tok/s)")


def main():
    parser = argparse.ArgumentParser(description="Generate text with Qwen 3.5-0.8B")
    parser.add_argument("--weights", required=True, help="Path to converted weights (.pt)")
    parser.add_argument("--tokenizer-dir", required=True,
                        help="Path to tokenizer dir (with vocab.json + merges.txt)")
    parser.add_argument("--prompt", default="The meaning of life is",
                        help="Input prompt")
    parser.add_argument("--chat", action="store_true",
                        help="Use chat (ChatML) format")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    tokenizer = Qwen35Tokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model(args.weights, device=args.device, dtype=dtype_map[args.dtype])

    generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        chat=args.chat,
        device=args.device,
    )


if __name__ == "__main__":
    main()
