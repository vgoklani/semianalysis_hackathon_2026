"""
Convert HuggingFace Qwen 3.5-0.8B weights to our pure PyTorch format.

Usage:
    python -m qwen3_5.convert_weights --hf-dir ./Qwen3.5-0.8B --output ./qwen35_weights.pt

Prerequisites:
    pip install safetensors
    Download model: huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./Qwen3.5-0.8B
"""

import argparse
import json
import re
from pathlib import Path
from collections import OrderedDict

import torch

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


# Mapping from HuggingFace key patterns to our model key patterns
# HF uses: model.text_model.layers.{i}.{submodule}
# We use:  layers.{i}.{submodule}

def map_hf_key(hf_key: str) -> str | None:
    """Map a HuggingFace state dict key to our model's key. Returns None to skip."""

    # Skip vision encoder and MTP layers
    if "visual" in hf_key or "mtp" in hf_key:
        return None

    # Embedding
    if hf_key == "model.embed_tokens.weight":
        return "embed_tokens.weight"

    # Final norm
    if hf_key == "model.norm.weight":
        return "norm.weight"

    # LM head (may not exist if tied)
    if hf_key == "lm_head.weight":
        return "lm_head.weight"

    # Handle the text_model prefix (Qwen3.5 wraps text in text_model)
    key = hf_key
    if key.startswith("model.text_model."):
        key = key.replace("model.text_model.", "", 1)
    elif key.startswith("model."):
        key = key.replace("model.", "", 1)

    # Embedding (alternative path)
    if key == "embed_tokens.weight":
        return "embed_tokens.weight"
    if key == "norm.weight":
        return "norm.weight"

    # Layer-level mappings
    # Pattern: layers.{i}.{module}.{param}
    layer_match = re.match(r"layers\.(\d+)\.(.*)", key)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)

        # Input/post-attention layernorm
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return f"layers.{layer_idx}.{rest}"

        # MLP
        if rest.startswith("mlp."):
            return f"layers.{layer_idx}.{rest}"

        # Self-attention (both full and linear share similar names)
        if rest.startswith("self_attn."):
            attn_rest = rest.replace("self_attn.", "")

            # Common projections
            known = [
                "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "o_proj.weight", "g_proj.weight",
                # DeltaNet specific
                "beta_proj.weight",
                "conv1d.weight", "conv1d.bias",
                "q_norm.weight", "k_norm.weight",
            ]
            if attn_rest in known:
                return f"layers.{layer_idx}.self_attn.{attn_rest}"

            # Catch any remaining attention params
            return f"layers.{layer_idx}.self_attn.{attn_rest}"

    return None


def load_hf_safetensors(hf_dir: Path) -> dict:
    """Load all safetensors files from a HuggingFace model directory."""
    if safe_open is None:
        raise ImportError("pip install safetensors")

    state_dict = {}
    for sf_file in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(str(sf_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def load_hf_pytorch(hf_dir: Path) -> dict:
    """Load pytorch_model.bin or pytorch_model-*.bin files."""
    state_dict = {}
    bin_files = sorted(hf_dir.glob("pytorch_model*.bin"))
    for bf in bin_files:
        sd = torch.load(bf, map_location="cpu", weights_only=True)
        state_dict.update(sd)
    return state_dict


def convert(hf_dir: str, output_path: str):
    hf_dir = Path(hf_dir)

    # Load HF weights
    if list(hf_dir.glob("*.safetensors")):
        print("Loading safetensors...")
        hf_state = load_hf_safetensors(hf_dir)
    else:
        print("Loading pytorch bin files...")
        hf_state = load_hf_pytorch(hf_dir)

    print(f"Loaded {len(hf_state)} tensors from HuggingFace model")

    # Map keys
    new_state = OrderedDict()
    skipped = []
    for hf_key, tensor in hf_state.items():
        our_key = map_hf_key(hf_key)
        if our_key is None:
            skipped.append(hf_key)
            continue
        new_state[our_key] = tensor

    print(f"Mapped {len(new_state)} tensors, skipped {len(skipped)}")
    if skipped:
        print(f"Skipped keys (first 10): {skipped[:10]}")

    # Save
    torch.save(new_state, output_path)
    print(f"Saved to {output_path}")

    # Also print some stats
    total_params = sum(p.numel() for p in new_state.values())
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen 3.5 HF weights to pure PyTorch")
    parser.add_argument("--hf-dir", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output", default="qwen35_weights.pt", help="Output path for converted weights")
    args = parser.parse_args()
    convert(args.hf_dir, args.output)


if __name__ == "__main__":
    main()
