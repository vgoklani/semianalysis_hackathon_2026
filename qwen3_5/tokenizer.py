"""
Pure tiktoken-based tokenizer for Qwen 3.5.

Loads the HuggingFace vocab.json + merges.txt and converts them into
a tiktoken Encoding. No HuggingFace dependencies required.
"""

import json
import base64
import os
import re
from pathlib import Path
from typing import Optional

import tiktoken


# The pretokenization regex used by Qwen2/Qwen3 tokenizers (same as GPT-4o style)
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Special tokens for Qwen 3.5
SPECIAL_TOKENS = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047,
    "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|quad_start|>": 248051,
    "<|quad_end|>": 248052,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|vision_pad|>": 248055,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
    "<tool_call>": 248058,
    "</tool_call>": 248059,
    "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061,
    "<|fim_suffix|>": 248062,
    "<|fim_pad|>": 248063,
    "<|repo_name|>": 248064,
    "<|file_sep|>": 248065,
    "<tool_response>": 248066,
    "</tool_response>": 248067,
    "<think>": 248068,
    "</think>": 248069,
    "<|audio_start|>": 248070,
    "<|audio_end|>": 248071,
    "<tts_pad>": 248072,
    "<tts_text_bos>": 248073,
    "<tts_text_eod>": 248074,
    "<tts_text_bos_single>": 248075,
    "<|audio_pad|>": 248076,
}


class Qwen35Tokenizer:
    """
    Tokenizer for Qwen 3.5 using tiktoken as the BPE engine.

    Usage:
        tokenizer = Qwen35Tokenizer.from_pretrained("path/to/tokenizer/dir")
        ids = tokenizer.encode("Hello, world!")
        text = tokenizer.decode(ids)
    """

    def __init__(self, encoding: tiktoken.Encoding):
        self.encoding = encoding
        self.eos_token_id = SPECIAL_TOKENS["<|im_end|>"]
        self.pad_token_id = SPECIAL_TOKENS["<|endoftext|>"]
        self.special_tokens = SPECIAL_TOKENS

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str) -> "Qwen35Tokenizer":
        """
        Load from a directory containing vocab.json and merges.txt
        (downloaded from HuggingFace).
        """
        tokenizer_dir = Path(tokenizer_dir)
        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                f"Need vocab.json and merges.txt in {tokenizer_dir}. "
                f"Download from https://huggingface.co/Qwen/Qwen3.5-0.8B"
            )

        # Load vocab: token_string -> id
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load merges
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Skip header line if present
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        merges = [line.strip() for line in lines if line.strip()]

        # Build tiktoken-compatible BPE ranks
        # tiktoken expects: {token_bytes: rank}
        # The rank ordering must be: first all single-byte tokens in vocab order,
        # then merge results in merge order.
        bpe_ranks = {}
        rank = 0

        # First: add all single-byte tokens from vocab
        byte_tokens = {}
        for token_str, token_id in vocab.items():
            token_bytes = token_str.encode("utf-8")
            if len(token_bytes) == 1:
                byte_tokens[token_bytes] = token_id

        # Add byte-level tokens sorted by their vocab id
        for token_bytes in sorted(byte_tokens, key=lambda b: byte_tokens[b]):
            bpe_ranks[token_bytes] = rank
            rank += 1

        # Then: add merge results in order
        for merge in merges:
            parts = merge.split()
            if len(parts) == 2:
                token_bytes = (parts[0] + parts[1]).encode("utf-8")
                if token_bytes not in bpe_ranks:
                    bpe_ranks[token_bytes] = rank
                    rank += 1

        # Build the tiktoken encoding
        # Convert to base64 format for tiktoken
        mergeable_ranks = {}
        for token_bytes, r in bpe_ranks.items():
            mergeable_ranks[token_bytes] = r

        encoding = tiktoken.Encoding(
            name="qwen35",
            pat_str=PRETOKENIZE_REGEX,
            mergeable_ranks=mergeable_ranks,
            special_tokens=SPECIAL_TOKENS,
        )

        return cls(encoding)

    @classmethod
    def from_tiktoken_cache(cls, cache_path: str) -> "Qwen35Tokenizer":
        """
        Load from a pre-built tiktoken cache file (base64 token + rank per line).
        """
        mergeable_ranks = {}
        with open(cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank_str = line.split()
                token_bytes = base64.b64decode(token_b64)
                mergeable_ranks[token_bytes] = int(rank_str)

        encoding = tiktoken.Encoding(
            name="qwen35",
            pat_str=PRETOKENIZE_REGEX,
            mergeable_ranks=mergeable_ranks,
            special_tokens=SPECIAL_TOKENS,
        )
        return cls(encoding)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        if add_special_tokens:
            return self.encoding.encode(
                text, allowed_special=set(SPECIAL_TOKENS.keys())
            )
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            special_ids = set(SPECIAL_TOKENS.values())
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.encoding.decode(token_ids)

    def encode_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        """
        Encode a chat conversation in ChatML format.

        messages: [{"role": "user", "content": "Hello"}, ...]
        """
        tokens = []
        im_start = SPECIAL_TOKENS["<|im_start|>"]
        im_end = SPECIAL_TOKENS["<|im_end|>"]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            tokens.append(im_start)
            tokens.extend(self.encode(f"{role}\n{content}"))
            tokens.append(im_end)
            tokens.extend(self.encode("\n"))

        if add_generation_prompt:
            tokens.append(im_start)
            tokens.extend(self.encode("assistant\n"))

        return tokens

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab
