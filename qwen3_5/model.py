"""
Pure PyTorch implementation of Qwen 3.5-0.8B (hybrid linear/full attention).

Architecture:
- 24 layers: pattern [linear, linear, linear, full] x 6
- Linear layers: Gated DeltaNet (linear attention with delta rule)
- Full layers: Grouped Query Attention (8 Q heads, 2 KV heads, head_dim=256)
- SwiGLU FFN (intermediate_size=3584)
- RMSNorm, RoPE (partial rotary 25%, theta=10M)
- Attention output gating
- Tied embeddings (embed == lm_head)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen35Config:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    # Full attention params
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    # Linear attention (DeltaNet) params
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # FFN
    intermediate_size: int = 3584
    hidden_act: str = "silu"
    # Vocab / positions
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    # RoPE
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    # Norm
    rms_norm_eps: float = 1e-6
    # Misc
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    full_attention_interval: int = 4
    # Generation
    eos_token_id: int = 248046
    pad_token_id: int = 248044

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def layer_type(self, layer_idx: int) -> str:
        if (layer_idx + 1) % self.full_attention_interval == 0:
            return "full"
        return "linear"


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10_000_000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
    """Apply RoPE to the rotary dimensions of x. x shape: (B, n_heads, S, head_dim)."""
    rot_dim = cos.shape[-1] * 2
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        seq_len = x.shape[2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    x_rot_out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    x_rot_out = x_rot_out.flatten(-2)
    return torch.cat([x_rot_out, x_pass], dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FullAttention(nn.Module):
    """Grouped Query Attention with output gating and partial RoPE."""

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rotary_dim = config.rotary_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if config.attn_output_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        else:
            self.g_proj = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin, position_ids)
        k = apply_rotary_emb(k, cos, sin, position_ids)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)

        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(x))
            attn_output = attn_output * gate

        return self.o_proj(attn_output), new_kv_cache


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer.

    The delta rule updates a recurrent state matrix S:
        S_t = alpha_t * S_t-1 + beta_t * (v_t outer k_t)
    where alpha is a forget gate and beta is an input gate.

    Output: o_t = S_t @ q_t

    For prefill, we use a chunkwise parallel form.
    For decoding, we use the recurrent form.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(config.hidden_size, self.num_key_heads * self.key_head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_heads * self.key_head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_value_heads * self.value_head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_value_heads * self.value_head_dim, config.hidden_size, bias=False)

        # Beta (input gate) projection
        self.beta_proj = nn.Linear(config.hidden_size, self.num_key_heads, bias=False)

        # Short convolution on q, k, v
        total_conv_dim = (self.num_key_heads * self.key_head_dim +
                          self.num_key_heads * self.key_head_dim +
                          self.num_value_heads * self.value_head_dim)
        self.conv1d = nn.Conv1d(
            total_conv_dim,
            total_conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=total_conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Output gate
        if config.attn_output_gate:
            self.g_proj = nn.Linear(config.hidden_size, self.num_value_heads * self.value_head_dim, bias=False)
        else:
            self.g_proj = None

        # Layer norms for q and k
        self.q_norm = RMSNorm(self.key_head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.key_head_dim, eps=config.rms_norm_eps)

    def _apply_conv(self, qkv: torch.Tensor, conv_state: Optional[torch.Tensor] = None):
        """Apply causal depthwise conv1d. qkv shape: (B, S, D)."""
        B, S, D = qkv.shape
        if conv_state is not None:
            qkv_with_state = torch.cat([conv_state, qkv], dim=1)
            qkv_conv = self.conv1d(qkv_with_state.transpose(1, 2))[..., :S].transpose(1, 2)
            new_conv_state = qkv_with_state[:, -(self.conv_kernel_size - 1):, :]
        else:
            qkv_conv = self.conv1d(qkv.transpose(1, 2))[..., :S].transpose(1, 2)
            if S >= self.conv_kernel_size - 1:
                new_conv_state = qkv[:, -(self.conv_kernel_size - 1):, :]
            else:
                new_conv_state = F.pad(qkv, (0, 0, self.conv_kernel_size - 1 - S, 0))
        return F.silu(qkv_conv), new_conv_state

    def _recurrent_forward(self, q, k, v, beta, recurrent_state):
        """
        Recurrent DeltaNet for single-step decoding.
        q: (B, H_k, 1, D_k)
        k: (B, H_k, 1, D_k)
        v: (B, H_v, 1, D_v)
        beta: (B, H_k, 1, 1)
        recurrent_state: (B, H_k, D_v, D_k) or None
        """
        B = q.shape[0]
        k = k.squeeze(2)  # (B, H_k, D_k)
        v = v.squeeze(2)  # (B, H_v, D_v)
        q = q.squeeze(2)  # (B, H_k, D_k)
        beta = beta.squeeze(2).squeeze(-1)  # (B, H_k)

        # Normalize k to unit norm for stability
        k = F.normalize(k, p=2, dim=-1)

        if recurrent_state is None:
            recurrent_state = torch.zeros(B, self.num_key_heads, self.value_head_dim, self.key_head_dim,
                                          device=q.device, dtype=q.dtype)

        # Delta rule update:
        # S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T @ k_t) outer k_t
        # This corrects the memory using the delta (error) signal
        Sk = torch.einsum('bhvk,bhk->bhv', recurrent_state, k)  # (B, H, D_v)
        delta = v - Sk  # (B, H, D_v)
        beta_expanded = beta.unsqueeze(-1)  # (B, H, 1)
        update = torch.einsum('bhv,bhk->bhvk', beta_expanded * delta, k)  # (B, H, D_v, D_k)
        recurrent_state = recurrent_state + update

        # Output: o_t = S_t @ q_t
        output = torch.einsum('bhvk,bhk->bhv', recurrent_state, q)  # (B, H, D_v)
        output = output.unsqueeze(2)  # (B, H, 1, D_v)

        return output, recurrent_state

    def _parallel_forward(self, q, k, v, beta):
        """
        Parallel DeltaNet for prefill (full sequence).
        q: (B, H_k, S, D_k)
        k: (B, H_k, S, D_k)
        v: (B, H_v, S, D_v)
        beta: (B, H_k, S, 1)

        Uses the materialized form for moderate sequences.
        For very long sequences, a chunkwise algorithm would be better.
        """
        B, H, S, D_k = q.shape
        D_v = v.shape[-1]

        # Normalize k
        k = F.normalize(k, p=2, dim=-1)

        # For the parallel form, we compute the full causal linear attention
        # with delta rule corrections applied sequentially.
        # This is O(S^2) but simpler than the chunkwise O(S*C) algorithm.

        # Simple sequential approach for correctness
        # (For production, replace with chunkwise parallel algorithm)
        output = torch.zeros(B, H, S, D_v, device=q.device, dtype=q.dtype)
        state = torch.zeros(B, H, D_v, D_k, device=q.device, dtype=q.dtype)

        for t in range(S):
            k_t = k[:, :, t, :]  # (B, H, D_k)
            v_t = v[:, :, t, :]  # (B, H, D_v)
            q_t = q[:, :, t, :]  # (B, H, D_k)
            beta_t = beta[:, :, t, :]  # (B, H, 1)

            # Delta rule: S = S + beta * (v - S^T k) outer k
            Sk = torch.einsum('bhvk,bhk->bhv', state, k_t)
            delta = v_t - Sk
            update = torch.einsum('bhv,bhk->bhvk', beta_t * delta, k_t)
            state = state + update

            # Output
            o_t = torch.einsum('bhvk,bhk->bhv', state, q_t)
            output[:, :, t, :] = o_t

        return output, state

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ):
        B, S, _ = x.shape

        # Project q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Concatenate for convolution
        qkv = torch.cat([q, k, v], dim=-1)
        qkv, new_conv_state = self._apply_conv(qkv, conv_state)

        # Split back
        q_dim = self.num_key_heads * self.key_head_dim
        k_dim = self.num_key_heads * self.key_head_dim
        q, k, v = qkv.split([q_dim, k_dim, self.num_value_heads * self.value_head_dim], dim=-1)

        # Reshape to heads
        q = q.view(B, S, self.num_key_heads, self.key_head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_key_heads, self.key_head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_value_heads, self.value_head_dim).transpose(1, 2)

        # Apply head norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Beta (input gate)
        beta = torch.sigmoid(self.beta_proj(x))  # (B, S, H_k)
        beta = beta.transpose(1, 2).unsqueeze(-1)  # (B, H_k, S, 1)

        # Apply DeltaNet
        if S == 1:
            attn_output, new_recurrent_state = self._recurrent_forward(q, k, v, beta, recurrent_state)
        else:
            attn_output, new_recurrent_state = self._parallel_forward(q, k, v, beta)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)

        # Output gate
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(x))
            attn_output = attn_output * gate

        output = self.o_proj(attn_output)
        cache = (new_conv_state, new_recurrent_state)
        return output, cache


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.layer_type == "full":
            self.self_attn = FullAttention(config)
        else:
            self.self_attn = GatedDeltaNet(config)

        self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_cache: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = x
        x_norm = self.input_layernorm(x)

        if self.layer_type == "full":
            kv_cache = layer_cache if layer_cache is not None else None
            attn_out, new_cache = self.self_attn(
                x_norm, cos, sin, position_ids, kv_cache, attention_mask
            )
        else:
            conv_state = layer_cache[0] if layer_cache is not None else None
            recurrent_state = layer_cache[1] if layer_cache is not None else None
            attn_out, new_cache = self.self_attn(x_norm, conv_state, recurrent_state)

        x = residual + attn_out

        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))

        return x, new_cache


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE
        self._rope_cos = None
        self._rope_sin = None

    def _get_rope(self, device, seq_len=4096):
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            alloc_len = max(seq_len, 4096)
            self._rope_cos, self._rope_sin = precompute_rope(
                self.config.rotary_dim, alloc_len, self.config.rope_theta, device
            )
        return self._rope_cos.to(device), self._rope_sin.to(device)

    def _make_causal_mask(self, seq_len: int, start_pos: int, device, dtype):
        if seq_len == 1:
            return None
        total_len = start_pos + seq_len
        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=start_pos + 1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[list] = None,
    ):
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)

        start_pos = 0
        if cache is not None and cache[0] is not None:
            layer_type_0 = self.config.layer_type(0)
            if layer_type_0 == "full":
                k_cache = cache[0][0]
                start_pos = k_cache.shape[2] if k_cache is not None else 0
            # For linear layers, position tracking is implicit in recurrent state
            # Find first full attention layer to get position
            for i in range(self.config.num_hidden_layers):
                if self.config.layer_type(i) == "full" and cache[i] is not None:
                    start_pos = cache[i][0].shape[2]
                    break

        cos, sin = self._get_rope(x.device, start_pos + S)
        attention_mask = self._make_causal_mask(S, start_pos, x.device, x.dtype)

        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + S, device=x.device).unsqueeze(0).expand(B, -1)

        new_cache = [None] * self.config.num_hidden_layers

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(
                x, cos, sin, position_ids, layer_cache, attention_mask
            )
            new_cache[i] = layer_new_cache

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        """Simple autoregressive generation."""
        cache = [None] * self.config.num_hidden_layers

        # Prefill
        logits, cache = self.forward(input_ids, cache=cache)
        next_token_logits = logits[:, -1, :]

        generated = []
        for _ in range(max_new_tokens):
            # Sample
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits[next_token_logits < top_k_vals[:, -1:]] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = float("-inf")
                    next_token_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)

            if next_token.item() == self.config.eos_token_id:
                break

            # Decode step
            logits, cache = self.forward(next_token, cache=cache)
            next_token_logits = logits[:, -1, :]

        return torch.cat(generated, dim=-1)
