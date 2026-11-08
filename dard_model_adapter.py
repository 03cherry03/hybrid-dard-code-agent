"""Per-instance, inference-only rectangular attention for Dream DARD layouts.

The packed decoder places all usable context keys before predictor-only rows.
A rectangular boolean mask opts into projecting K/V from that context prefix.
Square masks and native generation are delegated to the original implementation.
No weights, global model classes, or cross-step caches are changed.

The projection/RoPE/GQA order follows Apple's public DreamSdpaAttention in
DiffuCoder revision 4fdd4580064ca5d11808069ce78f88d068753c96. This optimization
is independently implemented; BF16 kernel/shape changes need measured validation.
"""
from __future__ import annotations

import torch
from torch import nn


ADAPTER_VERSION = "dream_context_only_kv_v1"


def _rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(value, repeats):
    if repeats == 1:
        return value
    batch, heads, length, dimension = value.shape
    return value[:, :, None].expand(batch, heads, repeats, length, dimension).reshape(
        batch, heads * repeats, length, dimension)


class DreamRectangularAttention(nn.Module):
    """Keep native attention unchanged; rectangular masks select context-only K/V."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.train(original.training)

    @property
    def rotary_emb(self):
        # DreamModel.reset_rope_parameters() accesses this public attribute.
        return self.original.rotary_emb

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None):
        rectangular = (isinstance(attention_mask, torch.Tensor)
                       and attention_mask.ndim == 4
                       and attention_mask.shape[-1] != hidden_states.shape[1])
        if not rectangular:
            return self.original(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings)
        if self.training or self.original.training:
            raise ValueError("Rectangular Dream attention is inference-only")
        if use_cache or past_key_value is not None:
            raise ValueError("DARD rectangular attention cannot reuse a KV cache")
        if output_attentions:
            raise ValueError("DARD rectangular attention does not expose attention weights")
        attention = self.original
        batch, query_length, _ = hidden_states.shape
        key_length = attention_mask.shape[-1]
        if (attention_mask.dtype != torch.bool or not 0 < key_length < query_length
                or attention_mask.shape[-2] != query_length
                or attention_mask.shape[0] not in (1, batch)
                or attention_mask.shape[1] not in (1, attention.num_heads)):
            raise ValueError("Expected boolean [batch, heads, Q, K] mask with 0 < K < Q")

        # Query-only rows never contribute keys at any layer. Their K/V
        # projections can therefore be omitted without changing the graph.
        context = hidden_states[:, :key_length]
        query = attention.q_proj(hidden_states).view(
            batch, query_length, attention.num_heads, attention.head_dim).transpose(1, 2)
        key = attention.k_proj(context).view(
            batch, key_length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        value = attention.v_proj(context).view(
            batch, key_length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)

        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("Rectangular Dream attention requires explicit RoPE positions")
            cos, sin = attention.rotary_emb(value, position_ids)
        else:
            cos, sin = position_embeddings
        if cos.shape[-2] != query_length or sin.shape != cos.shape:
            raise ValueError("RoPE embeddings must cover every query position")
        query = query * cos[:, None] + _rotate_half(query) * sin[:, None]
        key = key * cos[:, None, :key_length] + _rotate_half(key) * sin[:, None, :key_length]
        key = _repeat_kv(key, attention.num_key_value_groups)
        value = _repeat_kv(value, attention.num_key_value_groups)
        if query.device.type == "cuda":
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, attention.hidden_size)
        return attention.o_proj(output), None, None


def install_dream_rectangular_attention(model):
    """Install on one loaded Dream instance, validating all layers before mutation.

    Native model forward hooks remain attached to the same outer model, and
    ordinary native generation still calls the original attention modules.
    The returned receipt can be included in model metadata. Installation is
    idempotent; state-dict paths change while installed, so uninstall before save.
    """
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None or len(layers) == 0:
        raise TypeError("Rectangular attention adapter requires Dream model.layers")
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb", "hidden_size",
                "num_heads", "num_key_value_heads", "num_key_value_groups", "head_dim")
    for layer in layers:
        module = getattr(layer, "self_attn", None)
        original = module.original if isinstance(module, DreamRectangularAttention) else module
        if (type(original).__name__ != "DreamSdpaAttention"
                or any(not hasattr(original, field) for field in required)
                or original.head_dim % 2
                or original.num_heads != original.num_key_value_heads * original.num_key_value_groups):
            raise TypeError("Only compatible DreamSdpaAttention layers are supported")
    for layer in layers:
        if not isinstance(layer.self_attn, DreamRectangularAttention):
            layer.self_attn = DreamRectangularAttention(layer.self_attn)
    model._dard_rectangular_attention = True
    return {"version": ADAPTER_VERSION, "layers": len(layers),
            "native_attention_delegated": True, "cross_step_kv_cache": False}


def uninstall_dream_rectangular_attention(model):
    """Restore original attention modules and original state-dict parameter paths."""
    for layer in getattr(getattr(model, "model", None), "layers", ()):
        if isinstance(layer.self_attn, DreamRectangularAttention):
            layer.self_attn = layer.self_attn.original
    model._dard_rectangular_attention = False
