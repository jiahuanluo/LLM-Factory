# coding=utf-8
# Copyright 2024 The GTE Team Authors and Alibaba Group.
# Licensed under the Apache License, Version 2.0 (the "License");
"""PyTorch NEW model (FA2/SDPA clean version).

相对官方原版的改动：
- 移除 unpad_inputs / use_memory_efficient_attention / pack_qkv 三个 flag
- 移除 IndexFirstAxis / IndexPutFirstAxis / unpad_input / pad_input 模板代码
- 移除 subset_indices 训练优化（padded 路径自然正确，无需 re-pad eval fix）
- NewAttention 改用 F.scaled_dot_product_attention（PyTorch ≥ 2.0，自动选择 FA2/mem-efficient/math 后端）

副作用：logits 始终是 [B, L, V]，Trainer.evaluate 自然对齐 padded labels，无需任何 eval fix。
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration import NewConfig

logger = logging.get_logger(__name__)

# === flash-attn 可选依赖 ===
# 装了 flash-attn + GPU sm_80+ → 走 FA2 varlen API（padded batch 也用 FA2，100% 命中）
# 没装 / V100 / CPU → fallback 到 SDPA（mem-efficient / math 后端）
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


# ============================================================
# RoPE
# ============================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=512, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # persistent=True：HF 5.x from_pretrained 走 meta device，非 persistent 的 buffer 会留在 meta 状态。
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=True)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=True)

    def forward(self, x, seq_len=None):
        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.is_meta
            or self.cos_cached.device != x.device
        ):
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with fixed and mixed NTK scaling. https://kexue.fm/archives/9706 """

    def __init__(self, dim, max_position_embeddings=512, base=10000, device=None,
                 scaling_factor=1.0, mixed_b=None):
        self.scaling_factor = scaling_factor
        self.mixed_b = mixed_b
        super().__init__(dim, max_position_embeddings, base, device)
        max_position_embeddings = max_position_embeddings * self.scaling_factor
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (self.scaling_factor if self.mixed_b is None else 1)
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            if self.mixed_b is None:
                inv_freq = inv_freq / self.scaling_factor ** (2 / self.dim)
            else:
                a = torch.tensor(self.scaling_factor).log() / (self.dim / 2) ** self.mixed_b
                lambda_1_m = (a * torch.arange(1, self.dim // 2 + 1).float().to(device) ** self.mixed_b).exp()
                inv_freq = inv_freq / lambda_1_m
            self.register_buffer("inv_freq", inv_freq, persistent=True)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=True)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=True)


# ============================================================
# Norm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # weight 是 fp32 Parameter，与 fp16 乘会 promote 到 fp32；统一转 input_dtype 避免下游 Q/K/V dtype 不一致
        return self.weight.to(input_dtype) * hidden_states.to(input_dtype)


LAYER_NORM = {
    'layer_norm': nn.LayerNorm,
    'rms_norm': RMSNorm,
}


# ============================================================
# Embeddings
# ============================================================

class NewEmbeddings(nn.Module):
    def __init__(self, config: NewConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == 'absolute':
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
        elif self.position_embedding_type == 'rope':
            self._init_rope(config)
        else:
            raise ValueError(f"Unknown position_embedding_type: {self.position_embedding_type}")

        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        _ln_class = LAYER_NORM[config.layer_norm_type]
        self.LayerNorm = _ln_class(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings), persistent=False
        )

    def _init_rope(self, config):
        head_dim = int(config.hidden_size / config.num_attention_heads)
        kwargs = dict(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        if config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(**kwargs)
        else:
            kwargs.update(scaling_factor=config.rope_scaling["factor"])
            scaling_type = config.rope_scaling.get("type") or config.rope_scaling.get("rope_type")
            if scaling_type == 'ntk':
                kwargs.update(mixed_b=config.rope_scaling.get('mixed_b', None))
                self.rotary_emb = NTKScalingRotaryEmbedding(**kwargs)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        if inputs_embeds is None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = torch.arange(seq_length, device=inputs_embeds.device if inputs_embeds is not None else input_ids.device)
            position_ids = position_ids.expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == 'rope':
            rope_cos, rope_sin = self.rotary_emb(inputs_embeds, seq_len=seq_length)
            rope_cos = rope_cos[position_ids].unsqueeze(2)
            rope_sin = rope_sin[position_ids].unsqueeze(2)
            rope_embeds = (rope_cos, rope_sin)
        else:
            rope_embeds = None

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = position_ids.mul(0)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, rope_embeds


# ============================================================
# Attention (SDPA only)
# ============================================================

class NewAttention(nn.Module):
    """Q/K/V 分开投影 + 双后端 attention。

    优先级：
      1. flash-attn 包 + CUDA + fp16/bf16 → flash_attn_varlen_func（padded batch 也跑 FA2，100% 命中）
      2. 否则 → F.scaled_dot_product_attention（PyTorch 自动选 mem-efficient / math）
    """

    def __init__(self, config: NewConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
            )
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.dropout_p = config.attention_probs_dropout_prob
        self.logn_attention_scale = config.logn_attention_scale
        self.logn_attention_clip1 = config.logn_attention_clip1
        self.softmax_scale = 1.0 / math.sqrt(self.attention_head_size)

    def _shape(self, x):
        return x.view(x.shape[:-1] + (self.num_attention_heads, self.attention_head_size))

    def _can_use_flash_attn(self, q: torch.Tensor) -> bool:
        """flash-attn 触发条件：包已装 + CUDA + fp16/bf16 + sm_80+。"""
        if not _FLASH_ATTN_AVAILABLE:
            return False
        if not q.is_cuda:
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        # sm_80+ 检查（flash-attn 内部会拒绝 sm_70，提前判断省得报错）
        major, _ = torch.cuda.get_device_capability(q.device)
        if major < 8:
            return False
        return True

    def _flash_attn_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """flash-attn 路径（已预计算 cu_seqlens/indices）。

        attn_metadata 由 NewModel.forward 预计算一次，所有层共享：
          {
            'is_padded': bool,
            'cu_seqlens': torch.int32 [B+1] or None,
            'indices': torch.long [total_valid] or None,
            'max_seqlen': int or None,
          }
        """
        B, L, H, D = q.shape

        if not attn_metadata['is_padded']:
            # 无 padding → 非 varlen（更省 cu_seqlens 开销）
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
            return out.reshape(B, L, H * D)

        # 有 padding → varlen，用预计算的 cu_seqlens/indices
        cu_seqlens = attn_metadata['cu_seqlens']
        indices = attn_metadata['indices']
        max_seqlen = attn_metadata['max_seqlen']

        q_flat = q.reshape(B * L, H, D)
        k_flat = k.reshape(B * L, H, D)
        v_flat = v.reshape(B * L, H, D)
        q_unpad = q_flat[indices]
        k_unpad = k_flat[indices]
        v_unpad = v_flat[indices]

        out_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
        )

        out_padded = q_flat.new_zeros(B * L, H * D)
        out_padded[indices] = out_unpad.reshape(-1, H * D)
        return out_padded.view(B, L, H * D)

    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        """SDPA fallback：V100/CPU 或没装 flash-attn 时走这里。"""
        B, L, H, D = q.shape
        # [B, L, H, D] -> [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = attn_metadata.get('sdpa_mask')  # 预计算的 bool [B, 1, L, L] or None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[dict] = None,
        rope_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        q = self._shape(self.q_proj(hidden_states))
        k = self._shape(self.k_proj(hidden_states))
        v = self._shape(self.v_proj(hidden_states))

        if self.config.position_embedding_type == 'rope':
            q, k = apply_rotary_pos_emb(q, k, *rope_embeds)

        if self.logn_attention_scale and attention_scale is not None:
            q = q * attention_scale.to(q.dtype)

        if attn_metadata is None:
            attn_metadata = {'is_padded': False, 'cu_seqlens': None, 'indices': None,
                             'max_seqlen': None, 'sdpa_mask': None}

        if self._can_use_flash_attn(q):
            context = self._flash_attn_forward(q, k, v, attn_metadata)
        else:
            context = self._sdpa_forward(q, k, v, attn_metadata)

        attn_output = self.o_proj(context)
        return (attn_output,)


class NewGatedMLP(nn.Module):
    """GLU Variants Improve Transformer."""

    def __init__(self, config: NewConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.up_gate_proj = nn.Linear(config.hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else None

    def forward(self, hidden_states):
        up_gate = self.up_gate_proj(hidden_states)
        up_states, gate = torch.split(up_gate, self.intermediate_size, dim=-1)
        gated_states = self.act_fn(gate) * up_states
        if self.hidden_dropout is not None:
            gated_states = self.hidden_dropout(gated_states)
        return self.down_proj(gated_states)


class NewLayer(nn.Module):
    def __init__(self, config: NewConfig):
        super().__init__()
        self.attention = NewAttention(config)
        self.mlp = NewGatedMLP(config)
        ln_class = LAYER_NORM[config.layer_norm_type]
        self.attn_ln = ln_class(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_ln = ln_class(config.hidden_size, eps=config.layer_norm_eps)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[dict] = None,
        rope_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        # Post-LN: residual on raw input, then LN
        residual = hidden_states
        attn_output = self.attention(
            hidden_states,
            attn_metadata=attn_metadata,
            rope_embeds=rope_embeds,
            attention_scale=attention_scale,
        )[0]
        if self.hidden_dropout is not None:
            attn_output = self.hidden_dropout(attn_output)
        hidden_states = residual + attn_output
        hidden_states = self.attn_ln(hidden_states)

        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        if self.hidden_dropout is not None:
            mlp_output = self.hidden_dropout(mlp_output)
        hidden_states = residual + mlp_output
        hidden_states = self.mlp_ln(hidden_states)

        return (hidden_states,)


class NewEncoder(nn.Module):
    def __init__(self, config: NewConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([NewLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[dict] = None,
        rope_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_scale: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attn_metadata,
                    rope_embeds,
                    attention_scale,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attn_metadata=attn_metadata,
                    rope_embeds=rope_embeds,
                    attention_scale=attention_scale,
                )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class NewPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# ============================================================
# PreTrainedModel + base
# ============================================================

class NewPreTrainedModel(PreTrainedModel):
    config_class = NewConfig
    base_model_prefix = "new"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class NewModel(NewPreTrainedModel):
    def __init__(self, config: NewConfig, add_pooling_layer=False):
        super().__init__(config)
        self.config = config
        self.embeddings = NewEmbeddings(config)
        self.encoder = NewEncoder(config)
        self.pooler = NewPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _build_attn_metadata(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, ...],
    ) -> dict:
        """预计算 attention 辅助数据，所有层共享。

        返回 dict 含：
          - is_padded: bool       — 是否有 padding（决定 FA2 走 varlen 还是普通）
          - cu_seqlens: [B+1] int32 — varlen 模式下样本边界（前缀和）
          - indices: [total_valid] long — valid token 在 [B*L] 中的位置
          - max_seqlen: int        — batch 内最长 valid 长度
          - sdpa_mask: bool [B, 1, L, L] or None — SDPA fallback 用
        """
        if attention_mask is None:
            return {'is_padded': False, 'cu_seqlens': None, 'indices': None,
                    'max_seqlen': None, 'sdpa_mask': None}

        # 一次性 GPU→CPU sync：判断是否全 1
        is_padded = not bool(attention_mask.all())
        if not is_padded:
            return {'is_padded': False, 'cu_seqlens': None, 'indices': None,
                    'max_seqlen': None, 'sdpa_mask': None}

        mask_bool = attention_mask.to(torch.bool)  # [B, L]
        valid_per_seq = mask_bool.sum(dim=1)  # [B]
        # cu_seqlens: [0, len_1, len_1+len_2, ..., total_valid]
        cu_seqlens = F.pad(valid_per_seq.cumsum(0).to(torch.int32), (1, 0), value=0)
        # 一次性 GPU→CPU sync：取 max
        max_seqlen = int(valid_per_seq.max().item())
        indices = torch.nonzero(mask_bool.flatten(), as_tuple=False).flatten()

        # SDPA fallback 用的 bool mask（仅当走 SDPA 时才用）
        sdpa_mask = mask_bool[:, None, None, :] & mask_bool[:, None, :, None]

        return {
            'is_padded': True,
            'cu_seqlens': cu_seqlens,
            'indices': indices,
            'max_seqlen': max_seqlen,
            'sdpa_mask': sdpa_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        embedding_output, rope_embeds = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # === 预计算 attention metadata，所有层共享 ===
        # 把 cu_seqlens/indices/max_seqlen/bool_mask 在 forward 开始时算一次，
        # 避免 NewAttention 每层都重算（24 层 × 每层 GPU→CPU sync 会慢一倍）
        attn_metadata = self._build_attn_metadata(attention_mask, input_shape)

        attention_scale = None
        if self.config.logn_attention_scale:
            seq_len = input_shape[-1]
            if seq_len > 8192:
                logn = math.log(seq_len / 8192) + 1.0
                if self.config.logn_attention_clip1:
                    logn = max(logn, 1.0)
                attention_scale = torch.tensor(logn)

        encoder_outputs = self.encoder(
            embedding_output,
            attn_metadata=attn_metadata,
            rope_embeds=rope_embeds,
            attention_scale=attention_scale,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return tuple(v for v in [sequence_output, pooled_output, encoder_outputs.hidden_states] if v is not None)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=None,
        )


# ============================================================
# Heads
# ============================================================

class NewLMPredictionHead(nn.Module):
    def __init__(self, config: NewConfig):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            ACT2FN[config.hidden_act],
            LAYER_NORM[config.layer_norm_type](config.hidden_size, eps=config.layer_norm_eps),
        )
        # decoder.weight 与 word_embeddings 共享（通过 NewForMaskedLM._tied_weights_keys 处理）
        # decoder.bias 是独立参数，不要再 alias 到 self.bias 否则 save 时报 shared tensors 错误
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class NewForMaskedLM(NewPreTrainedModel):
    _tied_weights_keys = {"lm_head.decoder.weight": "new.embeddings.word_embeddings.weight"}

    def __init__(self, config: NewConfig):
        super().__init__(config)
        self.new = NewModel(config, add_pooling_layer=False)
        self.lm_head = NewLMPredictionHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.new(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [B, L, H]
        prediction_scores = self.lm_head(sequence_output)  # [B, L, V]

        masked_lm_loss = None
        if labels is not None:
            # 标准 MLM loss：labels 中 -100 位置忽略，logits 保持原始 [B, L, V] 形状返回
            mask = labels != -100
            loss_scores = prediction_scores[mask]            # [num_masked, V]
            loss_labels = labels[mask]                       # [num_masked]
            masked_lm_loss = self.loss_fct(loss_scores, loss_labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NewForSequenceClassification(NewPreTrainedModel):
    def __init__(self, config: NewConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.new = NewModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.new(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze().float())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NewForTokenClassification(NewPreTrainedModel):
    def __init__(self, config: NewConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.new = NewModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.new(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
