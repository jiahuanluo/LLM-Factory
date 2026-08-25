"""PbcCreditModel：user + accounts 双模态 Model 1 变体（SQL 管道数据）。

模态：
  - user（个人信息，固定 32 numeric + 14 cat）
  - accounts（6 类账户：d1/r1/r2/r3/r4/c1，变长；每账户 13 numeric + 13 cat + 60 月 paystate）

交互：
  - int_aa：d1 × r2（非循环贷 × 贷记卡）

顶层：8 个 pooled（user, d1, r1, r2, r3, r4, c1, int_aa）+ [CLS] → TopTrunk。

数据由 SQL 端（mvp_user_d1.sql）产出，cat / paystate encode 均已完成；
后续 SQL 补齐 query / public 模态后再加回对应分支。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .fields import PAYSTATE_VOCAB_SIZE, ACCOUNT_TYPES


# ============================================================
# Config
# ============================================================

@dataclass
class PbcCreditModelConfig:
    d: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    top_hidden: int = 512
    # 顶层 trunk（替代 concat MLP，让 pooled 互相关注）
    top_n_layers: int = 2
    top_n_heads: int = 8

    # user（32 numeric + 14 cat）
    user_numeric_dim: int = 32
    user_cat_tables: dict = field(default_factory=lambda: {})

    # account（13 numeric + 13 cat + 60 月 paystate；6 类账户共享同一 encoder）
    account_numeric_dim: int = 13
    account_cat_tables: dict = field(default_factory=dict)
    paystate_vocab_size: int = PAYSTATE_VOCAB_SIZE


# ============================================================
# Building blocks（复用 home_credit 模式）
# ============================================================

class CategoricalEmbedding(nn.Module):
    def __init__(self, field_table_sizes: dict, embed_dim: int):
        super().__init__()
        # field_table_sizes: {field_name: vocab_size}
        self.fields = list(field_table_sizes.keys())
        self.embeds = nn.ModuleDict({
            name: nn.Embedding(size + 1, embed_dim)  # +1 for UNK safety
            for name, size in field_table_sizes.items()
        })
        self.out_dim = len(self.fields) * embed_dim

    def forward(self, cat_ids: torch.Tensor, cat_mask: torch.Tensor | None = None) -> torch.Tensor:
        """cat_ids: [..., F] long → [..., F*embed_dim]."""
        outs = []
        for i, name in enumerate(self.fields):
            emb = self.embeds[name](cat_ids[..., i].clamp_min(0))  # [..., embed_dim]
            if cat_mask is not None:
                m = cat_mask[..., i].unsqueeze(-1).float()
                emb = emb * m
            outs.append(emb)
        return torch.cat(outs, dim=-1)


class SeqEncoder(nn.Module):
    """对 [B, N, in_dim] 编码到 [B, d] pooled 和 [B, N, d] tokens。"""

    def __init__(self, in_dim: int, d: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d = d

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features: [B, N, in_dim], mask: [B, N].
        Returns: pooled [B, d], tokens [B, N, d].
        """
        if features.shape[1] == 0:
            B = features.shape[0]
            return torch.zeros(B, self.d, device=features.device), \
                   torch.zeros(B, 0, self.d, device=features.device)

        h = self.input_norm(features)
        h = self.proj(h)
        pad_mask = (mask == 0)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled, h


class FixedEncoder(nn.Module):
    """对固定维度向量编码到 d。"""

    def __init__(self, in_dim: int, d: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class PayStateEncoder(nn.Module):
    """把每行 60 月 paystate id 序列编码到 d。

    输入: [B, N, 60] long
    输出: pooled [B, N, d], per_month [B, N, 60, d]
    """

    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int, dropout: float,
                 max_len: int = 60):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, d))  # learnable month position
        nn.init.normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d = d

    def forward(self, paystate: torch.Tensor, acct_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """paystate: [B, N, 60], acct_mask: [B, N].

        Returns:
          pooled: [B, N, d] per-account paystate pooled vector
          per_month: [B, N, 60, d] transformer contextualized per-month tokens (pre-pool)
        """
        B, N, L = paystate.shape
        if N == 0:
            return (torch.zeros(B, 0, self.d, device=paystate.device),
                    torch.zeros(B, 0, L, self.d, device=paystate.device))
        x = paystate.reshape(B * N, L)  # [B*N, 60]
        emb = self.embedding(x) + self.pos_emb  # [B*N, 60, d]
        pad_mask = (x == 0)  # [B*N, 60]
        h = self.transformer(emb, src_key_padding_mask=pad_mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        # pool over 60 月（忽略 pad）
        m = (x != 0).unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B*N, d]
        return pooled.reshape(B, N, self.d), h.reshape(B, N, L, self.d)


class InteractiveModule(nn.Module):
    """A × B cross-attention: q=A attend to kv=B, pool over A."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, q_tokens, q_mask, kv_tokens, kv_mask) -> torch.Tensor:
        """q_tokens/kv_tokens: [B, N, d]. Returns [B, d] (pooled over q)."""
        if q_tokens.shape[1] == 0 or kv_tokens.shape[1] == 0:
            return torch.zeros(q_tokens.shape[0], q_tokens.shape[2],
                               device=q_tokens.device)
        kv_pad_mask = (kv_mask == 0)
        attn_out, _ = self.cross_attn(q_tokens, kv_tokens, kv_tokens,
                                       key_padding_mask=kv_pad_mask,
                                       need_weights=False)
        h = self.norm(q_tokens + attn_out)
        h = self.norm2(h + self.ffn(h))
        m = q_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class TopTrunk(nn.Module):
    """n_modality 个模态 pooled + [CLS] → transformer → CLS → head。

    替代 concat MLP：模态间双向 attention；r1/r3/r4/c1 也参与交互。
    """

    def __init__(self, d: int, n_modality: int = 8,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, top_hidden: int = 256):
        super().__init__()
        self.n_modality = n_modality
        # learnable CLS token（独立于 modality poolings）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        # 位置 / 模态类型 embedding：[CLS, user, d1, r1, ..., int_aa]
        self.pos_emb = nn.Parameter(torch.zeros(n_modality + 1, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d, top_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(top_hidden, 1),
        )
        self.d = d

    def forward(self, pooled_list: list[torch.Tensor]) -> torch.Tensor:
        """pooled_list: 长度 n_modality 的 [B, d] tensor 列表。Returns [B, 1]."""
        B = pooled_list[0].shape[0]
        x = torch.stack(pooled_list, dim=1)  # [B, n_modality, d]
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)  # [B, n_modality+1, d]
        x = x + self.pos_emb.unsqueeze(0)
        h = self.transformer(x)
        cls_out = h[:, 0]  # [B, d]
        return self.head(cls_out)


# ============================================================
# Main model
# ============================================================

_ACCOUNT_TYPES_LOWER = [t.lower() for t in ACCOUNT_TYPES]


class PbcCreditModel(nn.Module):
    def __init__(self, config: PbcCreditModelConfig, pretrain_mode: bool = False):
        super().__init__()
        self.config = config
        self.pretrain_mode = pretrain_mode

        d = config.d

        # === user ===
        user_cat_sizes = config.user_cat_tables  # {field_name: vocab_size}
        self.user_cat_emb = CategoricalEmbedding(user_cat_sizes, embed_dim=8)
        self.user_encoder = FixedEncoder(
            config.user_numeric_dim + self.user_cat_emb.out_dim, d, config.dropout,
        )

        # === accounts (shared by 6 types: d1/r1/r2/r3/r4/c1) ===
        acc_cat_sizes = config.account_cat_tables
        self.acc_cat_emb = CategoricalEmbedding(acc_cat_sizes, embed_dim=4)
        self.acc_seq_encoder = SeqEncoder(
            config.account_numeric_dim + self.acc_cat_emb.out_dim,
            d, config.n_heads, config.n_layers, config.dropout,
        )
        self.paystate_encoder = PayStateEncoder(
            config.paystate_vocab_size, d, config.n_heads, n_layers=1,
            dropout=config.dropout,
        )
        # account 最终 token = seq_token + paystate_token
        self.acc_fuse = nn.Linear(d * 2, d)

        # === interactive pair ===
        self.int_aa = InteractiveModule(d, config.n_heads, config.dropout)  # d1 × r2

        # === top (finetune) ===
        # 8 pooled vectors: user, d1, r1, r2, r3, r4, c1, int_aa
        self.top = TopTrunk(
            d=d, n_modality=8,
            n_heads=config.top_n_heads, n_layers=config.top_n_layers,
            dropout=config.dropout, top_hidden=config.top_hidden,
        )

        # === pretrain mask heads ===
        if pretrain_mode:
            self.user_mask_head = nn.Linear(d, config.user_numeric_dim)
            self.acc_mask_head = nn.Linear(d, config.account_numeric_dim)
            self.paystate_mask_head = nn.Linear(d, config.paystate_vocab_size)

    def _encode_accounts(self, batch: dict):
        """对 6 类账户（d1/r1/r2/r3/r4/c1）共享同一套 encoder。

        返回 dict[type] = (pooled, tokens, pay_per_month).
        """
        results = {}
        for t in _ACCOUNT_TYPES_LOWER:
            numeric = batch[f'{t}_numeric']  # [B, N, Fn]
            cat_ids = batch[f'{t}_cat_ids']
            paystate = batch[f'{t}_paystate']
            mask = batch[f'{t}_mask']  # [B, N]

            B, N = numeric.shape[:2]
            if N == 0:
                pooled = torch.zeros(B, self.config.d, device=numeric.device)
                tokens = torch.zeros(B, 0, self.config.d, device=numeric.device)
                pay_per_month = torch.zeros(B, 0, paystate.shape[-1], self.config.d,
                                            device=numeric.device)
            else:
                cat_flat = self.acc_cat_emb(cat_ids)
                feats = torch.cat([numeric, cat_flat], dim=-1)
                seq_pooled, seq_tokens = self.acc_seq_encoder(feats, mask)
                pay_pooled, pay_per_month = self.paystate_encoder(paystate, mask)
                tokens = self.acc_fuse(torch.cat([seq_tokens, pay_pooled], dim=-1))
                # re-mask pad positions (fuse may leak)
                m = mask.unsqueeze(-1).float()
                tokens = tokens * m
                pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            results[t] = (pooled, tokens, pay_per_month)
        return results

    def forward(self, batch: dict):
        # === encode branches ===
        user_cat = self.user_cat_emb(batch['user_cat_ids'])
        user_h = self.user_encoder(torch.cat([batch['user_numeric'], user_cat], dim=-1))

        accs = self._encode_accounts(batch)

        if self.pretrain_mode:
            return self._forward_pretrain(batch, accs, user_h)

        # === interactive ===
        d1_p, d1_t, _ = accs['d1']
        r1_p, r1_t, _ = accs['r1']
        r2_p, r2_t, _ = accs['r2']
        r3_p, r3_t, _ = accs['r3']
        r4_p, r4_t, _ = accs['r4']
        c1_p, c1_t, _ = accs['c1']

        int_aa = self.int_aa(d1_t, batch['d1_mask'], r2_t, batch['r2_mask'])

        # === top trunk: 8 个 pooled 作为 token 序列 + [CLS] ===
        pooled_list = [user_h, d1_p, r1_p, r2_p, r3_p, r4_p, c1_p, int_aa]
        logit = self.top(pooled_list)
        return logit

    def _forward_pretrain(self, batch, accs, user_h):
        out = {}
        # 暴露 user_emb 给 contrastive consistency loss 用
        out['user_emb'] = user_h  # [B, d]

        # user numeric reconstruction（[B, 18] 按特征位 mask）
        u_pos = batch.get('user_masked_pos')
        if u_pos is not None and u_pos.any():
            u_pred = self.user_mask_head(user_h)  # [B, 18]
            out['user_numeric_pred'] = u_pred[u_pos]
            out['user_numeric_target'] = batch['user_numeric_raw'][u_pos]

        # accounts numeric + paystate reconstruction（6 类共享 mask head）
        for t in _ACCOUNT_TYPES_LOWER:
            pooled, tokens, pay_per_month = accs[t]

            pos = batch.get(f'{t}_masked_pos')
            if pos is not None and pos.any():
                out[f'acc_{t}_numeric_pred'] = self.acc_mask_head(tokens[pos])
                out[f'acc_{t}_numeric_target'] = batch[f'{t}_numeric_raw'][pos]

            pay_pos = batch.get(f'{t}_paystate_masked_pos')
            if pay_pos is not None and pay_pos.any() and pay_per_month.shape[1] > 0:
                # pay_per_month: [B, N, 60, d] — transformer-contextualized per-month tokens
                pay_pred_all = self.paystate_mask_head(pay_per_month)  # [B, N, 60, V]
                out[f'acc_{t}_paystate_pred'] = pay_pred_all[pay_pos]
                out[f'acc_{t}_paystate_target'] = batch[f'{t}_paystate_raw'][pay_pos]

        return out
