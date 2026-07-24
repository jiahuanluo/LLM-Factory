"""PbcCreditModel：5 模态 + 3 交互对的 Model 1 变体。

模态：
  - user（个人信息，固定维度）
  - summary（信息概要 13 表聚合，固定维度）
  - accounts（5 类账户：d1/r1/r2/r3/r4，变长 + 60 月 paystate）
  - queries（查询记录，变长）
  - publics（公共信息，变长）

交互：
  - int_aa：d1 × r2（非循环贷 × 贷记卡）
  - int_aq：accounts pooled × queries
  - int_ap：accounts pooled × publics
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fields import (
    PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE,
    USER_CAT_FIELDS, ACCOUNT_CAT_FIELDS, QUERY_CAT_FIELDS, SUMMARY_TABLES,
)


# ============================================================
# Config
# ============================================================

@dataclass
class PbcCreditModelConfig:
    d: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    top_hidden: int = 256

    # user
    user_numeric_dim: int = 10
    user_cat_tables: dict = field(default_factory=lambda: {})

    # summary
    summary_numeric_dim: int = 36
    summary_cat_tables: dict = field(default_factory=dict)

    # account
    account_numeric_dim: int = 8
    account_cat_tables: dict = field(default_factory=dict)
    paystate_vocab_size: int = PAYSTATE_VOCAB_SIZE

    # query
    query_numeric_dim: int = 1
    query_cat_tables: dict = field(default_factory=dict)

    # public
    public_numeric_dim: int = 2
    public_type_vocab_size: int = PUBLIC_TYPE_VOCAB_SIZE

    # text (gte-new)
    text_encoder_name: str = 'models/gte-new'   # 本地路径或 HF repo
    text_hidden_size: int = 1024                # gte-new hidden
    text_vocab_size: int = 30528
    text_max_len: int = 512
    use_text_branch: bool = True                # 是否启用 text 分支


def _summary_field_counts():
    """统计 summary 分支的 numeric / cat 字段数。"""
    n_num = 0
    n_cat = 0
    for _name, is_list, nums, cats in SUMMARY_TABLES:
        n_num += 1 if is_list else 0  # count 字段（list 才有）
        n_num += len(nums)
        n_cat += len(cats)
    return n_num, n_cat


# ============================================================
# Building blocks（大部分复用 home_credit 模式）
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
    输出: [B, N, d]
    """

    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d = d

    def forward(self, paystate: torch.Tensor, acct_mask: torch.Tensor) -> torch.Tensor:
        """paystate: [B, N, 60], acct_mask: [B, N].
        Returns: [B, N, d] per-account paystate pooled vector.
        """
        B, N, L = paystate.shape
        if N == 0:
            return torch.zeros(B, 0, self.d, device=paystate.device)
        x = paystate.reshape(B * N, L)  # [B*N, 60]
        emb = self.embedding(x)  # [B*N, 60, d]
        pad_mask = (x == 0)  # [B*N, 60]
        h = self.transformer(emb, src_key_padding_mask=pad_mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        # pool over 60 月（忽略 pad）
        m = (x != 0).unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B*N, d]
        return pooled.reshape(B, N, self.d)


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


# ============================================================
# Main model
# ============================================================

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

        # === summary ===
        summary_cat_sizes = config.summary_cat_tables
        self.summary_cat_emb = CategoricalEmbedding(summary_cat_sizes, embed_dim=8)
        self.summary_encoder = FixedEncoder(
            config.summary_numeric_dim + self.summary_cat_emb.out_dim, d, config.dropout,
        )

        # === accounts (shared by 5 types) ===
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

        # === queries ===
        q_cat_sizes = config.query_cat_tables
        self.query_cat_emb = CategoricalEmbedding(q_cat_sizes, embed_dim=4)
        self.query_encoder = SeqEncoder(
            config.query_numeric_dim + self.query_cat_emb.out_dim,
            d, config.n_heads, config.n_layers, config.dropout,
        )

        # === publics ===
        self.public_type_emb = nn.Embedding(config.public_type_vocab_size + 1, 4)
        self.public_encoder = SeqEncoder(
            config.public_numeric_dim + 4,
            d, config.n_heads, config.n_layers, config.dropout,
        )

        # === interactive pairs (3) ===
        self.int_aa = InteractiveModule(d, config.n_heads, config.dropout)  # d1 × r2
        self.int_aq = InteractiveModule(d, config.n_heads, config.dropout)  # accounts × queries
        self.int_ap = InteractiveModule(d, config.n_heads, config.dropout)  # accounts × publics

        # === top (finetune) ===
        # 11 pooled vectors: user, summary, d1, r1, r2, r3, r4, query, public, int_aa, int_aq, int_ap
        # 实际是 12 个，但 int_aa 也算 1 个
        n_pooled = 12
        self.top = nn.Sequential(
            nn.Linear(d * n_pooled, config.top_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.top_hidden, 1),
        )

        # === pretrain mask heads ===
        if pretrain_mode:
            self.acc_mask_head = nn.Linear(d, config.account_numeric_dim)
            self.paystate_mask_head = nn.Linear(d, config.paystate_vocab_size)
            self.query_mask_head = nn.Linear(d, config.query_numeric_dim)
            self.public_mask_head = nn.Linear(d, config.public_numeric_dim)
            self.summary_mask_head = nn.Linear(d, config.summary_numeric_dim)

        # === text encoder (gte-new) ===
        if config.use_text_branch:
            from transformers import AutoConfig, AutoModelForMaskedLM
            try:
                self.text_encoder = AutoModelForMaskedLM.from_pretrained(
                    config.text_encoder_name, trust_remote_code=True,
                )
            except OSError:
                import warnings
                warnings.warn(f'No pretrained weights at {config.text_encoder_name}; '
                              f'initializing from config (生产端应加载真实权重)')
                text_cfg = AutoConfig.from_pretrained(
                    config.text_encoder_name, trust_remote_code=True,
                )
                self.text_encoder = AutoModelForMaskedLM.from_config(text_cfg)
            # 节省显存：启用 gradient checkpointing（大模型必备）
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
            self.text_proj = nn.Linear(config.text_hidden_size, d)

    def _encode_accounts(self, batch: dict):
        """对 5 类账户共享同一套 encoder，返回 dict[type] = (pooled, tokens)."""
        results = {}
        for t in ['d1', 'r1', 'r2', 'r3', 'r4']:
            numeric = batch[f'{t}_numeric']  # [B, N, Fn]
            cat_ids = batch[f'{t}_cat_ids']
            paystate = batch[f'{t}_paystate']
            mask = batch[f'{t}_mask']  # [B, N]

            B, N = numeric.shape[:2]
            if N == 0:
                pooled = torch.zeros(B, self.config.d, device=numeric.device)
                tokens = torch.zeros(B, 0, self.config.d, device=numeric.device)
            else:
                cat_flat = self.acc_cat_emb(cat_ids)
                feats = torch.cat([numeric, cat_flat], dim=-1)
                seq_pooled, seq_tokens = self.acc_seq_encoder(feats, mask)
                pay_tokens = self.paystate_encoder(paystate, mask)  # [B, N, d]
                tokens = self.acc_fuse(torch.cat([seq_tokens, pay_tokens], dim=-1))
                # re-mask pad positions (fuse may leak)
                m = mask.unsqueeze(-1).float()
                tokens = tokens * m
                pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            results[t] = (pooled, tokens)
        return results

    def forward(self, batch: dict):
        # === encode branches ===
        user_cat = self.user_cat_emb(batch['user_cat_ids'])
        user_h = self.user_encoder(torch.cat([batch['user_numeric'], user_cat], dim=-1))

        s_cat = self.summary_cat_emb(batch['summary_cat_ids'])
        summary_h = self.summary_encoder(torch.cat([batch['summary_numeric'], s_cat], dim=-1))

        accs = self._encode_accounts(batch)

        q_cat = self.query_cat_emb(batch['query_cat_ids'])
        q_feats = torch.cat([batch['query_numeric'], q_cat], dim=-1)
        query_h, query_tokens = self.query_encoder(q_feats, batch['query_mask'])

        p_ids = batch['public_cat_ids'].clamp_min(0).squeeze(-1)
        p_cat = self.public_type_emb(p_ids)
        p_feats = torch.cat([batch['public_numeric'], p_cat], dim=-1)
        public_h, public_tokens = self.public_encoder(p_feats, batch['public_mask'])

        # === text 分支（gte-new） ===
        text_h = None
        if self.config.use_text_branch:
            # 用 base model 拿 last_hidden_state（gte-new 的 output_hidden_states 有 bug
            # 会返回错的 shape；直接走 self.text_encoder.new 更可靠）
            base = getattr(self.text_encoder, 'new', None) or getattr(self.text_encoder, 'bert', None)
            if base is not None:
                base_out = base(
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch['text_attention_mask'],
                )
                last_hidden = base_out.last_hidden_state  # [B, L, H]
                text_h = self.text_proj(last_hidden[:, 0])  # [CLS] → [B, d]

        if self.pretrain_mode:
            return self._forward_pretrain(
                batch, accs, query_tokens, public_tokens, summary_h, text_h,
            )

        # === interactive ===
        d1_p, d1_t = accs['d1']
        r1_p, r1_t = accs['r1']
        r2_p, r2_t = accs['r2']
        r3_p, r3_t = accs['r3']
        r4_p, r4_t = accs['r4']

        int_aa = self.int_aa(d1_t, batch['d1_mask'], r2_t, batch['r2_mask'])

        # accounts pooled（5 类 concat 后过 transformer 太复杂，简化为 d1+r2 联合）
        acc_tokens = torch.cat([d1_t, r2_t], dim=1)  # [B, N_d1+N_r2, d]
        acc_mask = torch.cat([batch['d1_mask'], batch['r2_mask']], dim=1)
        int_aq = self.int_aq(acc_tokens, acc_mask,
                              query_tokens, batch['query_mask'])
        int_ap = self.int_ap(acc_tokens, acc_mask,
                              public_tokens, batch['public_mask'])

        # === concat + top ===
        pooled_list = [
            user_h, summary_h,
            d1_p, r1_p, r2_p, r3_p, r4_p,
            query_h, public_h,
            int_aa, int_aq, int_ap,
        ]
        if text_h is not None:
            pooled_list.append(text_h)
        concat = torch.cat(pooled_list, dim=-1)

        # top 维度自适应（text 分支开关会改变 concat 长度）
        if not hasattr(self, '_top_in_dim'):
            self._top_in_dim = concat.shape[-1]
            # 重建 top 层匹配实际维度
            d = self.config.d
            self.top = nn.Sequential(
                nn.Linear(self._top_in_dim, self.config.top_hidden),
                nn.ReLU(), nn.Dropout(self.config.dropout),
                nn.Linear(self.config.top_hidden, 1),
            ).to(concat.device)
        logit = self.top(concat)
        return logit

    def _forward_pretrain(self, batch, accs, query_tokens, public_tokens, summary_h, text_h=None):
        out = {}
        # accounts numeric + paystate reconstruction
        for t in ['d1', 'r2', 'r3']:
            pos = batch.get(f'{t}_masked_pos')
            if pos is None or not pos.any():
                continue
            pooled, tokens = accs[t]
            out[f'acc_{t}_numeric_pred'] = self.acc_mask_head(tokens[pos])
            out[f'acc_{t}_numeric_target'] = batch[f'{t}_numeric_raw'][pos]

            pay_pos = batch.get(f'{t}_paystate_masked_pos')
            if pay_pos is not None and pay_pos.any():
                pay_tokens_expanded = tokens.unsqueeze(2).expand(-1, -1, 60, -1)
                pay_pred_all = self.paystate_mask_head(pay_tokens_expanded)
                out[f'acc_{t}_paystate_pred'] = pay_pred_all[pay_pos]
                out[f'acc_{t}_paystate_target'] = batch[f'{t}_paystate_raw'][pay_pos]

        q_pos = batch.get('query_masked_pos')
        if q_pos is not None and q_pos.any():
            out['query_numeric_pred'] = self.query_mask_head(query_tokens[q_pos])
            out['query_numeric_target'] = batch['query_numeric_raw'][q_pos]

        p_pos = batch.get('public_masked_pos')
        if p_pos is not None and p_pos.any():
            out['public_numeric_pred'] = self.public_mask_head(public_tokens[p_pos])
            out['public_numeric_target'] = batch['public_numeric_raw'][p_pos]

        s_pos = batch.get('summary_masked_pos')
        if s_pos is not None and s_pos.any():
            s_pred_flat = self.summary_mask_head(summary_h)
            out['summary_numeric_pred'] = s_pred_flat[s_pos]
            out['summary_numeric_target'] = batch['summary_numeric_raw'][s_pos]

        # text MLM：logits 直接由 NewForMaskedLM 给
        if self.config.use_text_branch and 'text_input_ids' in batch:
            text_out = self.text_encoder(
                input_ids=batch['text_input_ids'],
                attention_mask=batch['text_attention_mask'],
            )
            out['text_logits'] = text_out.logits  # [B, L, V]
            out['text_target'] = batch['text_input_ids_raw']  # 原始未 mask 的 ids
            out['text_mask_pos'] = batch.get('text_masked_pos')

        return out
