"""HomeCreditModel — Model 1 adapted for Home Credit with 6 input branches.

Architecture:
  6 Encoders (User/Query/Bureau/PrevApp/Card/Text)
  + 3 Interactive Module pairs (Bureau×PrevApp, Bureau×Card, PrevApp×Card)
  + top Dense+ReLU+Dense

Per spec §3, D=128, MHA heads=4, encoder layers=2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class HomeCreditModelConfig:
    # Hidden dim
    d: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Field dimensions (must match sample_builder)
    user_numeric_dim: int = 62
    user_cat_field_sizes: dict = field(default_factory=lambda: {
        # Will be filled at runtime from cat_vocab.json; defaults reflect approx train sizes
        'CODE_GENDER': 4, 'FLAG_OWN_CAR': 3, 'FLAG_OWN_REALTY': 3,
        'NAME_CONTRACT_TYPE': 3, 'NAME_TYPE_SUITE': 9, 'NAME_INCOME_TYPE': 8,
        'NAME_EDUCATION_TYPE': 6, 'NAME_FAMILY_STATUS': 7, 'NAME_HOUSING_TYPE': 7,
        'OCCUPATION_TYPE': 19, 'ORGANIZATION_TYPE': 59, 'WEEKDAY_APPR_PROCESS_START': 8,
        'HOUSETYPE_MODE': 4, 'WALLSMATERIAL_MODE': 8, 'EMERGENCYSTATE_MODE': 3, 'FONDKAPREMONT_MODE': 5,
    })
    user_cat_embed_dim: int = 8  # per categorical field

    bureau_numeric_dim: int = 12
    bureau_cat_field_sizes: dict = field(default_factory=lambda: {
        'CREDIT_ACTIVE': 5, 'CREDIT_CURRENCY': 4, 'CREDIT_TYPE': 16,
    })
    bureau_cat_embed_dim: int = 4

    prev_numeric_dim: int = 19
    prev_cat_field_sizes: dict = field(default_factory=lambda: {
        'NAME_CONTRACT_TYPE': 5, 'WEEKDAY_APPR_PROCESS_START': 8,
        'FLAG_LAST_APPL_PER_CONTRACT': 3, 'NAME_CASH_LOAN_PURPOSE': 26,
        'NAME_CONTRACT_STATUS': 5, 'NAME_PAYMENT_TYPE': 5,
        'CODE_REJECT_REASON': 10, 'NAME_TYPE_SUITE': 9, 'NAME_CLIENT_TYPE': 5,
        'NAME_GOODS_CATEGORY': 23, 'NAME_PORTFOLIO': 5, 'NAME_PRODUCT_TYPE': 4,
        'CHANNEL_TYPE': 9, 'NAME_SELLER_INDUSTRY': 12, 'NAME_YIELD_GROUP': 6,
        'PRODUCT_COMBINATION': 124,
    })
    prev_cat_embed_dim: int = 4

    card_numeric_dim: int = 20
    card_cat_field_sizes: dict = field(default_factory=lambda: {
        'NAME_CONTRACT_STATUS': 8,
    })
    card_cat_embed_dim: int = 4

    # Text
    text_vocab_size: int = 30522  # neobert
    text_max_len: int = 64
    text_pad_id: int = 0

    # Top
    top_hidden: int = 256


# ============================================================
# Building blocks
# ============================================================

class CategoricalEmbedding(nn.Module):
    """Embed each categorical field, concat all to a flat vector."""

    def __init__(self, field_sizes: dict, embed_dim: int):
        super().__init__()
        self.fields = list(field_sizes.keys())
        self.embeds = nn.ModuleDict({
            name: nn.Embedding(size + 1, embed_dim)  # +1 for <UNK> at id 0
            for name, size in field_sizes.items()
        })
        self.out_dim = len(self.fields) * embed_dim

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        """cat_ids: [..., F] long → [..., F*embed_dim] float."""
        outs = []
        for i, name in enumerate(self.fields):
            outs.append(self.embeds[name](cat_ids[..., i]))
        return torch.cat(outs, dim=-1)


class SeqEncoder(nn.Module):
    """Encode a variable-length set/sequence of rows to a single D-dim vector.

    Input row features (numeric + embedded categoricals) are LayerNorm'd,
    projected to D, then passed through N MHA layers + average pooling.
    """

    def __init__(self, in_dim: int, d: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)  # tame unnormalized tabular inputs
        self.proj = nn.Linear(in_dim, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,  # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d = d

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """features: [B, N, in_dim], mask: [B, N] (1=valid).
        Returns: [B, d].
        """
        if features.shape[1] == 0:
            # Empty sequence → return zero vector (will be masked downstream)
            return torch.zeros(features.shape[0], self.d, device=features.device, dtype=features.dtype)
        h = self.input_norm(features)
        h = self.proj(h)
        # nn.TransformerEncoder expects src_key_padding_mask=True at padded positions
        pad_mask = (mask == 0)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        # Avg pool over valid positions
        m = mask.unsqueeze(-1).float()  # [B, N, 1]
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class UserEncoder(nn.Module):
    """Encode fixed-dim user profile (numeric + cat embeddings) to D."""

    def __init__(self, numeric_dim: int, cat_embedding: CategoricalEmbedding, d: int, dropout: float):
        super().__init__()
        self.cat_embedding = cat_embedding
        self.proj = nn.Sequential(
            nn.Linear(numeric_dim + cat_embedding.out_dim, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, numeric: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        cat_emb = self.cat_embedding(cat_ids)
        x = torch.cat([numeric, cat_emb], dim=-1)
        return self.proj(x)


class TextEncoder(nn.Module):
    """Standard transformer encoder over token IDs."""

    def __init__(self, vocab_size: int, d: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=dropout, activation='relu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d = d

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] == 0:
            return torch.zeros(input_ids.shape[0], self.d, device=input_ids.device)
        h = self.embedding(input_ids)
        pad_mask = (attention_mask == 0)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


class InteractiveModule(nn.Module):
    """Cross-attention: query tokens from A attend to key/value tokens from B.

    Single direction (q=A, kv=B). Output: pooled cross-attended representation.
    """

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

    def forward(self, q_tokens: torch.Tensor, q_mask: torch.Tensor,
                kv_tokens: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        """q_tokens/kv_tokens: [B, N, d]. q_mask/kv_mask: [B, N].
        Returns: [B, d] (pooled over q).
        """
        if q_tokens.shape[1] == 0 or kv_tokens.shape[1] == 0:
            return torch.zeros(q_tokens.shape[0], q_tokens.shape[2], device=q_tokens.device)
        kv_pad_mask = (kv_mask == 0)
        attn_out, _ = self.cross_attn(q_tokens, kv_tokens, kv_tokens,
                                       key_padding_mask=kv_pad_mask, need_weights=False)
        h = self.norm(q_tokens + attn_out)
        h = self.norm2(h + self.ffn(h))
        # Avg pool over q's valid positions
        m = q_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled


# ============================================================
# Main model
# ============================================================

class HomeCreditModel(nn.Module):
    """Multi-modal Home Credit Model 1.

    Modes:
      - finetune: forward returns logits [B, 1] for BCEWithLogits
      - pretrain: forward returns dict of per-branch predictions for mask loss
                  (caller must pass pretrain_batch with *_masked_pos fields)
    """

    def __init__(self, config: HomeCreditModelConfig, pretrain_mode: bool = False):
        super().__init__()
        self.config = config
        self.pretrain_mode = pretrain_mode

        d = config.d

        # === Encoders ===
        # User
        self.user_cat_emb = CategoricalEmbedding(config.user_cat_field_sizes, config.user_cat_embed_dim)
        self.user_encoder = UserEncoder(
            config.user_numeric_dim, self.user_cat_emb, d, config.dropout,
        )

        # Query (just 6 tokens × 1 feature; embed each to D)
        self.query_proj = nn.Linear(1, d)
        self.query_encoder = SeqEncoder(d, d, config.n_heads, config.n_layers, config.dropout)

        # Bureau
        self.bureau_cat_emb = CategoricalEmbedding(config.bureau_cat_field_sizes, config.bureau_cat_embed_dim)
        self.bureau_encoder = SeqEncoder(
            config.bureau_numeric_dim + self.bureau_cat_emb.out_dim,
            d, config.n_heads, config.n_layers, config.dropout,
        )

        # PrevApp
        self.prev_cat_emb = CategoricalEmbedding(config.prev_cat_field_sizes, config.prev_cat_embed_dim)
        self.prev_encoder = SeqEncoder(
            config.prev_numeric_dim + self.prev_cat_emb.out_dim,
            d, config.n_heads, config.n_layers, config.dropout,
        )

        # Card
        self.card_cat_emb = CategoricalEmbedding(config.card_cat_field_sizes, config.card_cat_embed_dim)
        self.card_encoder = SeqEncoder(
            config.card_numeric_dim + self.card_cat_emb.out_dim,
            d, config.n_heads, config.n_layers, config.dropout,
        )

        # Text
        self.text_encoder = TextEncoder(
            config.text_vocab_size, d, config.n_heads, config.n_layers, config.dropout,
        )

        # === Interactive Module (3 pairs) ===
        self.int_bp = InteractiveModule(d, config.n_heads, config.dropout)  # Bureau × PrevApp
        self.int_bc = InteractiveModule(d, config.n_heads, config.dropout)  # Bureau × Card
        self.int_pc = InteractiveModule(d, config.n_heads, config.dropout)  # PrevApp × Card

        # === Top (finetune mode) ===
        # 9 pooled vectors: user, query, bureau, prev, card, text, int_bp, int_bc, int_pc
        self.top = nn.Sequential(
            nn.Linear(d * 9, config.top_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.top_hidden, 1),
        )

        # === Mask heads (pretrain mode) ===
        if pretrain_mode:
            # Reconstruction heads for each branch (predict full row from masked position)
            self.bureau_mask_head = nn.Linear(d, config.bureau_numeric_dim)
            self.prev_mask_head = nn.Linear(d, config.prev_numeric_dim)
            self.card_mask_head = nn.Linear(d, config.card_numeric_dim)
            self.text_mask_head = nn.Linear(d, config.text_vocab_size)

            # Learnable [MASK] token embedding per branch (added to position before encoding)
            self.bureau_mask_emb = nn.Parameter(torch.randn(config.bureau_numeric_dim) * 0.02)
            self.prev_mask_emb = nn.Parameter(torch.randn(config.prev_numeric_dim) * 0.02)
            self.card_mask_emb = nn.Parameter(torch.randn(config.card_numeric_dim) * 0.02)
            # Text uses a special token id; we'll add MASK_ID = vocab_size to embedding
            self.text_mask_id = config.text_vocab_size  # out-of-vocab slot
            # Extend text embedding to include mask token
            with torch.no_grad():
                old = self.text_encoder.embedding.weight
                new = nn.Parameter(torch.empty(config.text_vocab_size + 1, old.shape[1]))
                nn.init.normal_(new, std=0.02)
                new[:config.text_vocab_size] = old
                self.text_encoder.embedding.weight = new
                self.text_encoder.embedding.num_embeddings = config.text_vocab_size + 1

    # === Sub-encoder helpers ===

    def _encode_branch(self, encoder: SeqEncoder, cat_emb: CategoricalEmbedding,
                       numeric: torch.Tensor, cat_ids: torch.Tensor, mask: torch.Tensor,
                       masked_pos: torch.Tensor | None = None,
                       numeric_mask_emb: torch.Tensor | None = None):
        """Embed categoricals per row, concat with numeric, then run seq encoder.

        In pretrain mode, if masked_pos and numeric_mask_emb are provided, the
        masked rows are replaced with the sentinel BEFORE encoding (so the
        encoder cannot peek at the true value). cat_ids at masked rows become 0
        (<UNK>).

        Returns: (pooled [B, d], tokens [B, N, d]).
        """
        if numeric.shape[1] == 0:
            B = numeric.shape[0]
            tokens = torch.zeros(B, 0, self.config.d, device=numeric.device)
            pooled = torch.zeros(B, self.config.d, device=numeric.device)
            return pooled, tokens

        if masked_pos is not None and numeric_mask_emb is not None:
            mp_expand_num = masked_pos.unsqueeze(-1).expand_as(numeric)
            mp_expand_cat = masked_pos.unsqueeze(-1).expand_as(cat_ids)
            numeric_used = torch.where(
                mp_expand_num,
                numeric_mask_emb.to(numeric.dtype).expand_as(numeric),
                numeric,
            )
            cat_ids_used = torch.where(mp_expand_cat, torch.zeros_like(cat_ids), cat_ids)
        else:
            numeric_used = numeric
            cat_ids_used = cat_ids

        cat_flat = cat_emb(cat_ids_used)
        feats = torch.cat([numeric_used, cat_flat], dim=-1)  # [B, N, in_dim]
        h = encoder.input_norm(feats)
        h = encoder.proj(h)
        pad_mask = (mask == 0)
        h = encoder.transformer(h, src_key_padding_mask=pad_mask)
        # Samples whose mask is all-zero (no valid positions) get NaN from
        # attention softmax over all-masked keys. Replace with 0 — these
        # samples contribute nothing to the pooled output anyway.
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return pooled, h

    def forward(self, batch: dict) -> dict | torch.Tensor:
        user_h = self.user_encoder(batch['user_numeric'], batch['user_cat_ids'])

        # Query: treat 6 numeric values as a sequence of 1-D tokens
        qf = self.query_proj(batch['query_features'])  # [B, 6, d]
        qm = batch['query_mask']
        # Apply transformer + pool via SeqEncoder's internals
        q_tokens = self.query_encoder.transformer(
            qf, src_key_padding_mask=(qm == 0),
        )
        q_tokens = torch.nan_to_num(q_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        qm_full = qm.unsqueeze(-1).float()
        query_h = (q_tokens * qm_full).sum(dim=1) / qm_full.sum(dim=1).clamp(min=1.0)

        # In pretrain mode, pass masked_pos so encoder replaces those rows
        # with [MASK] sentinel before encoding (no peeking at true values)
        bureau_mp = batch.get('bureau_masked_pos') if self.pretrain_mode else None
        prev_mp = batch.get('prev_masked_pos') if self.pretrain_mode else None
        card_mp = batch.get('card_masked_pos') if self.pretrain_mode else None

        bureau_h, bureau_tokens = self._encode_branch(
            self.bureau_encoder, self.bureau_cat_emb,
            batch['bureau_numeric'], batch['bureau_cat_ids'], batch['bureau_mask'],
            masked_pos=bureau_mp,
            numeric_mask_emb=self.bureau_mask_emb if self.pretrain_mode else None,
        )
        prev_h, prev_tokens = self._encode_branch(
            self.prev_encoder, self.prev_cat_emb,
            batch['prev_numeric'], batch['prev_cat_ids'], batch['prev_mask'],
            masked_pos=prev_mp,
            numeric_mask_emb=self.prev_mask_emb if self.pretrain_mode else None,
        )
        card_h, card_tokens = self._encode_branch(
            self.card_encoder, self.card_cat_emb,
            batch['card_numeric'], batch['card_cat_ids'], batch['card_mask'],
            masked_pos=card_mp,
            numeric_mask_emb=self.card_mask_emb if self.pretrain_mode else None,
        )

        text_h = self.text_encoder(
            batch['text_input_ids'], batch['text_attention_mask'],
        )

        if self.pretrain_mode:
            return self._forward_pretrain(batch, bureau_tokens, prev_tokens, card_tokens)

        # Interactive pairs (only needed for finetune)
        int_bp = self.int_bp(bureau_tokens, batch['bureau_mask'],
                             prev_tokens, batch['prev_mask'])
        int_bc = self.int_bc(bureau_tokens, batch['bureau_mask'],
                             card_tokens, batch['card_mask'])
        int_pc = self.int_pc(prev_tokens, batch['prev_mask'],
                             card_tokens, batch['card_mask'])

        concat = torch.cat([
            user_h, query_h, bureau_h, prev_h, card_h,
            text_h, int_bp, int_bc, int_pc,
        ], dim=-1)  # [B, 9*d]
        logit = self.top(concat)  # [B, 1]
        return logit

    def _forward_pretrain(self, batch, bureau_tokens, prev_tokens, card_tokens):
        """Compute per-branch reconstruction at masked positions.

        Branch tokens are already from the masked-input encoding pass.
        Text needs a separate re-encoding because text mask replaces token IDs.
        """
        out = {}
        if 'bureau_masked_pos' in batch and batch['bureau_masked_pos'].any():
            mp = batch['bureau_masked_pos']
            out['bureau_pred'] = self.bureau_mask_head(bureau_tokens[mp])
            out['bureau_target'] = batch['bureau_numeric'][mp]
        if 'prev_masked_pos' in batch and batch['prev_masked_pos'].any():
            mp = batch['prev_masked_pos']
            out['prev_pred'] = self.prev_mask_head(prev_tokens[mp])
            out['prev_target'] = batch['prev_numeric'][mp]
        if 'card_masked_pos' in batch and batch['card_masked_pos'].any():
            mp = batch['card_masked_pos']
            out['card_pred'] = self.card_mask_head(card_tokens[mp])
            out['card_target'] = batch['card_numeric'][mp]
        if 'text_masked_pos' in batch and batch['text_masked_pos'].any():
            mp = batch['text_masked_pos']
            input_ids = batch['text_input_ids'].clone()
            input_ids[mp] = self.text_mask_id
            text_emb = self.text_encoder.embedding(input_ids)
            pad_mask = (batch['text_attention_mask'] == 0)
            text_tokens = self.text_encoder.transformer(
                text_emb, src_key_padding_mask=pad_mask,
            )
            text_tokens = torch.nan_to_num(text_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            out['text_logits'] = self.text_mask_head(text_tokens[mp])
            out['text_target'] = batch['text_input_ids'][mp]
        return out
