"""Home Credit Model 1 sample builder.

Reads raw CSVs from data/home-credit/ and constructs multi-modal samples for
pretrain (train + test unlabeled) and finetune (train + val splits).

Output (in data/home-credit/processed/):
  samples_train.pkl           — labeled train samples (~246k)
  samples_val.pkl             — labeled val samples (~61k, holdout)
  samples_test_unlabeled.pkl  — unlabeled samples (~48k, pretrain only)
  normalizer.json             — mean/std for user numeric fields (train stats)
  cat_vocab.json              — per-table per-field categorical vocab
  split.json                  — train/val/test SK_ID_CURR lists

Sample format (per SK_ID_CURR):
  {
    'sk_id_curr': int,
    'user_numeric':       Tensor[F_user_num],
    'user_cat_ids':       Tensor[F_user_cat],
    'user_cat_mask':      Tensor[F_user_cat],
    'query_features':     Tensor[6, 1],
    'query_mask':         Tensor[6],
    'bureau_numeric':     Tensor[N_bureau, 12],
    'bureau_cat_ids':     Tensor[N_bureau, 3],
    'bureau_mask':        Tensor[N_bureau],
    'prev_numeric':       Tensor[N_prev, 19],
    'prev_cat_ids':       Tensor[N_prev, 16],
    'prev_mask':          Tensor[N_prev],
    'card_numeric':       Tensor[N_card, 20],
    'card_cat_ids':       Tensor[N_card, 1],
    'card_mask':          Tensor[N_card],
    'text_input_ids':     Tensor[L_text],
    'text_attention_mask':Tensor[L_text],
    'target':             Tensor[1]   # only for labeled splits
  }
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ============================================================
# Field definitions
# ============================================================

QUERY_FIELDS = [
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
]

USER_NUMERIC_FIELDS = [
    # AMT (4)
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    # DAYS (5, relative to application, negative)
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    # CNT (2)
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    # EXT_SOURCE (3)
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    # REGION (3)
    'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
    # SOCIAL_CIRCLE (4)
    'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
    # Misc (2)
    'HOUR_APPR_PROCESS_START', 'OWN_CAR_AGE',
    # Building normalized (low-missing subset, 7)
    'TOTALAREA_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'FLOORSMAX_AVG',
    'APARTMENTS_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'LIVINGAREA_AVG',
    # Flags (6)
    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
    'FLAG_PHONE', 'FLAG_EMAIL',
    # REG/LIVE (6)
    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
    # Document flags (20)
    *[f'FLAG_DOCUMENT_{i}' for i in range(2, 22)],
]

USER_CAT_FIELDS = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START',
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'FONDKAPREMONT_MODE',
]

TEXT_FIELDS = [
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'NAME_CONTRACT_TYPE',
]

BUREAU_NUMERIC_FIELDS = [
    'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT',
    'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE',
    'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY',
]
BUREAU_CAT_FIELDS = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

PREV_NUMERIC_FIELDS = [
    'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
    'HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT',
    'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION',
    'SELLERPLACE_AREA', 'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
    'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'NFLAG_INSURED_ON_APPROVAL',
]
PREV_CAT_FIELDS = [
    'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT',
    'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
    'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',
    'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
]

CARD_NUMERIC_FIELDS = [
    'MONTHS_BALANCE', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
    'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
    'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
    'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
    'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
    'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF',
]
CARD_CAT_FIELDS = ['NAME_CONTRACT_STATUS']


# ============================================================
# Vocab and normalizer
# ============================================================

def build_cat_vocab(app_df, bureau_df, prev_df, card_df):
    """Per-table per-field vocab. id 0 reserved for <UNK> (NaN)."""
    vocab = {}
    for table_name, df, fields in [
        ('user', app_df, USER_CAT_FIELDS),
        ('bureau', bureau_df, BUREAU_CAT_FIELDS),
        ('prev', prev_df, PREV_CAT_FIELDS),
        ('card', card_df, CARD_CAT_FIELDS),
    ]:
        vocab[table_name] = {}
        for f in fields:
            vals = sorted(df[f].dropna().astype(str).unique().tolist())
            vocab[table_name][f] = {v: i + 1 for i, v in enumerate(vals)}
    return vocab


def compute_normalizer(df, numeric_fields):
    """Mean/std ignoring NaN. Replaces DAYS_EMPLOYED=365243 sentinel with NaN first."""
    df = df.copy()
    if 'DAYS_EMPLOYED' in df.columns:
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    norm = {}
    for f in numeric_fields:
        if f not in df.columns:
            continue
        m = float(df[f].mean())
        s = float(df[f].std())
        norm[f] = {'mean': m, 'std': s if s > 0 else 1.0}
    return norm


# ============================================================
# Helpers
# ============================================================

def row_to_text(row):
    """Pseudo-text from 7 enum fields. Whitespace/slash replaced for clean tokenization."""
    parts = []
    for f in TEXT_FIELDS:
        v = row[f]
        if pd.isna(v):
            v = 'unknown'
        s = f"{f}_{v}".replace(' ', '_').replace('/', '_')
        parts.append(s)
    return " ".join(parts)


def clean_app_df(df, normalizer):
    """Replace DAYS_EMPLOYED sentinel, z-score normalize user + query numeric fields."""
    df = df.copy()
    if 'DAYS_EMPLOYED' in df.columns:
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    for f in list(USER_NUMERIC_FIELDS) + list(QUERY_FIELDS):
        if f in normalizer and f in df.columns:
            m, s = normalizer[f]['mean'], normalizer[f]['std']
            df[f] = (df[f] - m) / s
    return df


def normalize_subtable(df, numeric_fields, normalizer):
    """Apply per-field z-score to sub-table numeric columns in place."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    for f in numeric_fields:
        if f in df.columns and f in normalizer:
            m, s = normalizer[f]['mean'], normalizer[f]['std']
            df[f] = (df[f] - m) / s
    return df


def encode_cat_value(v, vocab_field):
    """NaN → 0 (<UNK>); unknown string → 0."""
    if pd.isna(v):
        return 0
    return vocab_field.get(str(v), 0)


def _build_branch_tensors(group, numeric_fields, cat_fields, cat_vocab, table_name):
    """Build (numeric, cat_ids, mask) tensors for one sub-table group."""
    if group is None or len(group) == 0:
        n_num = len(numeric_fields)
        n_cat = len(cat_fields)
        return (
            torch.zeros(0, n_num, dtype=torch.float32),
            torch.zeros(0, n_cat, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
    numeric = torch.tensor(
        group[numeric_fields].fillna(0).values,
        dtype=torch.float32,
    )
    cat_ids = torch.tensor(
        [
            [encode_cat_value(v, cat_vocab[table_name][f]) for f, v in zip(cat_fields, row)]
            for row in group[cat_fields].values.tolist()
        ],
        dtype=torch.long,
    )
    mask = torch.ones(len(group), dtype=torch.long)
    return numeric, cat_ids, mask


def build_sample(
    sk_id_curr,
    app_row,
    bureau_group,
    prev_group,
    card_group,
    cat_vocab,
    text_input_ids,
    text_attention_mask,
    has_target,
):
    """Build one sample dict for a given SK_ID_CURR."""
    user_numeric = torch.tensor(
        [0.0 if pd.isna(app_row[f]) else float(app_row[f]) for f in USER_NUMERIC_FIELDS],
        dtype=torch.float32,
    )
    user_cat_ids = torch.tensor(
        [encode_cat_value(app_row[f], cat_vocab['user'][f]) for f in USER_CAT_FIELDS],
        dtype=torch.long,
    )
    user_cat_mask = torch.tensor(
        [0 if pd.isna(app_row[f]) else 1 for f in USER_CAT_FIELDS],
        dtype=torch.long,
    )

    query_features = torch.tensor(
        [[0.0] if pd.isna(app_row[f]) else [float(app_row[f])] for f in QUERY_FIELDS],
        dtype=torch.float32,
    )  # [6, 1]
    query_mask = torch.tensor(
        [0 if pd.isna(app_row[f]) else 1 for f in QUERY_FIELDS],
        dtype=torch.long,
    )

    bureau_num, bureau_cat, bureau_mask = _build_branch_tensors(
        bureau_group, BUREAU_NUMERIC_FIELDS, BUREAU_CAT_FIELDS, cat_vocab, 'bureau',
    )
    prev_num, prev_cat, prev_mask = _build_branch_tensors(
        prev_group, PREV_NUMERIC_FIELDS, PREV_CAT_FIELDS, cat_vocab, 'prev',
    )
    card_num, card_cat, card_mask = _build_branch_tensors(
        card_group, CARD_NUMERIC_FIELDS, CARD_CAT_FIELDS, cat_vocab, 'card',
    )

    sample = {
        'sk_id_curr': int(sk_id_curr),
        'user_numeric': user_numeric,
        'user_cat_ids': user_cat_ids,
        'user_cat_mask': user_cat_mask,
        'query_features': query_features,
        'query_mask': query_mask,
        'bureau_numeric': bureau_num,
        'bureau_cat_ids': bureau_cat,
        'bureau_mask': bureau_mask,
        'prev_numeric': prev_num,
        'prev_cat_ids': prev_cat,
        'prev_mask': prev_mask,
        'card_numeric': card_num,
        'card_cat_ids': card_cat,
        'card_mask': card_mask,
        'text_input_ids': torch.tensor(text_input_ids, dtype=torch.long),
        'text_attention_mask': torch.tensor(text_attention_mask, dtype=torch.long),
    }
    if has_target:
        sample['target'] = torch.tensor([int(app_row['TARGET'])], dtype=torch.long)
    return sample


def stream_samples_to_files(
    app_df,
    bureau_groups,
    prev_groups,
    card_groups,
    cat_vocab,
    text_encodings,
    output_paths: dict,  # {position: path} for routing; or {'all': path} for single file
    has_target: bool,
    position_to_key=None,  # function(pos) -> key into output_paths; None means all go to output_paths['all']
    progress_label='',
):
    """Stream samples to pickle files (one pickle.dump per record).

    Memory-bounded: only one sample is materialized at a time. Use this to
    avoid OOM on large datasets.

    output_paths: dict of {key: open_file_handle_or_path}. If position_to_key
    is None, every sample goes to output_paths['all'].
    """
    # Open all output files for binary write
    own_handles = {}
    for k, p in output_paths.items():
        if hasattr(p, 'write'):
            own_handles[k] = p
        else:
            own_handles[k] = open(p, 'wb')

    try:
        n = len(app_df)
        counts = {k: 0 for k in output_paths}
        for i, (_, row) in enumerate(app_df.iterrows()):
            if i % 50000 == 0:
                stats = ' '.join(f'{k}={v}' for k, v in counts.items())
                print(f'  {progress_label} {i}/{n}  ({stats})')
            sk_id = row['SK_ID_CURR']
            sample = build_sample(
                sk_id, row,
                bureau_groups.get(sk_id),
                prev_groups.get(sk_id),
                card_groups.get(sk_id),
                cat_vocab,
                text_input_ids=text_encodings['input_ids'][i],
                text_attention_mask=text_encodings['attention_mask'][i],
                has_target=has_target,
            )
            key = position_to_key(i) if position_to_key else 'all'
            pickle.dump(sample, own_handles[key], protocol=4)
            counts[key] += 1
        stats = ' '.join(f'{k}={v}' for k, v in counts.items())
        print(f'  {progress_label} done ({stats})')
        return counts
    finally:
        for k, h in own_handles.items():
            if h not in (output_paths.get(k),):  # only close if we opened it
                if hasattr(output_paths.get(k), 'write'):
                    pass  # caller owns it
                else:
                    h.close()


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f'  saved {path} ({size_mb:.1f} MB)')


# ============================================================
# Main pipeline
# ============================================================

def build_all(raw_dir='data/home-credit', out_dir='data/home-credit/processed',
              tokenizer_path='models/neobert', val_size=0.2, seed=42):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print('=== Reading CSVs ===')
    app_train = pd.read_csv(raw / 'application_train.csv')
    app_test = pd.read_csv(raw / 'application_test.csv')
    bureau = pd.read_csv(raw / 'bureau.csv')
    prev_app = pd.read_csv(raw / 'previous_application.csv')
    card_bal = pd.read_csv(raw / 'credit_card_balance.csv')
    print(f'  app_train:{len(app_train):,}  app_test:{len(app_test):,}  '
          f'bureau:{len(bureau):,}  prev_app:{len(prev_app):,}  card_bal:{len(card_bal):,}')

    print('\n=== Building categorical vocab (train+test+bureau+prev+card) ===')
    cat_vocab = build_cat_vocab(
        pd.concat([app_train, app_test], ignore_index=True),
        bureau, prev_app, card_bal,
    )
    with open(out / 'cat_vocab.json', 'w') as f:
        json.dump(cat_vocab, f, indent=2)
    for t, fields in cat_vocab.items():
        sizes = {f: len(v) + 1 for f, v in fields.items()}  # +1 for <UNK> at id 0
        print(f'  {t}: {sizes}')

    print('\n=== Computing normalizers (train stats only) ===')
    normalizer = {}
    normalizer.update(compute_normalizer(app_train, USER_NUMERIC_FIELDS))
    # Sub-table normalizers: compute on the union of train+test SK_ID_CURRs' rows
    # (test set has no labels but its feature distribution is informative; safe to use
    # because we're only computing mean/std of input features, not target)
    normalizer.update(compute_normalizer(bureau, BUREAU_NUMERIC_FIELDS))
    normalizer.update(compute_normalizer(prev_app, PREV_NUMERIC_FIELDS))
    normalizer.update(compute_normalizer(card_bal, CARD_NUMERIC_FIELDS))
    # Also query features
    normalizer.update(compute_normalizer(
        pd.concat([app_train, app_test], ignore_index=True), QUERY_FIELDS,
    ))
    with open(out / 'normalizer.json', 'w') as f:
        json.dump(normalizer, f, indent=2)
    print(f'  {len(normalizer)} fields normalized')

    print('\n=== Cleaning app data (sentinel + normalize user fields) ===')
    app_train = clean_app_df(app_train, normalizer)
    app_test = clean_app_df(app_test, normalizer)

    print('\n=== Normalizing sub-table numeric fields ===')
    bureau = normalize_subtable(bureau, BUREAU_NUMERIC_FIELDS, normalizer)
    prev_app = normalize_subtable(prev_app, PREV_NUMERIC_FIELDS, normalizer)
    card_bal = normalize_subtable(card_bal, CARD_NUMERIC_FIELDS, normalizer)
    # Query fields normalization happens at sample-build time (cheap, in app frame)

    print('\n=== Grouping sub-tables by SK_ID_CURR ===')
    bureau_groups = dict(list(bureau.groupby('SK_ID_CURR')))
    prev_groups = dict(list(prev_app.groupby('SK_ID_CURR')))
    card_groups = dict(list(card_bal.groupby('SK_ID_CURR')))
    print(f'  bureau:{len(bureau_groups):,}  prev:{len(prev_groups):,}  card:{len(card_groups):,}')

    print('\n=== Tokenizing pseudo-text (neobert tokenizer) ===')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_texts = [row_to_text(row) for _, row in app_train.iterrows()]
    test_texts = [row_to_text(row) for _, row in app_test.iterrows()]
    train_enc = tokenizer(train_texts, truncation=True, max_length=64, padding=False)
    test_enc = tokenizer(test_texts, truncation=True, max_length=64, padding=False)
    avg_len = np.mean([len(x) for x in train_enc['input_ids']])
    print(f'  avg text length: {avg_len:.1f} tokens, max: 64')

    print('\n=== Train/val split (stratified, seed=42) ===')
    positions = np.arange(len(app_train))
    train_pos, val_pos = train_test_split(
        positions, test_size=val_size, random_state=seed,
        stratify=app_train['TARGET'],
    )
    print(f'  train positions: {len(train_pos):,}, val positions: {len(val_pos):,}')

    with open(out / 'split.json', 'w') as f:
        json.dump({
            'train_sk_id_curr': app_train.iloc[train_pos]['SK_ID_CURR'].astype(int).tolist(),
            'val_sk_id_curr': app_train.iloc[val_pos]['SK_ID_CURR'].astype(int).tolist(),
            'test_sk_id_curr': app_test['SK_ID_CURR'].astype(int).tolist(),
            'seed': seed,
            'val_size': val_size,
        }, f)

    print('\n=== Building samples (streaming to disk) ===')
    # Route app_train samples by position into train.pkl or val.pkl
    train_pos_set = set(train_pos.tolist())
    val_pos_set = set(val_pos.tolist())
    def route_labeled(i):
        return 'train' if i in train_pos_set else 'val'

    labeled_paths = {
        'train': out / 'samples_train.pkl',
        'val': out / 'samples_val.pkl',
    }
    stream_samples_to_files(
        app_train, bureau_groups, prev_groups, card_groups,
        cat_vocab, train_enc, labeled_paths,
        has_target=True, position_to_key=route_labeled,
        progress_label='train+val',
    )

    # Free app_train and train_enc after writing (no longer needed)
    del app_train, train_enc, train_texts
    import gc
    gc.collect()

    print('\n=== Building unlabeled test samples ===')
    stream_samples_to_files(
        app_test, bureau_groups, prev_groups, card_groups,
        cat_vocab, test_enc,
        {'all': out / 'samples_test_unlabeled.pkl'},
        has_target=False,
        progress_label='test',
    )
    del app_test, test_enc, test_texts
    del bureau_groups, prev_groups, card_groups
    gc.collect()

    print('\n=== File sizes ===')
    for name in ['samples_train.pkl', 'samples_val.pkl', 'samples_test_unlabeled.pkl']:
        p = out / name
        if p.exists():
            print(f'  {p.name}: {p.stat().st_size / 1024 / 1024:.1f} MB')

    print('\n=== Verifying (read first record of each file) ===')
    for name in ['samples_train.pkl', 'samples_val.pkl', 'samples_test_unlabeled.pkl']:
        p = out / name
        if not p.exists():
            continue
        with open(p, 'rb') as f:
            s = pickle.load(f)
        print(f'  {name}: first sample keys={sorted(s.keys())[:5]}... '
              f'user_numeric={tuple(s["user_numeric"].shape)} '
              f'bureau={tuple(s["bureau_numeric"].shape)} '
              f'text={tuple(s["text_input_ids"].shape)} '
              f'target={"yes" if "target" in s else "no"}')

    print('\nDone.')


if __name__ == '__main__':
    build_all()
