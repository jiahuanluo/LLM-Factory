"""Prepare Home Credit Model 1 samples from raw CSVs.

Usage:
  python scripts/prepare_home_credit_samples.py

For quick smoke test (subset only):
  python scripts/prepare_home_credit_samples.py --smoke
"""
import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from home_credit.sample_builder import build_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/home-credit')
    parser.add_argument('--out-dir', default='data/home-credit/processed')
    parser.add_argument('--tokenizer', default='models/neobert')
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true',
                        help='Run on first 2000 rows only for smoke testing')
    args = parser.parse_args()

    if args.smoke:
        # Monkey-patch pandas.read_csv to limit rows
        import pandas as pd
        _orig = pd.read_csv
        N = 2000
        def _limited(*a, **kw):
            kw.setdefault('nrows', N)
            return _orig(*a, **kw)
        pd.read_csv = _limited
        print(f'!! SMOKE MODE: limiting all CSVs to first {N} rows')
        # Smoke mode also redirects output dir
        args.out_dir = 'data/home-credit/processed_smoke'

    build_all(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        tokenizer_path=args.tokenizer,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
