"""
ablation_study.py — Isolate each v6 improvement's contribution on LSTM performance.

v5 → v6 introduced three simultaneous changes, making it impossible to attribute
the SR gain (+1.08 → +2.46) to any single modification. This script trains
three ablation variants with individual changes turned on/off, using a reduced
seed count (default: 8) to keep compute manageable.

Ablation variants (LSTM model, single feature set):
  A. v5-equiv  : use_rank_norm=False, grad_accum_steps=1  (raw-return ListNet)
  B. +RankNorm : use_rank_norm=True,  grad_accum_steps=1  (rank-norm only)
  C. Full v6   : use_rank_norm=True,  grad_accum_steps=4  (rank-norm + accum)

Note: The improved VSN is TFT-only; it cannot be ablated on LSTM directly.
      To assess VSN impact, compare TFT A vs full TFT v6 separately.

Estimated runtime: ~4 min/run × 8 seeds × 3 variants ≈ 96 min on GPU.

Usage:
    python ablation_study.py --config config.json
    python ablation_study.py --config config.json --seeds 4 --feat "+Onchain"
"""

import argparse
import json
import os
import numpy as np
import torch

from data_loader import load_panel, make_splits, detect_trump_start
from train import train_one_seed, compute_sharpe


# ═══════════════════════════════════════════════════════════════════════
# Ablation variant definitions
# ═══════════════════════════════════════════════════════════════════════

ABLATION_VARIANTS = [
    {
        'id': 'A',
        'label': 'v5-equiv  (raw targets, accum=1)',
        'overrides': {'use_rank_norm': False, 'grad_accum_steps': 1},
    },
    {
        'id': 'B',
        'label': '+RankNorm (rank targets, accum=1)',
        'overrides': {'use_rank_norm': True, 'grad_accum_steps': 1},
    },
    {
        'id': 'C',
        'label': 'Full v6   (rank targets, accum=4)',
        'overrides': {'use_rank_norm': True, 'grad_accum_steps': 4},
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def run_variant(base_cfg, variant, data, train_idx, valid_idx, test_idx,
                feat_name, feat_indices, num_seeds, device):
    """Train one ablation variant across num_seeds seeds, return ensemble SR."""
    cfg = {**base_cfg, **variant['overrides']}
    cfg['model_type'] = 'lstm'   # ablation always on LSTM

    seed_preds = None
    seed_tgts = None
    seed_msks = None
    seed_srs = []

    print(f"\n  [{variant['id']}] {variant['label']}")
    print(f"      use_rank_norm={cfg['use_rank_norm']}  "
          f"grad_accum_steps={cfg['grad_accum_steps']}")

    for seed in range(num_seeds):
        res = train_one_seed(
            cfg, data, train_idx, valid_idx, test_idx,
            feat_indices, feat_name, seed, device,
        )
        seed_srs.append(res['test_sr'])

        if seed_preds is None:
            seed_preds = {t: [res['predictions'][t]] for t in res['predictions']}
            seed_tgts = res['targets']
            seed_msks = res['masks']
        else:
            for t in res['predictions']:
                seed_preds[t].append(res['predictions'][t])

    # Ensemble
    ens_preds = {t: np.mean(seed_preds[t], axis=0) for t in seed_preds}
    ens_sr = compute_sharpe(ens_preds, seed_tgts, seed_msks)

    return {
        'id': variant['id'],
        'label': variant['label'],
        'ensemble_sr': ens_sr,
        'mean_seed_sr': float(np.mean(seed_srs)),
        'std_seed_sr': float(np.std(seed_srs)),
        'seed_srs': seed_srs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_ablation_table(results, feat_name, num_seeds):
    print(f"\n{'='*72}")
    print(f"  Ablation Study — LSTM | {feat_name} | {num_seeds} seeds per variant")
    print(f"  Ens SR = ensemble of {num_seeds} seeds  |  ΔSR = change from previous row")
    print(f"{'='*72}")
    print(f"  {'Variant':<38s} {'Ens SR':>7s} {'Mean SR':>8s} {'Std':>6s} {'ΔSR':>7s}")
    print(f"  {'─'*68}")

    prev_sr = None
    for r in results:
        delta = f"{r['ensemble_sr'] - prev_sr:+.3f}" if prev_sr is not None else "  base"
        print(f"  [{r['id']}] {r['label']:<36s} "
              f"{r['ensemble_sr']:+7.3f} {r['mean_seed_sr']:+8.3f} "
              f"{r['std_seed_sr']:6.3f} {delta:>7s}")
        prev_sr = r['ensemble_sr']

    total_gain = results[-1]['ensemble_sr'] - results[0]['ensemble_sr']
    print(f"  {'─'*68}")
    print(f"  Total improvement A → C: {total_gain:+.3f} SR")


def save_ablation_csv(results, feat_name, num_seeds, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'ablation_study.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Variant_ID', 'Label', 'Feature_Set', 'Num_Seeds',
                    'Ensemble_SR', 'Mean_Seed_SR', 'Std_Seed_SR'])
        for r in results:
            w.writerow([r['id'], r['label'], feat_name, num_seeds,
                        f"{r['ensemble_sr']:.3f}",
                        f"{r['mean_seed_sr']:.3f}",
                        f"{r['std_seed_sr']:.3f}"])
    print(f"\n  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Ablation study isolating v5→v6 improvements on LSTM'
    )
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--seeds', type=int, default=32,
                        help='Seeds per variant (default: 32)')
    parser.add_argument('--feat', default='Price+Technical',
                        help='Feature set to use (default: Price+Technical)')
    args = parser.parse_args()

    with open(args.config) as fh:
        base_cfg = json.load(fh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Ablation: LSTM | feat={args.feat} | {args.seeds} seeds/variant")

    data, dates, assets, variables = load_panel(base_cfg['data_path'])
    T, N, _ = data.shape
    print(f"Panel: T={T}, N={N}")

    trump_indices = base_cfg.get('trump_feature_indices', [44, 45, 46, 47, 48])
    trump_start = detect_trump_start(data, trump_indices)
    base_cfg['_detected_train_start_week'] = trump_start

    train_idx, valid_idx, test_idx = make_splits(
        T, base_cfg.get('train_ratio', 0.7), base_cfg.get('valid_ratio', 0.15)
    )

    feat_configs = base_cfg['feature_configs']
    if args.feat not in feat_configs:
        print(f"Unknown feature set '{args.feat}'. Available: {list(feat_configs.keys())}")
        return
    feat_indices = feat_configs[args.feat]

    # Run ablation
    results = []
    for variant in ABLATION_VARIANTS:
        r = run_variant(
            base_cfg, variant, data, train_idx, valid_idx, test_idx,
            args.feat, feat_indices, args.seeds, device,
        )
        results.append(r)

    print_ablation_table(results, args.feat, args.seeds)
    out_dir = base_cfg.get('figure_dir', './outputs')
    save_ablation_csv(results, args.feat, args.seeds, out_dir)


if __name__ == '__main__':
    main()
