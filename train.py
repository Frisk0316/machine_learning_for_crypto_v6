"""
train.py — v6: Cross-sectional ranking training for LSTM / TFT / CrossSectionalGatedNet.

Key changes from v5:
  1. Rank-normalized ListNet targets — cross-sectional rank percentiles [0,1]
     instead of raw weekly returns. Ensures each week contributes equally
     to training regardless of return magnitude/volatility.
     Reference: Poh, Roberts, Zohren (2021) "Building Cross-Sectional
     Systematic Strategies By Learning to Rank".
  2. Gradient accumulation — effective batch = 86 × grad_accum_steps assets.
     Reduces gradient noise from micro-batches (86 assets/step in v5).
  3. CrossSectionalGatedNet (cs_gated) — pure cross-sectional baseline.
  4. Improved TFT VSN — shared per-variable GRN (closer to Lim et al. 2021).

Usage:
    python train.py --config config.json [--model tft|lstm|cs_gated]
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import (
    load_panel, make_splits, detect_trump_start,
    CrossSectionalDataset, get_cross_sectional_data
)
from models import build_model


# ═══════════════════════════════════════════════════════════════════════
# Ranking loss
# ═══════════════════════════════════════════════════════════════════════

def listnet_loss(pred, target, mask, temperature=1.0, use_rank_norm=True):
    """
    ListNet ranking loss: cross-entropy between softmax distributions.

    v6 fix: cross-sectional rank-normalized targets (rank percentiles [0,1])
    instead of raw weekly returns. This ensures each week contributes equally
    to the loss regardless of return magnitude. A crash week (±20% returns)
    and a quiet week (±0.5%) now produce the same target distribution entropy.

    Reference: Poh, Roberts, Zohren (2021) "Building Cross-Sectional
    Systematic Strategies By Learning to Rank".

    Parameters
    ----------
    pred          : Tensor (N,) — model scores for all N assets
    target        : Tensor (N,) — actual returns for all N assets
    mask          : Tensor (N,) bool — valid assets (non-missing targets)
    temperature   : float — softmax temperature (lower = sharper distribution)
    use_rank_norm : bool — v6 behaviour (True) or v5 raw-return targets (False)
    """
    pred_valid = pred[mask]
    target_valid = target[mask]

    if pred_valid.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    if use_rank_norm:
        # v6: rank-normalize → lowest return = 0.0, highest return = 1.0
        ranks = torch.argsort(torch.argsort(target_valid)).float()
        target_norm = ranks / (len(ranks) - 1)
    else:
        # v5-equivalent: use raw returns as softmax targets (scale-sensitive)
        target_norm = target_valid

    p_true = F.softmax(target_norm / temperature, dim=0)
    log_p_pred = F.log_softmax(pred_valid / temperature, dim=0)
    return -torch.sum(p_true * log_p_pred)


# ═══════════════════════════════════════════════════════════════════════
# Portfolio evaluation (for model selection on validation set)
# ═══════════════════════════════════════════════════════════════════════

def compute_sharpe(predictions, targets, masks, annualise=52):
    """
    Compute long-short Sharpe ratio from cross-sectional predictions.

    At each time step, rank assets by prediction, go long top decile,
    short bottom decile (equal-weight within each decile).

    Portfolio weighting: EW (equal-weight) is the standard choice in academic
    cross-sectional return prediction (cf. Gu, Kelly, Xiu 2020). It tests
    the model's ability to rank uniformly across the universe without bias
    toward large-cap assets. VW (market-cap-weighted) metrics are reported
    separately in evaluate.py once market_cap data is available.

    Note: features in btc_panel.npz are already cross-sectionally rank-
    normalized (cat. A-C) and time-series z-score normalized (cat. D-G)
    by prepare_btc_data.py. No additional normalization is applied here.
    """
    weekly_returns = []
    for t_key in sorted(predictions.keys()):
        pred = predictions[t_key]
        actual = targets[t_key]
        mask = masks[t_key]

        if mask.sum() < 10:
            continue

        pred_valid = pred[mask]
        actual_valid = actual[mask]
        n = len(pred_valid)
        decile = max(1, n // 10)

        rank = np.argsort(np.argsort(-pred_valid))  # 0 = highest pred
        long_mask = rank < decile
        short_mask = rank >= n - decile

        r_long = actual_valid[long_mask].mean()
        r_short = actual_valid[short_mask].mean()
        weekly_returns.append(r_long - r_short)

    if len(weekly_returns) < 2:
        return 0.0
    weekly_returns = np.array(weekly_returns)
    sr = weekly_returns.mean() / (weekly_returns.std() + 1e-10) * np.sqrt(annualise)
    return sr


def predict_cross_section(model, cs_data, device):
    """Run model on cross-sectional data, return predictions/targets/masks by time."""
    model.eval()
    preds, tgts, msks = {}, {}, {}
    with torch.no_grad():
        for t_key, batch in cs_data.items():
            seq = batch['sequences'].to(device)
            idx = batch['asset_indices'].to(device)
            pred = model(seq, idx).cpu().numpy()
            preds[t_key] = pred
            tgts[t_key] = batch['targets']
            msks[t_key] = batch['masks']
    return preds, tgts, msks


# ═══════════════════════════════════════════════════════════════════════
# Training loop (v5: cross-sectional + ranking loss)
# ═══════════════════════════════════════════════════════════════════════

def set_deterministic(seed):
    """Ensure full reproducibility across CPU and GPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_seed(cfg, data, train_idx, valid_idx, test_idx,
                   feature_indices, feat_name, seed, device):
    """Train a single model with cross-sectional ranking loss."""
    set_deterministic(seed)

    lookback = cfg['lookback']
    n_features = len(feature_indices)
    n_assets = data.shape[1]
    temperature = cfg.get('ranking_temperature', 1.0)
    warmup_epochs = cfg.get('warmup_epochs', 10)
    base_lr = cfg['learning_rate']
    accum_steps = cfg.get('grad_accum_steps', 4)
    use_rank_norm = cfg.get('use_rank_norm', True)

    # Cross-sectional dataset for training (each item = all assets at one time step)
    train_cs = CrossSectionalDataset(data, train_idx, feature_indices, lookback)
    valid_cs = get_cross_sectional_data(data, valid_idx, feature_indices, lookback)
    test_cs = get_cross_sectional_data(data, test_idx, feature_indices, lookback)

    # Build model
    model_type = cfg.get('model_type', 'tft')
    model = build_model(model_type, n_features, n_assets, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=base_lr,
                                weight_decay=cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        patience=cfg.get('lr_scheduler_patience', 10),
        factor=cfg.get('lr_scheduler_factor', 0.5),
        min_lr=1e-6
    )

    best_val_sr = -999
    best_state = None
    no_improve = 0
    patience = cfg.get('early_stopping_patience', 20)
    history = {'train_loss': [], 'val_sr': []}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [{model_type.upper()}] {feat_name} seed={seed} | "
          f"features={n_features} params={n_params:,} "
          f"cs_batches={len(train_cs)}")

    for epoch in range(cfg['epochs']):
        model.train()

        # ── Linear warmup ──
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # ── Train: gradient accumulation over cross-sectional time steps ──
        # Effective batch = 86 assets × accum_steps steps, reducing gradient noise.
        order = torch.randperm(len(train_cs))
        total_loss = 0
        optimizer.zero_grad()

        for step, idx in enumerate(order):
            seq, target, mask, asset_idx = train_cs[idx.item()]
            seq = seq.to(device)
            target = target.to(device)
            mask = mask.to(device)
            asset_idx = asset_idx.to(device)

            pred = model(seq, asset_idx)
            loss = listnet_loss(pred, target, mask, temperature, use_rank_norm)
            (loss / accum_steps).backward()
            total_loss += loss.item()

            # Update weights every accum_steps steps (or at end of epoch)
            if (step + 1) % accum_steps == 0 or (step + 1) == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(len(train_cs), 1)

        # ── Validate ──
        preds_v, tgts_v, msks_v = predict_cross_section(model, valid_cs, device)
        val_sr = compute_sharpe(preds_v, tgts_v, msks_v)
        history['train_loss'].append(avg_loss)
        history['val_sr'].append(val_sr)

        # ── LR scheduling (after warmup) ──
        if epoch >= warmup_epochs:
            scheduler.step(val_sr)

        if val_sr > best_val_sr:
            best_val_sr = val_sr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or no_improve == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"      epoch {epoch+1:3d} | loss={avg_loss:.4f} "
                  f"val_SR={val_sr:+.3f} best={best_val_sr:+.3f} lr={lr_now:.1e}")

        if no_improve >= patience:
            print(f"      Early stop at epoch {epoch+1}")
            break

    # ── Reload best and evaluate test ──
    model.load_state_dict(best_state)
    model.to(device)

    preds_t, tgts_t, msks_t = predict_cross_section(model, test_cs, device)
    test_sr = compute_sharpe(preds_t, tgts_t, msks_t)

    # Feature importance (TFT only)
    feat_importance = None
    if hasattr(model, 'get_feature_importance'):
        for batch in test_cs.values():
            model.eval()
            with torch.no_grad():
                model(batch['sequences'].to(device),
                      batch['asset_indices'].to(device))
            break
        feat_importance = model.get_feature_importance()

    print(f"    ✓ seed={seed} val_SR={best_val_sr:+.3f} test_SR={test_sr:+.3f}")

    return {
        'best_val_sr': best_val_sr,
        'test_sr': test_sr,
        'predictions': preds_t,
        'targets': tgts_t,
        'masks': msks_t,
        'history': history,
        'feat_importance': feat_importance,
        'state_dict': best_state,
    }


def train_ensemble(cfg, data, dates, assets, variables,
                   train_idx, valid_idx, test_idx, device):
    """Train all feature configs × all seeds, save results."""
    results = {}
    feat_configs = cfg['feature_configs']
    num_seeds = cfg.get('num_seeds', 8)
    model_type = cfg.get('model_type', 'tft')

    trump_feat_set = set(cfg.get('trump_feature_indices', [44, 45, 46, 47, 48]))
    trump_start = cfg.get('_detected_train_start_week', 0)

    for feat_name, feat_indices in feat_configs.items():
        # Use Trump-aware training start if this config includes Trump features
        has_trump = bool(trump_feat_set & set(feat_indices))
        if has_trump and trump_start > 0:
            effective_train_idx = range(trump_start, train_idx.stop)
            print(f"\n{'='*60}")
            print(f"  Feature set: {feat_name} ({len(feat_indices)} features)")
            print(f"  Trump-aware training: week {trump_start}–{train_idx.stop-1}"
                  f" ({len(effective_train_idx)} weeks, was {len(train_idx)})")
            print(f"{'='*60}")
        else:
            effective_train_idx = train_idx
            print(f"\n{'='*60}")
            print(f"  Feature set: {feat_name} ({len(feat_indices)} features)")
            print(f"{'='*60}")

        seed_results = []
        for seed in range(num_seeds):
            res = train_one_seed(
                cfg, data, effective_train_idx, valid_idx, test_idx,
                feat_indices, feat_name, seed, device
            )
            seed_results.append(res)

        # ── Ensemble predictions (average across seeds) ──
        test_times = sorted(seed_results[0]['predictions'].keys())
        ensemble_preds = {}
        ensemble_tgts = {}
        ensemble_msks = {}

        for t in test_times:
            preds_stack = np.stack([r['predictions'][t] for r in seed_results])
            ensemble_preds[t] = preds_stack.mean(axis=0)
            ensemble_tgts[t] = seed_results[0]['targets'][t]
            ensemble_msks[t] = seed_results[0]['masks'][t]

        ensemble_sr = compute_sharpe(ensemble_preds, ensemble_tgts, ensemble_msks)

        # Average feature importance across seeds
        fi_list = [r['feat_importance'] for r in seed_results
                   if r['feat_importance'] is not None]
        avg_fi = np.mean(fi_list, axis=0) if fi_list else None

        val_srs = [r['best_val_sr'] for r in seed_results]
        test_srs = [r['test_sr'] for r in seed_results]

        print(f"\n  ── {feat_name} Ensemble ──")
        print(f"  Val  SR (mean±std): {np.mean(val_srs):+.3f} ± {np.std(val_srs):.3f}")
        print(f"  Test SR (per-seed): {np.mean(test_srs):+.3f} ± {np.std(test_srs):.3f}")
        print(f"  Test SR (ensemble): {ensemble_sr:+.3f}")

        results[feat_name] = {
            'seed_results': seed_results,
            'ensemble_preds': ensemble_preds,
            'ensemble_tgts': ensemble_tgts,
            'ensemble_msks': ensemble_msks,
            'ensemble_sr': ensemble_sr,
            'avg_feat_importance': avg_fi,
            'val_srs': val_srs,
            'test_srs': test_srs,
        }

    # ── Save results ──
    save_dir = cfg['output_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_results.npz')

    save_data = {}
    for feat_name, res in results.items():
        key = feat_name.replace('+', 'plus_').replace(' ', '_')
        save_data[f'{key}_ensemble_sr'] = np.array(res['ensemble_sr'])
        save_data[f'{key}_val_srs'] = np.array(res['val_srs'])
        save_data[f'{key}_test_srs'] = np.array(res['test_srs'])
        if res['avg_feat_importance'] is not None:
            save_data[f'{key}_feat_importance'] = res['avg_feat_importance']
        for t in sorted(res['ensemble_preds'].keys()):
            save_data[f'{key}_pred_t{t}'] = res['ensemble_preds'][t]
            save_data[f'{key}_tgt_t{t}'] = res['ensemble_tgts'][t]
            save_data[f'{key}_mask_t{t}'] = res['ensemble_msks'][t]
    save_data['test_times'] = np.array(sorted(
        list(results.values())[0]['ensemble_preds'].keys()
    ))
    save_data['dates'] = dates
    save_data['assets'] = assets

    np.savez(save_path, **save_data)
    print(f"\n  Results saved to {save_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default=None, help='Override model_type')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if args.model:
        cfg['model_type'] = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {cfg.get('model_type', 'tft').upper()} (v5: ranking loss + cross-asset attention)")

    # Load data
    data, dates, assets, variables = load_panel(cfg['data_path'])
    T, N, _ = data.shape
    print(f"Panel: T={T} weeks, N={N} assets, M={len(variables)} features")
    print(f"Date range: {dates[0]} → {dates[-1]}")

    # Detect Trump data start
    trump_indices = cfg.get('trump_feature_indices', [44, 45, 46, 47, 48])
    train_start_week = cfg.get('train_start_week', None)
    if train_start_week is None:
        train_start_week = detect_trump_start(data, trump_indices)
    cfg['_detected_train_start_week'] = train_start_week
    print(f"Trump data starts: week {train_start_week} ({dates[train_start_week]})")

    # Split
    train_idx, valid_idx, test_idx = make_splits(
        T, cfg.get('train_ratio', 0.7), cfg.get('valid_ratio', 0.15)
    )
    print(f"Split: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    t0 = time.time()
    results = train_ensemble(
        cfg, data, dates, assets, variables,
        train_idx, valid_idx, test_idx, device
    )
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Total training time: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {cfg.get('model_type','tft').upper()} v5 Ensemble ({cfg['num_seeds']} seeds)")
    print(f"  Loss: ListNet ranking | Cross-asset attention: ON")
    print(f"{'='*60}")
    print(f"  {'Info Set':<20s} {'Ens SR':>8s} {'Mean Test SR':>13s} {'Std':>6s}")
    print(f"  {'-'*50}")
    for name, res in results.items():
        print(f"  {name:<20s} {res['ensemble_sr']:+8.3f} "
              f"{np.mean(res['test_srs']):+13.3f} "
              f"{np.std(res['test_srs']):6.3f}")


if __name__ == '__main__':
    main()
