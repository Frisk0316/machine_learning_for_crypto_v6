"""
evaluate.py — Portfolio evaluation, comparison tables, and visualisation
for LSTM / TFT / traditional ML cross-sectional crypto return prediction.

Usage:
    python evaluate.py --config config.json [--model tft]
    python evaluate.py --config config.json --compare-all   # cross-model comparison
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
# Load saved results
# ═══════════════════════════════════════════════════════════════════════

def load_results(npz_path, feat_configs):
    """Reconstruct predictions from saved NPZ."""
    f = np.load(npz_path, allow_pickle=True)
    test_times = f['test_times']
    dates = f['dates']
    assets = f['assets']

    results = {}
    for feat_name in feat_configs:
        key = feat_name.replace('+', 'plus_').replace(' ', '_')
        preds, tgts, msks = {}, {}, {}
        for t in test_times:
            preds[int(t)] = f[f'{key}_pred_t{t}']
            tgts[int(t)] = f[f'{key}_tgt_t{t}']
            msks[int(t)] = f[f'{key}_mask_t{t}']

        fi_key = f'{key}_feat_importance'
        results[feat_name] = {
            'ensemble_preds': preds,
            'ensemble_tgts': tgts,
            'ensemble_msks': msks,
            'ensemble_sr': float(f[f'{key}_ensemble_sr']),
            'val_srs': f[f'{key}_val_srs'],
            'test_srs': f[f'{key}_test_srs'],
            'avg_feat_importance': f[fi_key] if fi_key in f else None,
        }
    return results, dates, assets, test_times


# ═══════════════════════════════════════════════════════════════════════
# Portfolio metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(preds, tgts, msks, annualise=52):
    """
    Compute full portfolio metrics: EW and PW long-short returns.

    Returns dict with keys: mean_pw, mean_ew, sr_pw, sr_ew, t_pw, T_weeks,
                            decile_sr_pw, decile_sr_ew
    """
    weekly_pw, weekly_ew = [], []
    n_deciles = 10
    decile_returns_pw = {d: [] for d in range(n_deciles)}
    decile_returns_ew = {d: [] for d in range(n_deciles)}

    for t_key in sorted(preds.keys()):
        pred = preds[t_key]
        actual = tgts[t_key]
        mask = msks[t_key]

        if mask.sum() < 10:
            continue

        p = pred[mask]
        a = actual[mask]
        n = len(p)
        decile_size = max(1, n // n_deciles)

        # Rank by prediction (descending)
        order = np.argsort(-p)

        for d in range(n_deciles):
            start = d * decile_size
            end = start + decile_size if d < n_deciles - 1 else n
            idx = order[start:end]
            if len(idx) == 0:
                continue

            # EW
            r_ew = a[idx].mean()
            decile_returns_ew[d].append(r_ew)

            # PW (performance-weighted by |prediction|)
            w = np.abs(p[idx])
            w_sum = w.sum()
            if w_sum > 0:
                r_pw = (a[idx] * w / w_sum).sum()
            else:
                r_pw = r_ew
            decile_returns_pw[d].append(r_pw)

        # Long-short: top decile - bottom decile
        top_idx = order[:decile_size]
        bot_idx = order[-decile_size:]

        r_long_ew = a[top_idx].mean()
        r_short_ew = a[bot_idx].mean()
        weekly_ew.append(r_long_ew - r_short_ew)

        w_top = np.abs(p[top_idx])
        w_bot = np.abs(p[bot_idx])
        w_top_s = w_top.sum()
        w_bot_s = w_bot.sum()
        r_long_pw = (a[top_idx] * w_top / w_top_s).sum() if w_top_s > 0 else r_long_ew
        r_short_pw = (a[bot_idx] * w_bot / w_bot_s).sum() if w_bot_s > 0 else r_short_ew
        weekly_pw.append(r_long_pw - r_short_pw)

    weekly_pw = np.array(weekly_pw)
    weekly_ew = np.array(weekly_ew)
    T_weeks = len(weekly_pw)

    sr_pw = weekly_pw.mean() / (weekly_pw.std() + 1e-10) * np.sqrt(annualise)
    sr_ew = weekly_ew.mean() / (weekly_ew.std() + 1e-10) * np.sqrt(annualise)
    mean_pw = weekly_pw.mean() * 100
    mean_ew = weekly_ew.mean() * 100
    t_pw = weekly_pw.mean() / (weekly_pw.std() / np.sqrt(T_weeks) + 1e-10)

    # Per-decile Sharpe
    decile_sr_pw, decile_sr_ew = {}, {}
    for d in range(n_deciles):
        arr_pw = np.array(decile_returns_pw[d])
        arr_ew = np.array(decile_returns_ew[d])
        if len(arr_pw) > 1:
            decile_sr_pw[d] = arr_pw.mean() / (arr_pw.std() + 1e-10) * np.sqrt(annualise)
            decile_sr_ew[d] = arr_ew.mean() / (arr_ew.std() + 1e-10) * np.sqrt(annualise)
        else:
            decile_sr_pw[d] = 0.0
            decile_sr_ew[d] = 0.0

    return {
        'mean_pw': mean_pw, 'mean_ew': mean_ew,
        'sr_pw': sr_pw, 'sr_ew': sr_ew,
        't_pw': t_pw, 'T_weeks': T_weeks,
        'weekly_pw': weekly_pw, 'weekly_ew': weekly_ew,
        'decile_sr_pw': decile_sr_pw, 'decile_sr_ew': decile_sr_ew,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════

def print_table3(all_metrics, model_type):
    """Print Table 3: Long-short portfolio performance."""
    print(f"\n{'='*80}")
    print(f"  Table 3: Long-Short Portfolio Performance — {model_type.upper()} Ensemble")
    print(f"{'='*80}")
    header = (f"  {'Info Set':<20s} │ {'mean_PW%':>8s} {'t-stat':>8s} "
              f"{'SR_PW':>7s} │ {'mean_EW%':>8s} {'SR_EW':>7s} │ {'T':>3s}")
    print(header)
    print(f"  {'─'*72}")

    rows = []
    for name, m in all_metrics.items():
        row = (f"  {name:<20s} │ {m['mean_pw']:+8.2f} {m['t_pw']:+8.2f} "
               f"{m['sr_pw']:+7.2f} │ {m['mean_ew']:+8.2f} {m['sr_ew']:+7.2f} │ {m['T_weeks']:3d}")
        print(row)
        rows.append({
            'Information set': name,
            'mean_PW (%)': f"{m['mean_pw']:.2f}",
            't-stat_PW': f"{m['t_pw']:.2f}",
            'SR_PW': f"{m['sr_pw']:.2f}",
            'mean_EW (%)': f"{m['mean_ew']:.2f}",
            'SR_EW': f"{m['sr_ew']:.2f}",
            'T (weeks)': str(m['T_weeks']),
        })
    return rows


def print_table_a1(all_metrics, model_type):
    """Print Table A1: Decile portfolio Sharpe ratios."""
    print(f"\n{'='*80}")
    print(f"  Table A1: Decile Sharpe Ratios — {model_type.upper()} Ensemble")
    print(f"{'='*80}")

    for name, m in all_metrics.items():
        print(f"\n  {name}:")
        print(f"    {'Decile':<8s}", end='')
        for d in range(10):
            label = 'Top' if d == 0 else ('Bot' if d == 9 else f'D{d+1}')
            print(f" {label:>6s}", end='')
        print(f" {'L-S':>7s}")

        print(f"    {'SR(PW)':<8s}", end='')
        for d in range(10):
            print(f" {m['decile_sr_pw'][d]:+6.2f}", end='')
        spread = m['decile_sr_pw'][0] - m['decile_sr_pw'][9]
        print(f" {spread:+7.2f}")

        print(f"    {'SR(EW)':<8s}", end='')
        for d in range(10):
            print(f" {m['decile_sr_ew'][d]:+6.2f}", end='')
        spread = m['decile_sr_ew'][0] - m['decile_sr_ew'][9]
        print(f" {spread:+7.2f}")


def print_comparison_table(all_metrics, model_type):
    """Print comparison with v3 Gated FFN results."""
    v3_results = {
        'Price+Technical': {'sr_pw': 2.10, 'sr_ew': 2.22, 't_pw': 2.04},
        '+Onchain':        {'sr_pw': -0.95, 'sr_ew': 0.17, 't_pw': -0.92},
        'All':             {'sr_pw': 2.90, 'sr_ew': 1.88, 't_pw': 2.82},
    }

    print(f"\n{'='*80}")
    print(f"  Comparison: {model_type.upper()} vs. v3 Gated FFN (86 assets)")
    print(f"{'='*80}")
    header = (f"  {'Info Set':<20s} │ {'v3 FFN SR':>9s} {'TFT SR':>8s} "
              f"{'Δ SR':>7s} │ {'v3 t-stat':>9s} {f'{model_type} t':>8s}")
    print(header)
    print(f"  {'─'*65}")

    for name in all_metrics:
        if name in v3_results:
            v3 = v3_results[name]
            m = all_metrics[name]
            delta = m['sr_pw'] - v3['sr_pw']
            print(f"  {name:<20s} │ {v3['sr_pw']:+9.2f} {m['sr_pw']:+8.2f} "
                  f"{delta:+7.2f} │ {v3['t_pw']:+9.2f} {m['t_pw']:+8.2f}")


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(all_metrics, dates, test_times, model_type, out_dir):
    """Plot cumulative long-short returns by info set."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, m in all_metrics.items():
        cum_pw = np.cumsum(m['weekly_pw'])
        cum_ew = np.cumsum(m['weekly_ew'])
        t_dates = [dates[t] for t in sorted(test_times)[:len(cum_pw)]]

        axes[0].plot(range(len(cum_pw)), cum_pw, label=f"{name} (SR={m['sr_pw']:+.2f})")
        axes[1].plot(range(len(cum_ew)), cum_ew, label=f"{name} (SR={m['sr_ew']:+.2f})")

    for ax, title in zip(axes, ['Performance-Weighted (PW)', 'Equal-Weighted (EW)']):
        ax.set_title(f'{model_type.upper()} Long-Short: {title}')
        ax.set_xlabel('Test Week')
        ax.set_ylabel('Cumulative Return')
        ax.axhline(0, color='grey', ls='--', lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{model_type}_cumulative_returns.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(results, feat_names, model_type, out_dir):
    """Plot TFT variable importance from VSN weights."""
    fig, axes = plt.subplots(len(results), 1,
                             figsize=(10, 4 * len(results)),
                             squeeze=False)

    for i, (name, res) in enumerate(results.items()):
        fi = res.get('avg_feat_importance')
        if fi is None:
            axes[i, 0].text(0.5, 0.5, 'N/A (LSTM)', transform=axes[i, 0].transAxes,
                           ha='center', va='center')
            axes[i, 0].set_title(f'{name}: Variable Importance')
            continue

        config_feats = list(json.loads(
            open('config.json').read()
        )['feature_configs'][name])
        names = [feat_names[j] if j < len(feat_names) else f'f{j}' for j in config_feats]

        # Sort by importance
        order = np.argsort(fi)[::-1]
        fi_sorted = fi[order]
        names_sorted = [names[j] for j in order]

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(fi_sorted)))
        axes[i, 0].barh(range(len(fi_sorted)), fi_sorted, color=colors)
        axes[i, 0].set_yticks(range(len(fi_sorted)))
        axes[i, 0].set_yticklabels(names_sorted, fontsize=7)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_xlabel('VSN Weight (avg)')
        axes[i, 0].set_title(f'{name}: TFT Variable Selection Importance')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{model_type}_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_decile_bars(all_metrics, model_type, out_dir):
    """Plot decile Sharpe ratios as bar chart."""
    n_configs = len(all_metrics)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 4), squeeze=False)

    for i, (name, m) in enumerate(all_metrics.items()):
        deciles = list(range(10))
        sr_pw = [m['decile_sr_pw'][d] for d in deciles]
        sr_ew = [m['decile_sr_ew'][d] for d in deciles]
        labels = ['Top'] + [f'D{d+1}' for d in range(1, 9)] + ['Bot']

        x = np.arange(10)
        w = 0.35
        axes[0, i].bar(x - w/2, sr_pw, w, label='PW', color='steelblue')
        axes[0, i].bar(x + w/2, sr_ew, w, label='EW', color='coral')
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(labels, fontsize=7, rotation=45)
        axes[0, i].axhline(0, color='grey', ls='--', lw=0.5)
        axes[0, i].set_ylabel('Annualised Sharpe')
        axes[0, i].set_title(f'{name}')
        axes[0, i].legend(fontsize=7)
        axes[0, i].grid(alpha=0.3, axis='y')

    plt.suptitle(f'{model_type.upper()} Decile Sharpe Ratios', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{model_type}_decile_sharpe.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_csv_tables(all_metrics, model_type, out_dir):
    """Save performance tables as CSV."""
    import csv

    # Table 3
    path = os.path.join(out_dir, f'{model_type}_table3.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Information set', 'mean_PW (%)', 't-stat_PW', 'SR_PW',
                     'mean_EW (%)', 'SR_EW', 'T (weeks)'])
        for name, m in all_metrics.items():
            w.writerow([name, f"{m['mean_pw']:.2f}", f"{m['t_pw']:.2f}",
                        f"{m['sr_pw']:.2f}", f"{m['mean_ew']:.2f}",
                        f"{m['sr_ew']:.2f}", m['T_weeks']])
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Market portfolio benchmark
# ═══════════════════════════════════════════════════════════════════════

def load_market_portfolio(output_dir):
    """Load market portfolio results from NPZ."""
    path = os.path.join(output_dir, 'market_portfolio.npz')
    if not os.path.exists(path):
        return None
    f = np.load(path, allow_pickle=True)
    result = {}
    for k in f.files:
        result[k] = f[k]
    return result


def print_market_portfolio(market, label="Market Portfolio"):
    """Print market portfolio benchmark.

    VW Market (value-weighted by market cap) is the primary benchmark when
    available. EW Market is reported as a secondary reference.
    Run fetch_market_cap.py in deep_learning_for_crypto/data_sources/ to enable VW.
    """
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    ew_sr = float(market['ew_sr'])
    ew_mean = float(market['ew_mean'])
    T = int(market['T_weeks'])
    if 'vw_sr' in market:
        vw_sr = float(market['vw_sr'])
        vw_mean = float(market['vw_mean'])
        print(f"  VW Market [PRIMARY]:  mean={vw_mean:+.2f}%/week  SR={vw_sr:+.3f}  T={T}")
        print(f"  EW Market [reference]: mean={ew_mean:+.2f}%/week  SR={ew_sr:+.3f}")
    else:
        print(f"  EW Market: mean={ew_mean:+.2f}%/week, SR={ew_sr:+.3f}, T={T}")
        print("  VW Market: N/A — run fetch_market_cap.py to enable VW benchmark")


# ═══════════════════════════════════════════════════════════════════════
# Cross-model comparison (--compare-all)
# ═══════════════════════════════════════════════════════════════════════

ALL_MODEL_NAMES = [
    'tft', 'lstm', 'cs_gated',
    'ols', 'elasticnet', 'pca_regression', 'pls',
    'random_forest', 'gradient_boosting',
]

DISPLAY_NAMES = {
    'tft': 'TFT', 'lstm': 'LSTM', 'cs_gated': 'CS_Gated',
    'ols': 'OLS', 'elasticnet': 'ElasticNet',
    'pca_regression': 'PCA', 'pls': 'PLS',
    'random_forest': 'RF', 'gradient_boosting': 'GBT',
}


def compare_all_models(cfg):
    """Load all available model results and print cross-model comparison."""
    output_dir = cfg['output_dir']
    feat_configs = cfg['feature_configs']
    feat_names_list = list(feat_configs.keys())

    # Collect SR for each model × feature config
    model_srs = {}  # {model_name: {feat_name: sr_pw}}
    available = []

    for model_name in ALL_MODEL_NAMES:
        npz_path = os.path.join(output_dir, f'{model_name}_results.npz')
        if not os.path.exists(npz_path):
            continue
        try:
            results, dates, assets, test_times = load_results(npz_path, feat_configs)
            srs = {}
            for name, res in results.items():
                m = compute_portfolio_metrics(
                    res['ensemble_preds'], res['ensemble_tgts'], res['ensemble_msks']
                )
                srs[name] = {'sr_pw': m['sr_pw'], 'sr_ew': m['sr_ew'],
                             'mean_pw': m['mean_pw'], 't_pw': m['t_pw']}
            model_srs[model_name] = srs
            available.append(model_name)
        except Exception as e:
            print(f"  Warning: could not load {model_name}: {e}")

    if not available:
        print("  No model results found.")
        return

    # Load market portfolio
    market = load_market_portfolio(output_dir)

    has_vw = market is not None and 'vw_sr' in market

    def _market_cols(header_mode=True):
        """Return market benchmark column headers or values."""
        if market is None:
            return []
        if header_mode:
            cols = []
            if has_vw:
                cols.append(f"{'VW Mkt':>7s}")
            cols.append(f"{'EW Mkt':>7s}")
            return cols
        else:
            cols = []
            if has_vw:
                cols.append(f"{float(market['vw_sr']):+7.2f}")
            cols.append(f"{float(market['ew_sr']):+7.2f}")
            return cols

    disp = [DISPLAY_NAMES.get(m, m) for m in available]

    # Print comparison table (SR PW)
    print(f"\n{'='*100}")
    print(f"  Cross-Model Comparison: Long-Short SR (PW)")
    if has_vw:
        print(f"  Market benchmark: VW Market (value-weighted by market cap) [primary]")
    print(f"{'='*100}")

    header_parts = [f"{'Info Set':<20s}"] + _market_cols() + [f"{d:>8s}" for d in disp]
    print(f"  {' │ '.join(header_parts)}")
    print(f"  {'─'*(len(header_parts)*10)}")

    for feat_name in feat_names_list:
        parts = [f"{feat_name:<20s}"] + _market_cols(header_mode=False)
        for model_name in available:
            srs = model_srs[model_name]
            if feat_name in srs:
                parts.append(f"{srs[feat_name]['sr_pw']:+8.2f}")
            else:
                parts.append(f"{'N/A':>8s}")
        print(f"  {' │ '.join(parts)}")

    # Print EW SR table
    print(f"\n{'='*100}")
    print(f"  Cross-Model Comparison: Long-Short SR (EW)")
    print(f"{'='*100}")

    header_parts = [f"{'Info Set':<20s}"] + _market_cols() + [f"{d:>8s}" for d in disp]
    print(f"  {' │ '.join(header_parts)}")
    print(f"  {'─'*(len(header_parts)*10)}")

    for feat_name in feat_names_list:
        parts = [f"{feat_name:<20s}"] + _market_cols(header_mode=False)
        for model_name in available:
            srs = model_srs[model_name]
            if feat_name in srs:
                parts.append(f"{srs[feat_name]['sr_ew']:+8.2f}")
            else:
                parts.append(f"{'N/A':>8s}")
        print(f"  {' │ '.join(parts)}")

    # Save CSV
    out_dir = cfg.get('figure_dir', './outputs')
    os.makedirs(out_dir, exist_ok=True)
    import csv
    path = os.path.join(out_dir, 'cross_model_comparison.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Information set']
        if market is not None:
            if has_vw:
                header.append('VW_Market_SR')
            header.append('EW_Market_SR')
        for m in available:
            header.extend([f'{DISPLAY_NAMES.get(m,m)}_SR_PW',
                          f'{DISPLAY_NAMES.get(m,m)}_SR_EW'])
        w.writerow(header)

        for feat_name in feat_names_list:
            row = [feat_name]
            if market is not None:
                if has_vw:
                    row.append(f"{float(market['vw_sr']):.3f}")
                row.append(f"{float(market['ew_sr']):.3f}")
            for model_name in available:
                srs = model_srs[model_name]
                if feat_name in srs:
                    row.extend([f"{srs[feat_name]['sr_pw']:.3f}",
                               f"{srs[feat_name]['sr_ew']:.3f}"])
                else:
                    row.extend(['N/A', 'N/A'])
            w.writerow(row)
    print(f"\n  Saved: {path}")

    return model_srs, market


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model', default=None)
    parser.add_argument('--compare-all', action='store_true',
                        help='Print cross-model comparison table')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = cfg.get('figure_dir', './outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Cross-model comparison mode
    if args.compare_all:
        compare_all_models(cfg)
        # Also print market portfolio
        market = load_market_portfolio(cfg['output_dir'])
        if market is not None:
            print_market_portfolio(market)
        return

    model_type = args.model or cfg.get('model_type', 'tft')

    # Load results
    npz_path = os.path.join(cfg['output_dir'], f'{model_type}_results.npz')
    results, dates, assets, test_times = load_results(npz_path, cfg['feature_configs'])

    # Feature names
    feat_names = cfg.get('_feature_names', [f'f{i}' for i in range(50)])

    # Compute portfolio metrics
    all_metrics = {}
    for name, res in results.items():
        m = compute_portfolio_metrics(
            res['ensemble_preds'], res['ensemble_tgts'], res['ensemble_msks']
        )
        all_metrics[name] = m

    # Print tables
    print_table3(all_metrics, model_type)
    print_table_a1(all_metrics, model_type)
    print_comparison_table(all_metrics, model_type)

    # Print market portfolio if available
    market = load_market_portfolio(cfg['output_dir'])
    if market is not None:
        print_market_portfolio(market)

    # Generate figures
    print(f"\nGenerating figures...")
    plot_cumulative_returns(all_metrics, dates, test_times, model_type, out_dir)
    plot_decile_bars(all_metrics, model_type, out_dir)
    plot_feature_importance(results, feat_names, model_type, out_dir)
    save_csv_tables(all_metrics, model_type, out_dir)

    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == '__main__':
    main()
