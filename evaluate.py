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
# Statistical inference helpers
# ═══════════════════════════════════════════════════════════════════════

def newey_west_tstat(returns, max_lag=None):
    """
    Newey-West HAC t-statistic for mean weekly return != 0.

    Corrects for autocorrelation in portfolio returns using the Bartlett
    kernel. Bandwidth follows the data-driven rule in Newey & West (1994):
    max_lag = max(1, floor(4 * (T/100)^{2/9})).

    Parameters
    ----------
    returns : array (T,) — weekly long-short returns
    max_lag : int or None — HAC bandwidth (None = data-driven)

    Returns
    -------
    t_nw : float — NW-corrected t-statistic
    se_nw : float — NW standard error of the mean
    """
    T = len(returns)
    if T < 4:
        return 0.0, 1e-10
    if max_lag is None:
        max_lag = max(1, int(4 * (T / 100) ** (2 / 9)))
    mu = returns.mean()
    e = returns - mu
    # Newey-West HAC variance with Bartlett kernel
    gamma_0 = np.dot(e, e) / T
    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        gamma_j = np.dot(e[j:], e[:-j]) / T
        weight = 1.0 - j / (max_lag + 1)   # Bartlett kernel
        nw_var += 2.0 * weight * gamma_j
    nw_var = max(nw_var, 1e-10)
    se_nw = np.sqrt(nw_var / T)
    return float(mu / se_nw), float(se_nw)


def sharpe_bootstrap_ci(returns, n_bootstrap=1000, ci=0.95, annualise=52, seed=42):
    """
    Circular block bootstrap confidence interval for the annualised Sharpe Ratio.

    Block bootstrap (Politis & Romano 1994) preserves the autocorrelation
    structure of weekly returns. Block length b = max(2, floor(T^{1/3})).

    Parameters
    ----------
    returns     : array (T,) — weekly long-short returns
    n_bootstrap : int — bootstrap replications
    ci          : float — confidence level (0.95 → 95% CI)
    annualise   : int — annualisation factor (52 for weekly)
    seed        : int — RNG seed for reproducibility

    Returns
    -------
    (lower, upper) : float — CI bounds at level `ci`
    """
    T = len(returns)
    if T < 8:
        sr = returns.mean() / (returns.std() + 1e-10) * np.sqrt(annualise)
        return float(sr), float(sr)
    block_size = max(2, int(T ** (1 / 3)))
    rng = np.random.default_rng(seed)
    sr_boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        n_blocks = int(np.ceil(T / block_size))
        starts = rng.integers(0, T, size=n_blocks)
        sample = np.concatenate([
            returns[np.arange(s, s + block_size) % T] for s in starts
        ])[:T]
        sr_boots[b] = sample.mean() / (sample.std() + 1e-10) * np.sqrt(annualise)
    alpha = (1.0 - ci) / 2.0
    return (float(np.percentile(sr_boots, alpha * 100)),
            float(np.percentile(sr_boots, (1 - alpha) * 100)))


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

    t_nw_pw, _ = newey_west_tstat(weekly_pw)
    t_nw_ew, _ = newey_west_tstat(weekly_ew)
    sr_ci_pw = sharpe_bootstrap_ci(weekly_pw, annualise=annualise)
    sr_ci_ew = sharpe_bootstrap_ci(weekly_ew, annualise=annualise)

    return {
        'mean_pw': mean_pw, 'mean_ew': mean_ew,
        'sr_pw': sr_pw, 'sr_ew': sr_ew,
        't_pw': t_pw, 't_nw_pw': t_nw_pw, 't_nw_ew': t_nw_ew,
        'sr_ci_pw': sr_ci_pw, 'sr_ci_ew': sr_ci_ew,
        'T_weeks': T_weeks,
        'weekly_pw': weekly_pw, 'weekly_ew': weekly_ew,
        'decile_sr_pw': decile_sr_pw, 'decile_sr_ew': decile_sr_ew,
        # raw predictions stored for transaction cost analysis
        '_preds': preds, '_tgts': tgts, '_msks': msks,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════════

def print_table3(all_metrics, model_type):
    """Print Table 3: Long-short portfolio performance with NW t-stats and bootstrap CIs."""
    print(f"\n{'='*100}")
    print(f"  Table 3: Long-Short Portfolio Performance — {model_type.upper()} Ensemble")
    print(f"  t_NW = Newey-West HAC t-stat (NW 1994 bandwidth)  |  95% CI = block-bootstrap Sharpe CI")
    print(f"{'='*100}")
    header = (f"  {'Info Set':<20s} │ {'mean_PW%':>8s} {'t_NW':>6s} "
              f"{'SR_PW':>6s} {'95% CI (PW)':>16s} │ "
              f"{'SR_EW':>6s} {'95% CI (EW)':>16s} │ {'T':>3s}")
    print(header)
    print(f"  {'─'*95}")

    rows = []
    for name, m in all_metrics.items():
        ci_pw = m.get('sr_ci_pw', (float('nan'), float('nan')))
        ci_ew = m.get('sr_ci_ew', (float('nan'), float('nan')))
        ci_pw_str = f"[{ci_pw[0]:+.2f},{ci_pw[1]:+.2f}]"
        ci_ew_str = f"[{ci_ew[0]:+.2f},{ci_ew[1]:+.2f}]"
        t_nw = m.get('t_nw_pw', m['t_pw'])
        row = (f"  {name:<20s} │ {m['mean_pw']:+8.2f} {t_nw:+6.2f} "
               f"{m['sr_pw']:+6.2f} {ci_pw_str:>16s} │ "
               f"{m['sr_ew']:+6.2f} {ci_ew_str:>16s} │ {m['T_weeks']:3d}")
        print(row)
        rows.append({
            'Information set': name,
            'mean_PW (%)': f"{m['mean_pw']:.2f}",
            't_NW': f"{t_nw:.2f}",
            'SR_PW': f"{m['sr_pw']:.2f}",
            'CI_95_lo': f"{ci_pw[0]:.2f}",
            'CI_95_hi': f"{ci_pw[1]:.2f}",
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


def print_subperiod_analysis(all_metrics, model_type, n_windows=4, annualise=52):
    """
    Sub-period robustness: split test predictions into n_windows equal sub-periods.

    This is NOT a full walk-forward (models are not retrained per window).
    It tests whether performance is consistent across different market conditions
    within the test set — a necessary (though not sufficient) condition for
    out-of-sample validity.

    Significance: * p<0.10, ** p<0.05, *** p<0.01 (one-sided NW t-test, SR > 0).
    """
    print(f"\n{'='*75}")
    print(f"  Sub-Period Robustness Analysis — {model_type.upper()} ({n_windows} equal windows, PW)")
    print(f"  Note: models trained once on the fixed train set; test set split post-hoc.")
    print(f"{'='*75}")

    for name, m in all_metrics.items():
        weekly_pw = m['weekly_pw']
        T = len(weekly_pw)
        window_size = T // n_windows
        if window_size < 4:
            print(f"\n  {name}: test period too short ({T} weeks) for {n_windows} sub-windows")
            continue

        t_nw_full = m.get('t_nw_pw', float('nan'))
        print(f"\n  {name}  (full test: SR={m['sr_pw']:+.2f}, t_NW={t_nw_full:+.2f}, T={T}):")
        print(f"    {'Window':<12s} {'Weeks':>5s} {'SR_PW':>7s} {'t_NW':>7s}  Sig")
        print(f"    {'─'*40}")

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size if i < n_windows - 1 else T
            sub = weekly_pw[start:end]
            sr = sub.mean() / (sub.std() + 1e-10) * np.sqrt(annualise)
            t_nw, _ = newey_west_tstat(sub)
            sig = ('***' if t_nw > 3.09 else
                   ('** ' if t_nw > 2.33 else
                    ('*  ' if t_nw > 1.65 else '   ')))
            print(f"    W{i+1} (w{start+1:02d}–w{end:02d})  {end - start:>5d} {sr:+7.2f} {t_nw:+7.2f}  {sig}")

    print(f"\n  Significance (one-sided): * p<0.10 (t>1.65)  ** p<0.05 (t>2.33)  *** p<0.01 (t>3.09)")


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
        w.writerow(['Information set', 'mean_PW (%)', 't_NW', 'SR_PW',
                     'CI_95_lo', 'CI_95_hi', 'mean_EW (%)', 'SR_EW', 'T (weeks)'])
        for name, m in all_metrics.items():
            t_nw = m.get('t_nw_pw', m['t_pw'])
            ci = m.get('sr_ci_pw', (float('nan'), float('nan')))
            w.writerow([name, f"{m['mean_pw']:.2f}", f"{t_nw:.2f}",
                        f"{m['sr_pw']:.2f}", f"{ci[0]:.2f}", f"{ci[1]:.2f}",
                        f"{m['mean_ew']:.2f}", f"{m['sr_ew']:.2f}", m['T_weeks']])
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Transaction cost sensitivity
# ═══════════════════════════════════════════════════════════════════════

_COST_LEVELS_BPS = [0, 5, 10, 25, 50, 75, 100, 150, 200]


def compute_sr_after_costs(preds, tgts, msks, cost_one_way_bps, annualise=52):
    """
    Apply one-way transaction costs based on weekly portfolio turnover.

    For an equal-weight long-short decile strategy:
      - Each week, compare current long (short) decile to previous week's.
      - Turnover = fraction of positions that change (entered + exited) / 2.
      - Cost per side per week = turnover × cost_one_way_bps / 10000.
      - Both long and short sides incur costs independently.

    Parameters
    ----------
    cost_one_way_bps : float — one-way cost in basis points (≈ half bid-ask spread)

    Returns
    -------
    sr_net  : float — annualised net-of-cost Sharpe Ratio
    weekly_net : array — weekly net returns after costs
    avg_turnover : float — average weekly fractional turnover [0, 1]
    """
    weekly_net = []
    turnovers = []
    prev_long_idx = None
    prev_short_idx = None

    for t_key in sorted(preds.keys()):
        pred = preds[t_key]
        actual = tgts[t_key]
        mask = msks[t_key]

        if mask.sum() < 10:
            continue

        p = pred[mask]
        a = actual[mask]
        n = len(p)
        dec = max(1, n // 10)

        order = np.argsort(-p)
        long_idx = set(order[:dec])
        short_idx = set(order[-dec:])

        # Gross EW long-short return
        r_gross = a[order[:dec]].mean() - a[order[-dec:]].mean()

        # Turnover: fraction of portfolio changed (entry + exit / 2*dec)
        if prev_long_idx is not None:
            long_chg = len(long_idx.symmetric_difference(prev_long_idx)) / (2 * dec)
            short_chg = len(short_idx.symmetric_difference(prev_short_idx)) / (2 * dec)
        else:
            long_chg = 1.0   # first period: full new position
            short_chg = 1.0

        # Cost: turnover × cost per unit (long + short sides)
        cost = (long_chg + short_chg) / 2 * cost_one_way_bps / 10_000
        weekly_net.append(r_gross - cost)
        turnovers.append((long_chg + short_chg) / 2)

        prev_long_idx = long_idx
        prev_short_idx = short_idx

    weekly_net = np.array(weekly_net)
    sr_net = weekly_net.mean() / (weekly_net.std() + 1e-10) * np.sqrt(annualise)
    return float(sr_net), weekly_net, float(np.mean(turnovers))


def find_breakeven_cost(preds, tgts, msks, annualise=52, max_bps=500):
    """Binary search for the one-way cost level where net SR = 0."""
    lo, hi = 0.0, float(max_bps)
    for _ in range(30):
        mid = (lo + hi) / 2
        sr, _, _ = compute_sr_after_costs(preds, tgts, msks, mid, annualise)
        if sr > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def print_transaction_cost_analysis(all_metrics, model_type, annualise=52):
    """
    Print SR sensitivity table across different cost assumptions.

    Typical crypto bid-ask spreads (one-way):
      BTC/ETH (top tier)  :  1–5 bps
      Mid-caps            : 10–50 bps
      Small caps          : 50–200 bps
      Universe avg (86 coins, rough estimate) : ~20–40 bps
    """
    print(f"\n{'='*90}")
    print(f"  Transaction Cost Sensitivity — {model_type.upper()} (EW long-short)")
    print(f"  One-way cost = half bid-ask spread.  SR_gross assumes zero cost.")
    print(f"  Crypto universe avg (86 coins) roughly 20–40 bps one-way.")
    print(f"{'='*90}")

    for name, m in all_metrics.items():
        preds = m.get('_preds')
        tgts_ = m.get('_tgts')
        msks_ = m.get('_msks')
        if preds is None:
            continue   # skip if raw predictions not stored (compare-all mode)

        breakeven = find_breakeven_cost(preds, tgts_, msks_, annualise)

        print(f"\n  {name}  (gross SR_EW = {m['sr_ew']:+.2f}, "
              f"breakeven ≈ {breakeven:.0f} bps one-way)")
        print(f"    {'Cost (bps)':>10s} {'SR_net':>8s} {'Δ SR':>8s} {'Avg TO%':>9s}  Status")
        print(f"    {'─'*52}")

        sr_gross = m['sr_ew']
        for cost_bps in _COST_LEVELS_BPS:
            sr_net, _, avg_to = compute_sr_after_costs(
                preds, tgts_, msks_, cost_bps, annualise
            )
            delta = sr_net - sr_gross
            status = 'break-even' if abs(sr_net) < 0.05 else (
                'profitable' if sr_net > 0 else 'LOSS')
            print(f"    {cost_bps:>10d} {sr_net:+8.2f} {delta:+8.2f} "
                  f"{avg_to * 100:>8.1f}%  {status}")

    print(f"\n  Breakeven cost is the one-way spread at which net SR = 0.")
    print(f"  Avg TO% = average weekly fraction of portfolio rebalanced.")


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
                             'mean_pw': m['mean_pw'], 't_pw': m['t_pw'],
                             't_nw_pw': m.get('t_nw_pw', m['t_pw'])}
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

    # Print NW t-stat table
    print(f"\n{'='*100}")
    print(f"  Cross-Model Comparison: Newey-West t-statistic (PW, one-sided SR > 0)")
    print(f"  * p<0.10 (t>1.65)  ** p<0.05 (t>2.33)  *** p<0.01 (t>3.09)")
    print(f"{'='*100}")
    header_parts = [f"{'Info Set':<20s}"] + [f"{d:>9s}" for d in disp]
    print(f"  {' │ '.join(header_parts)}")
    print(f"  {'─'*(sum(len(p) for p in header_parts) + 3 * (len(header_parts) - 1))}")
    for feat_name in feat_names_list:
        parts = [f"{feat_name:<20s}"]
        for model_name in available:
            srs_m = model_srs[model_name]
            if feat_name in srs_m:
                t_nw = srs_m[feat_name].get('t_nw_pw', float('nan'))
                if not np.isnan(t_nw):
                    sig = ('***' if t_nw > 3.09 else
                           ('** ' if t_nw > 2.33 else
                            ('*  ' if t_nw > 1.65 else '   ')))
                    parts.append(f"{t_nw:+6.2f}{sig}")
                else:
                    parts.append(f"{'N/A':>9s}")
            else:
                parts.append(f"{'N/A':>9s}")
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
        for mn in available:
            header.extend([f'{DISPLAY_NAMES.get(mn,mn)}_SR_PW',
                           f'{DISPLAY_NAMES.get(mn,mn)}_SR_EW',
                           f'{DISPLAY_NAMES.get(mn,mn)}_t_NW'])
        w.writerow(header)

        for feat_name in feat_names_list:
            row = [feat_name]
            if market is not None:
                if has_vw:
                    row.append(f"{float(market['vw_sr']):.3f}")
                row.append(f"{float(market['ew_sr']):.3f}")
            for model_name in available:
                srs_m = model_srs[model_name]
                if feat_name in srs_m:
                    row.extend([f"{srs_m[feat_name]['sr_pw']:.3f}",
                                f"{srs_m[feat_name]['sr_ew']:.3f}",
                                f"{srs_m[feat_name].get('t_nw_pw', float('nan')):.2f}"])
                else:
                    row.extend(['N/A', 'N/A', 'N/A'])
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
    parser.add_argument('--walk-forward-windows', type=int, default=4,
                        help='Number of sub-periods for robustness analysis (default: 4)')
    parser.add_argument('--no-cost-analysis', action='store_true',
                        help='Skip transaction cost sensitivity analysis')
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
    print_subperiod_analysis(all_metrics, model_type, n_windows=args.walk_forward_windows)
    print_comparison_table(all_metrics, model_type)

    # Print market portfolio if available
    market = load_market_portfolio(cfg['output_dir'])
    if market is not None:
        print_market_portfolio(market)

    # Transaction cost sensitivity
    if not args.no_cost_analysis:
        print_transaction_cost_analysis(all_metrics, model_type)

    # Generate figures
    print(f"\nGenerating figures...")
    plot_cumulative_returns(all_metrics, dates, test_times, model_type, out_dir)
    plot_decile_bars(all_metrics, model_type, out_dir)
    plot_feature_importance(results, feat_names, model_type, out_dir)
    save_csv_tables(all_metrics, model_type, out_dir)

    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == '__main__':
    main()
