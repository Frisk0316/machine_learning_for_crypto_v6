"""
granger_trump.py — Granger causality test for Trump social media features.

Addresses the reverse-causality concern: Trump may tweet IN RESPONSE to
market moves rather than BEFORE them. Granger causality tests whether past
Trump features add predictive power for future returns beyond past returns alone.

Test setup:
  H0: "Trump feature at t-1 does NOT help predict return at t"
  (controlling for AR lags of the return series)

  F-stat = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (T-k))
  where q = number of Trump lags added.

Testing against:
  1. Cross-sectional mean return (market-level signal, primary test)
  2. BTC and ETH individually (major assets)
  3. Summary: how many of N=86 assets reject H0 at 10%?

Data: training period only (no look-ahead into test set).
      Only weeks where Trump features are non-UNK (≥ 209 weeks).

Usage:
    python granger_trump.py --config config.json
    python granger_trump.py --config config.json --max-lags 4 --all-assets
"""

import argparse
import json
import numpy as np
from scipy import stats

from data_loader import load_panel, make_splits, detect_trump_start, UNK


# ═══════════════════════════════════════════════════════════════════════
# Granger causality test (manual OLS, no statsmodels dependency)
# ═══════════════════════════════════════════════════════════════════════

TRUMP_FEATURE_NAMES = {
    44: 'trump_post_count',
    45: 'trump_caps_ratio',
    46: 'trump_tariff_score',
    47: 'trump_crypto_score',
    48: 'trump_sentiment',
}


def granger_f_test(y, x, max_lags=4):
    """
    Bivariate Granger causality: does x Granger-cause y?

    For each lag order l = 1..max_lags, tests H0: "all l lags of x are zero
    in the regression y_t = c + Σ β_j y_{t-j} + Σ γ_k x_{t-k}".

    Returns list of dicts: {lag, F, p_value, df_num, df_denom}

    Parameters
    ----------
    y        : array (T,) — endogenous variable (weekly return)
    x        : array (T,) — exogenous variable (Trump feature, lagged)
    max_lags : int — maximum lag order to test
    """
    T = len(y)
    results = []

    for lag in range(1, max_lags + 1):
        n = T - lag  # effective sample size
        if n < lag * 3 + 5:
            break   # too few observations for this lag

        Y = y[lag:]

        # Restricted: Y ~ 1 + AR(1..lag)
        X_r = np.ones((n, 1 + lag))
        for j in range(lag):
            X_r[:, 1 + j] = y[lag - 1 - j: T - 1 - j]

        # Unrestricted: Y ~ 1 + AR(1..lag) + X(1..lag)
        X_u = np.zeros((n, 1 + lag + lag))
        X_u[:, : 1 + lag] = X_r
        for k in range(lag):
            X_u[:, 1 + lag + k] = x[lag - 1 - k: T - 1 - k]

        def ols_rss(A, b):
            coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            resid = b - A @ coef
            return float(np.dot(resid, resid))

        rss_r = ols_rss(X_r, Y)
        rss_u = ols_rss(X_u, Y)

        q = lag          # number of restrictions
        k = X_u.shape[1]  # params in unrestricted model

        if rss_u < 1e-12 or (n - k) <= 0:
            F, p = 0.0, 1.0
        else:
            F = ((rss_r - rss_u) / q) / (rss_u / (n - k))
            F = max(F, 0.0)
            p = float(1.0 - stats.f.cdf(F, dfn=q, dfd=n - k))

        results.append({'lag': lag, 'F': F, 'p': p,
                        'df_num': q, 'df_denom': n - k,
                        'rss_r': rss_r, 'rss_u': rss_u})

    return results


def best_lag_result(results):
    """Return the lag-1 result (most common choice) and the minimum p-value."""
    if not results:
        return None
    lag1 = results[0]  # always lag=1
    min_p = min(results, key=lambda r: r['p'])
    return lag1, min_p


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def extract_trump_period(data, train_idx, trump_feat_indices):
    """
    Return (returns_T_N, trump_T_5) for the training weeks where
    all Trump features are non-UNK.

    Panel convention: col 0 = next-week return target, col i+1 = feature i.

    Returns
    -------
    week_returns : (T', N) — cross-sectional weekly returns
    trump_feats  : (T', 5) — Trump features (market-wide, same for all assets)
    valid_mask   : (T', N) — bool mask (True = non-UNK target)
    t_indices    : list of week indices retained
    """
    trump_cols = [fi + 1 for fi in trump_feat_indices]   # +1: col 0 is target

    t_indices = []
    for t in train_idx:
        vals = data[t, 0, trump_cols]           # asset 0 (same for all)
        if np.all(vals > UNK + 1):
            t_indices.append(t)

    if not t_indices:
        return None, None, None, []

    T = len(t_indices)
    N = data.shape[1]
    week_returns = np.zeros((T, N), dtype=np.float64)
    trump_feats = np.zeros((T, len(trump_feat_indices)), dtype=np.float64)
    valid_mask = np.zeros((T, N), dtype=bool)

    for i, t in enumerate(t_indices):
        trump_feats[i] = data[t, 0, trump_cols]      # market-wide
        for n in range(N):
            ret = float(data[t, n, 0])
            valid = ret > UNK + 1
            week_returns[i, n] = ret if valid else np.nan
            valid_mask[i, n] = valid

    return week_returns, trump_feats, valid_mask, t_indices


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def _sig_stars(p):
    if p < 0.01:   return '***'
    if p < 0.05:   return '** '
    if p < 0.10:   return '*  '
    return '   '


def print_granger_results(feature_results, target_label):
    """Print Granger causality results for one target series."""
    print(f"\n  Target: {target_label}")
    print(f"    {'Feature':<22s} {'F(lag=1)':>10s} {'p(lag=1)':>9s} {'Sig':>4s}"
          f"  │  {'F(best lag)':>12s} {'p(best)':>8s} {'best lag':>8s}")
    print(f"    {'─'*75}")
    for feat_name, res_list in feature_results.items():
        if not res_list:
            print(f"    {feat_name:<22s} {'N/A':>10s}")
            continue
        r1 = res_list[0]                              # lag=1
        rb = min(res_list, key=lambda r: r['p'])      # best lag
        print(f"    {feat_name:<22s} {r1['F']:>10.3f} {r1['p']:>9.4f} {_sig_stars(r1['p'])}"
              f"  │  {rb['F']:>12.3f} {rb['p']:>8.4f}  lag={rb['lag']}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Granger causality test: do Trump features lead crypto returns?'
    )
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--max-lags', type=int, default=4,
                        help='Maximum lag order to test (default: 4)')
    parser.add_argument('--all-assets', action='store_true',
                        help='Test all N=86 assets and report rejection rate')
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    data, dates, assets, variables = load_panel(cfg['data_path'])
    T, N, _ = data.shape
    print(f"Panel: T={T} weeks, N={N} assets")

    trump_indices = cfg.get('trump_feature_indices', [44, 45, 46, 47, 48])
    trump_start = detect_trump_start(data, trump_indices)
    print(f"Trump data available from week {trump_start} ({dates[trump_start]})")

    train_idx, valid_idx, test_idx = make_splits(
        T, cfg.get('train_ratio', 0.7), cfg.get('valid_ratio', 0.15)
    )

    # Extract data for Trump period within training window
    week_returns, trump_feats, valid_mask, t_indices = extract_trump_period(
        data, train_idx, trump_indices
    )

    if week_returns is None or len(t_indices) < 20:
        print("Insufficient Trump-period training data for Granger test.")
        return

    T_trump = len(t_indices)
    print(f"Trump-period training weeks: {T_trump}  "
          f"({dates[t_indices[0]]} → {dates[t_indices[-1]]})")
    print(f"Max lags: {args.max_lags}")

    # ── 1. Test against cross-sectional mean return ──────────────────────
    print(f"\n{'='*80}")
    print(f"  Granger Causality Tests")
    print(f"  H0: Trump feature t-1 does NOT Granger-cause return t")
    print(f"  (one-sided: does past Trump activity PREDICT future returns?)")
    print(f"{'='*80}")

    cs_mean = np.nanmean(week_returns, axis=1)    # (T',) cross-sectional mean

    cs_results = {}
    for feat_idx, feat_name in TRUMP_FEATURE_NAMES.items():
        trump_col = trump_indices.index(feat_idx)
        x = trump_feats[:, trump_col]
        # Run test on full Trump-period series
        res = granger_f_test(cs_mean, x, max_lags=args.max_lags)
        cs_results[feat_name] = res

    print_granger_results(cs_results, "Cross-sectional mean return")

    # ── 2. BTC and ETH individually ───────────────────────────────────────
    for asset_name in ['BTC', 'ETH']:
        # Find asset index
        asset_names_upper = [str(a).upper() for a in assets]
        matches = [i for i, a in enumerate(asset_names_upper)
                   if asset_name in a or a in asset_name]
        if not matches:
            print(f"\n  {asset_name}: not found in asset list")
            continue
        n_idx = matches[0]

        # Only use weeks where this asset has valid returns
        valid_weeks = valid_mask[:, n_idx]
        if valid_weeks.sum() < 20:
            print(f"\n  {asset_name}: insufficient valid weeks ({valid_weeks.sum()})")
            continue

        y = week_returns[valid_weeks, n_idx]
        trump_valid = trump_feats[valid_weeks]

        indiv_results = {}
        for feat_idx, feat_name in TRUMP_FEATURE_NAMES.items():
            trump_col = trump_indices.index(feat_idx)
            x = trump_valid[:, trump_col]
            res = granger_f_test(y, x, max_lags=args.max_lags)
            indiv_results[feat_name] = res

        print_granger_results(indiv_results, f"{asset_name} (index {n_idx})")

    # ── 3. Panel summary: how many assets reject H0? ──────────────────────
    if args.all_assets:
        print(f"\n{'='*80}")
        print(f"  Panel Summary — Rejection rate at 10% for lag=1")
        print(f"  (Bonferroni-corrected threshold: p < {0.10 / N:.5f})")
        print(f"{'='*80}")
        print(f"  {'Feature':<22s} {'Reject @10%':>12s} {'Reject @5%':>11s}"
              f"  {'Reject @1%':>11s}  {'Bonferroni @10%':>15s}")
        print(f"  {'─'*75}")

        for feat_idx, feat_name in TRUMP_FEATURE_NAMES.items():
            trump_col = trump_indices.index(feat_idx)
            x = trump_feats[:, trump_col]

            p_vals = []
            for n in range(N):
                valid = valid_mask[:, n]
                if valid.sum() < 20:
                    continue
                y_n = week_returns[valid, n]
                x_n = x[valid]
                res = granger_f_test(y_n, x_n, max_lags=1)
                if res:
                    p_vals.append(res[0]['p'])

            if not p_vals:
                continue
            p_arr = np.array(p_vals)
            n_tested = len(p_arr)
            bonf_thresh = 0.10 / n_tested
            print(f"  {feat_name:<22s} "
                  f"{(p_arr < 0.10).sum():>5d}/{n_tested:<5d}  "
                  f"{(p_arr < 0.05).sum():>5d}/{n_tested:<5d}  "
                  f"{(p_arr < 0.01).sum():>5d}/{n_tested:<5d}  "
                  f"{(p_arr < bonf_thresh).sum():>5d}/{n_tested:<5d}")

    # ── 4. Interpretation note ────────────────────────────────────────────
    print(f"""
  Interpretation:
  * Reject H0 → Trump feature has predictive power BEYOND past returns.
    This supports including it as a predictor (leads market, not lags).
  * Fail to reject → Trump feature may lag rather than lead returns.
    Raises reverse-causality concern; inclusion may be noise.

  Note: Granger causality ≠ structural causality. Even a leading indicator
  may reflect shared information (e.g. both Trump tweets and returns react
  to the same macro event at different speeds).

  Significance: * p<0.10  ** p<0.05  *** p<0.01
""")


if __name__ == '__main__':
    main()
