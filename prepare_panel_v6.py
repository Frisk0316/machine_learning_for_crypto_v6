"""
prepare_panel_v6.py — Build btc_panel_v6.npz by augmenting btc_panel.npz
with 4 new granular ETF features from Farside Investors data.

New features (appended at the end of the feature dimension):
  Index 49: btc_etf_gbtc        — GBTC weekly net flow ($M, z-scored)
  Index 50: btc_etf_excl_gbtc   — BTC ETF Total ex-GBTC ($M, z-scored)
  Index 51: eth_etf_ethe        — ETHE weekly net flow ($M, z-scored)
  Index 52: eth_etf_excl_ethe   — ETH ETF Total ex-ETHE ($M, z-scored)

These complement the existing aggregated ETF features (indices 27-32) by
providing structural signals: GBTC/ETHE outflows are legacy Grayscale fund
holders converting to spot ETFs (supply-side pressure), while ex-GBTC/ex-ETHE
captures purely new institutional demand.

Usage:
    python prepare_panel_v6.py [--panel PATH] [--btc-etf PATH] [--eth-etf PATH] [--out PATH]
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

UNK = -99.99
ROLL_WINDOW = 52   # rolling z-score window (same as other macro features)
MIN_PERIODS = 4


def _parse_farside_value(s):
    """Convert Farside string value to float. (X) → -X, '-' or '' → NaN."""
    s = str(s).strip()
    if s in ('-', '', 'nan', 'None'):
        return float('nan')
    # Remove parentheses and convert to negative
    m = re.match(r'^\(([0-9.,]+)\)$', s)
    if m:
        return -float(m.group(1).replace(',', ''))
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return float('nan')


def _load_btc_etf(path):
    """
    Load Farside BTC spot ETF daily flows.

    CSV structure:
      Row 0: fund names header (IBIT, FBTC, ..., GBTC, BTC, Total)
      Row 1: fee row (skip)
      Row 2: 'Date' header row
      Row 3+: data rows

    Returns DataFrame with Date index and columns: GBTC, Total.
    """
    raw = pd.read_csv(path, header=None)

    # Row 0 is the fund name header; find column indices for GBTC and Total
    headers = raw.iloc[0].tolist()
    gbtc_col = None
    total_col = None
    for i, h in enumerate(headers):
        h = str(h).strip()
        if h == 'GBTC':
            gbtc_col = i
        if h == 'Total':
            total_col = i

    if gbtc_col is None or total_col is None:
        raise ValueError(f"Could not find GBTC or Total column in {path}")

    # Data starts from row 3 (0-indexed); column 0 is date
    data = raw.iloc[3:].copy()
    data.columns = range(data.shape[1])
    data = data[data[0].notna()].copy()
    data = data[~data[0].astype(str).str.strip().isin(['', 'Date', 'Fee'])].copy()

    dates = pd.to_datetime(data[0].astype(str).str.strip(), dayfirst=True, errors='coerce')
    gbtc = data[gbtc_col].apply(_parse_farside_value).values
    total = data[total_col].apply(_parse_farside_value).values

    df = pd.DataFrame({
        'gbtc': gbtc,
        'total': total,
    }, index=dates)
    df = df[df.index.notna()].sort_index()
    return df


def _load_eth_etf(path):
    """
    Load Farside ETH spot ETF daily flows.

    CSV structure:
      Row 0: fund category (Blackrock, Fidelity, ...)
      Row 1: fund ticker (ETHA, FETH, ..., ETHE, ETH, Total)
      Row 2: fee row (skip)
      Row 3: 'Seed' row (skip)
      Row 4: 'Date' header row
      Row 5+: data rows

    Returns DataFrame with Date index and columns: ethe, total.
    """
    raw = pd.read_csv(path, header=None)

    # Row 0 has category names (including "Total"); row 1 has ticker names (including "ETHE")
    ethe_col = None
    total_col = None
    for i, h in enumerate(raw.iloc[1].tolist()):
        if str(h).strip() == 'ETHE':
            ethe_col = i
    for i, h in enumerate(raw.iloc[0].tolist()):
        if str(h).strip() == 'Total':
            total_col = i

    if ethe_col is None or total_col is None:
        raise ValueError(f"Could not find ETHE (row1) or Total (row0) column in {path}. "
                         f"Row0: {raw.iloc[0].tolist()}, Row1: {raw.iloc[1].tolist()}")

    # Data starts from row 5; column 0 is date
    data = raw.iloc[5:].copy()
    data.columns = range(data.shape[1])
    data = data[data[0].notna()].copy()
    data = data[~data[0].astype(str).str.strip().isin(['', 'Date', 'Fee', 'Seed'])].copy()

    dates = pd.to_datetime(data[0].astype(str).str.strip(), dayfirst=True, errors='coerce')
    ethe = data[ethe_col].apply(_parse_farside_value).values
    total = data[total_col].apply(_parse_farside_value).values

    df = pd.DataFrame({
        'ethe': ethe,
        'total': total,
    }, index=dates)
    df = df[df.index.notna()].sort_index()
    return df


def _daily_to_weekly(daily_df, week_dates):
    """
    Aggregate daily ETF flows to weekly (aligned to panel's week-end dates).

    Each week sums flows from [prev_week_end+1 day, week_end].
    NaN daily values are treated as 0 (no flow reported).

    Returns ndarray of shape (T,) for each column.
    """
    week_dates_ts = pd.DatetimeIndex([pd.Timestamp(str(d)) for d in week_dates])
    results = {col: np.full(len(week_dates_ts), np.nan) for col in daily_df.columns}

    for i, wd in enumerate(week_dates_ts):
        # Week window: (previous week end, current week end]
        if i == 0:
            start = wd - pd.Timedelta(days=7)
        else:
            start = week_dates_ts[i - 1]
        mask = (daily_df.index > start) & (daily_df.index <= wd)
        week_data = daily_df[mask]
        if len(week_data) == 0:
            continue
        for col in daily_df.columns:
            vals = week_data[col].fillna(0).values
            results[col][i] = vals.sum()

    return results


def _rolling_zscore(arr, window=52, min_periods=4):
    """Apply rolling z-score normalization (same as prepare_btc_data.py)."""
    ser = pd.Series(np.where(np.isfinite(arr), arr, np.nan))
    roll_mean = ser.rolling(window, min_periods=min_periods).mean()
    roll_std  = ser.rolling(window, min_periods=min_periods).std()
    out = np.full(len(arr), UNK, dtype=np.float64)
    for t in range(len(arr)):
        if not np.isfinite(arr[t]):
            continue
        mu  = roll_mean.iloc[t]
        std = roll_std.iloc[t]
        if pd.isna(mu) or pd.isna(std) or std < 1e-10:
            continue
        out[t] = float(np.clip((arr[t] - mu) / std, -3.0, 3.0))
    return out


def build_v6_panel(panel_path, btc_etf_path, eth_etf_path):
    """
    Load original panel and append 4 new ETF features.

    New feature indices: 49 (gbtc), 50 (btc_excl_gbtc), 51 (ethe), 52 (eth_excl_ethe)
    """
    print(f"Loading panel: {panel_path}")
    npz = np.load(panel_path, allow_pickle=True)
    data   = npz['data']      # (T, N, M+1)
    dates  = npz['date']
    assets = npz['wficn']
    variables = npz['variable']

    T, N, _ = data.shape
    print(f"  Shape: T={T}, N={N}, features={data.shape[2]-1}")

    # Load and aggregate ETF flows
    print(f"\nLoading BTC ETF data: {btc_etf_path}")
    btc_df = _load_btc_etf(btc_etf_path)
    print(f"  {len(btc_df)} trading days ({btc_df.index[0].date()} → {btc_df.index[-1].date()})")

    print(f"Loading ETH ETF data: {eth_etf_path}")
    eth_df = _load_eth_etf(eth_etf_path)
    print(f"  {len(eth_df)} trading days ({eth_df.index[0].date()} → {eth_df.index[-1].date()})")

    # Aggregate to weekly
    btc_weekly = _daily_to_weekly(btc_df, dates)
    eth_weekly = _daily_to_weekly(eth_df, dates)

    # Derive the 4 new features
    btc_gbtc_raw     = btc_weekly['gbtc']                             # GBTC flow
    btc_excl_raw     = btc_weekly['total'] - np.where(               # Total ex-GBTC
        np.isfinite(btc_weekly['gbtc']), btc_weekly['gbtc'], 0.0)
    eth_ethe_raw     = eth_weekly['ethe']                             # ETHE flow
    eth_excl_raw     = eth_weekly['total'] - np.where(               # Total ex-ETHE
        np.isfinite(eth_weekly['ethe']), eth_weekly['ethe'], 0.0)

    # Fill NaN → UNK before z-scoring so _rolling_zscore treats them as missing
    def _prep(arr):
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    print("\nNormalizing new ETF features (rolling z-score, window=52)...")
    feat_btc_gbtc    = _rolling_zscore(_prep(btc_gbtc_raw))
    feat_btc_excl    = _rolling_zscore(_prep(btc_excl_raw))
    feat_eth_ethe    = _rolling_zscore(_prep(eth_ethe_raw))
    feat_eth_excl    = _rolling_zscore(_prep(eth_excl_raw))

    new_feats = [feat_btc_gbtc, feat_btc_excl, feat_eth_ethe, feat_eth_excl]
    new_names = ['btc_etf_gbtc', 'btc_etf_excl_gbtc', 'eth_etf_ethe', 'eth_etf_excl_ethe']

    for name, arr in zip(new_names, new_feats):
        valid = (arr != UNK).sum()
        print(f"  {name}: {valid}/{T} weeks valid "
              f"({arr[arr != UNK].min():.2f} to {arr[arr != UNK].max():.2f})")

    # Broadcast each new feature across all N assets and append as new data columns
    # data shape: (T, N, M+1) — new shape: (T, N, M+5)
    extra = np.full((T, N, 4), UNK, dtype=np.float32)
    for j, arr in enumerate(new_feats):
        extra[:, :, j] = arr[:, np.newaxis]  # broadcast T → all N assets

    data_v6 = np.concatenate([data, extra], axis=2).astype(np.float32)
    print(f"\nNew panel shape: {data_v6.shape}  (added 4 ETF feature columns)")

    return data_v6, dates, assets, variables, new_names


def main():
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.abspath(__file__))
    datasets = os.path.join(base, '..', 'deep_learning_for_crypto', 'datasets')
    crypto_ml = os.path.join(base, '..', 'crypto_ml', 'data')

    parser.add_argument('--panel',   default=os.path.join(datasets, 'btc_panel.npz'))
    parser.add_argument('--btc-etf', default=os.path.join(crypto_ml, 'btc_spot_etf_from_farside.csv'))
    parser.add_argument('--eth-etf', default=os.path.join(crypto_ml, 'eth_spot_etf_from_farside.csv'))
    parser.add_argument('--out',     default=os.path.join(datasets, 'btc_panel_v6.npz'))
    args = parser.parse_args()

    data_v6, dates, assets, variables, new_names = build_v6_panel(
        args.panel, args.btc_etf, args.eth_etf
    )

    print(f"\nSaving to {args.out} ...")
    np.savez(args.out,
             data=data_v6,
             date=dates,
             wficn=assets,
             variable=variables)
    print(f"Done. New features added: {new_names}")
    print(f"Feature indices: btc_etf_gbtc=49, btc_etf_excl_gbtc=50, "
          f"eth_etf_ethe=51, eth_etf_excl_ethe=52")


if __name__ == '__main__':
    main()
