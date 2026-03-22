"""
train_traditional.py — Traditional ML baselines for cross-sectional crypto ranking.

Models: OLS, ElasticNet, PCA Regression, PLS, Random Forest, Gradient Boosting.

Uses CURRENT features only (no lookback window) — standard for these methods.
Same data, feature configs, train/valid/test split as deep learning models.
Saves results in same NPZ format for evaluate.py compatibility.

Usage:
    python train_traditional.py --config config.json
"""

import argparse
import json
import os
import time
import warnings
from itertools import product

import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_loader import load_panel, make_splits, detect_trump_start, UNK


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_flat_data(data, time_indices, feature_indices):
    """
    Extract current-period features (no lookback) for traditional models.

    Returns
    -------
    X : np.ndarray (n_valid, n_features) — pooled feature matrix
    y : np.ndarray (n_valid,) — pooled targets
    meta : list of (t, n) — time step and asset index for each observation
    """
    feat_cols = [f + 1 for f in feature_indices]  # +1 because col 0 = target
    N = data.shape[1]

    X_list, y_list, meta_list = [], [], []
    for t in time_indices:
        for n in range(N):
            target = data[t, n, 0]
            if target <= UNK + 1:
                continue
            feats = data[t, n, feat_cols].copy()
            feats[feats <= UNK + 1] = 0.0  # match DL convention
            X_list.append(feats)
            y_list.append(target)
            meta_list.append((t, n))

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            meta_list)


def predict_cross_sectional(model, data, time_indices, feature_indices, pca_model=None):
    """
    Generate per-time-step cross-sectional predictions.

    Returns preds, tgts, msks dicts keyed by time step.
    """
    feat_cols = [f + 1 for f in feature_indices]
    N = data.shape[1]
    preds, tgts, msks = {}, {}, {}

    for t in time_indices:
        X_t = data[t][:, feat_cols].copy()  # (N, n_features)
        targets = data[t, :, 0].copy()
        mask = targets > UNK + 1
        X_t[X_t <= UNK + 1] = 0.0

        if pca_model is not None:
            X_t = pca_model.transform(X_t)

        pred = model.predict(X_t).astype(np.float32)
        if pred.ndim > 1:
            pred = pred.ravel()
        tgts[t] = np.where(mask, targets, 0.0).astype(np.float32)
        msks[t] = mask
        preds[t] = pred

    return preds, tgts, msks


# ═══════════════════════════════════════════════════════════════════════
# Sharpe ratio (reuse same logic as train.py)
# ═══════════════════════════════════════════════════════════════════════

def compute_sharpe(predictions, targets, masks, annualise=52):
    """Long-short Sharpe from cross-sectional predictions."""
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
        rank = np.argsort(np.argsort(-pred_valid))
        long_mask = rank < decile
        short_mask = rank >= n - decile
        r_long = actual_valid[long_mask].mean()
        r_short = actual_valid[short_mask].mean()
        weekly_returns.append(r_long - r_short)

    if len(weekly_returns) < 2:
        return 0.0
    weekly_returns = np.array(weekly_returns)
    return weekly_returns.mean() / (weekly_returns.std() + 1e-10) * np.sqrt(annualise)


# ═══════════════════════════════════════════════════════════════════════
# Model training with hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "ols": {
        "class": "LinearRegression",
        "grid": [{}],
    },
    "elasticnet": {
        "class": "ElasticNet",
        "grid_keys": ["alpha", "l1_ratio"],
        "defaults": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9, 1.0],
        },
    },
    "pca_regression": {
        "class": "PCA+LinearRegression",
        "grid_keys": ["n_components"],
        "defaults": {"n_components": [1, 2, 3, 5]},
    },
    "pls": {
        "class": "PLSRegression",
        "grid_keys": ["n_components"],
        "defaults": {"n_components": [1, 2, 3, 5]},
    },
    "random_forest": {
        "class": "RandomForestRegressor",
        "grid_keys": ["max_depth", "max_features"],
        "defaults": {
            "max_depth": [1, 2, 3, 5],
            "max_features": ["sqrt", 0.33, 0.5],
            "n_estimators": 300,
        },
    },
    "gradient_boosting": {
        "class": "GradientBoostingRegressor",
        "grid_keys": ["max_depth", "learning_rate", "n_estimators"],
        "defaults": {
            "max_depth": [1, 2],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 300],
        },
    },
}


def _build_model(model_name, params):
    """Instantiate a sklearn model with given params."""
    if model_name == "ols":
        return LinearRegression(), None
    elif model_name == "elasticnet":
        return ElasticNet(max_iter=10000, **params), None
    elif model_name == "pca_regression":
        pca = PCA(n_components=params["n_components"])
        lr = LinearRegression()
        return lr, pca
    elif model_name == "pls":
        return PLSRegression(n_components=params["n_components"]), None
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=42, n_jobs=-1
        ), None
    elif model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8, random_state=42
        ), None
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_param_grid(model_name, cfg_overrides):
    """Build parameter grid from defaults + config overrides."""
    mc = MODEL_CONFIGS[model_name]
    if model_name == "ols":
        return [{}]

    grid_keys = mc["grid_keys"]
    defaults = mc["defaults"]
    params = {k: cfg_overrides.get(k, defaults[k]) for k in defaults}

    # Only iterate over grid_keys; others are fixed
    iter_vals = []
    for k in grid_keys:
        v = params[k]
        iter_vals.append(v if isinstance(v, list) else [v])

    grid = []
    for combo in product(*iter_vals):
        p = {k: v for k, v in zip(grid_keys, combo)}
        # Add non-grid params
        for k, v in params.items():
            if k not in grid_keys:
                p[k] = v if not isinstance(v, list) else v[0]
        grid.append(p)
    return grid


def train_model(model_name, X_train, y_train, data, valid_idx, test_idx,
                feature_indices, cfg_overrides):
    """
    Train a traditional model with hyperparameter tuning on validation SR.

    Returns (best_model, best_pca, best_val_sr, best_params).
    """
    param_grid = _get_param_grid(model_name, cfg_overrides)
    n_features = X_train.shape[1]

    best_sr, best_model, best_pca, best_params = -999, None, None, None

    for params in param_grid:
        # Cap PCA/PLS n_components at feature count
        if "n_components" in params:
            params["n_components"] = min(params["n_components"], n_features)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, pca = _build_model(model_name, params)

                X_fit = X_train
                if pca is not None:
                    X_fit = pca.fit_transform(X_train)

                if model_name == "pls":
                    model.fit(X_fit, y_train.reshape(-1, 1))
                else:
                    model.fit(X_fit, y_train)

                # Evaluate on validation set
                preds_v, tgts_v, msks_v = predict_cross_sectional(
                    model, data, valid_idx, feature_indices, pca_model=pca
                )
                val_sr = compute_sharpe(preds_v, tgts_v, msks_v)

                if val_sr > best_sr:
                    best_sr = val_sr
                    best_model = model
                    best_pca = pca
                    best_params = params.copy()
        except Exception as e:
            print(f"      Error with {params}: {e}")
            continue

    return best_model, best_pca, best_sr, best_params


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

TREE_MODELS = {"random_forest", "gradient_boosting"}


def _build_tree_model_with_seed(model_name, params, seed):
    """Instantiate a RF or GBT model with a specific random seed."""
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=seed, n_jobs=-1
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.8, random_state=seed
        )
    raise ValueError(f"Not a tree model: {model_name}")


def run_all_models(cfg, data, dates, assets, train_idx, valid_idx, test_idx):
    """Train all traditional models × all feature configs.

    Ensemble policy:
    - OLS / ElasticNet / PCA-OLS / PLS: deterministic given hyperparameters,
      so a single model is equivalent to an ensemble. Best HP chosen via
      validation Sharpe.
    - Random Forest / Gradient Boosting: have random_state → trained with
      `tree_ensemble_seeds` different seeds (default 5) after HP selection,
      predictions averaged. Comparable in spirit to the DL 32-seed ensemble.
    """
    feat_configs = cfg["feature_configs"]
    trump_feat_set = set(cfg.get("trump_feature_indices", [44, 45, 46, 47, 48]))
    trump_start = cfg.get("_detected_train_start_week", 0)

    trad_cfg = cfg.get("traditional_models", {})
    tree_seeds = cfg.get("tree_ensemble_seeds", 5)

    # Compute EW and VW market portfolio once
    market_results = compute_market_portfolio(data, test_idx)

    model_names = ["ols", "elasticnet", "pca_regression", "pls",
                   "random_forest", "gradient_boosting"]

    for model_name in model_names:
        is_tree = model_name in TREE_MODELS
        print(f"\n{'═'*60}")
        print(f"  Model: {model_name.upper()}"
              + (f"  [ensemble={tree_seeds} seeds]" if is_tree else "  [deterministic]"))
        print(f"{'═'*60}")

        cfg_overrides = trad_cfg.get(model_name, {})
        all_results = {}

        for feat_name, feat_indices in feat_configs.items():
            has_trump = bool(trump_feat_set & set(feat_indices))
            if has_trump and trump_start > 0:
                effective_train_idx = range(trump_start, train_idx.stop)
                print(f"\n  {feat_name} ({len(feat_indices)} features) "
                      f"[Trump-aware: week {trump_start}–{train_idx.stop-1}]")
            else:
                effective_train_idx = train_idx
                print(f"\n  {feat_name} ({len(feat_indices)} features)")

            X_train, y_train, _ = prepare_flat_data(
                data, effective_train_idx, feat_indices
            )
            print(f"    Training samples: {len(y_train)}")

            # ── Hyperparameter selection (all models) ──
            model, pca, val_sr, best_params = train_model(
                model_name, X_train, y_train, data,
                valid_idx, test_idx, feat_indices, cfg_overrides
            )

            if model is None:
                print(f"    ✗ Training failed")
                continue

            print(f"    Best params: {best_params}")
            print(f"    Val SR: {val_sr:+.3f}")

            if is_tree:
                # ── Tree ensemble: average predictions from multiple seeds ──
                seed_pred_lists = {t: [] for t in test_idx}
                tgts_t, msks_t = None, None

                for seed in range(tree_seeds):
                    m = _build_tree_model_with_seed(model_name, best_params, seed)
                    m.fit(X_train, y_train)
                    preds_s, tgts_s, msks_s = predict_cross_sectional(
                        m, data, test_idx, feat_indices
                    )
                    for t in test_idx:
                        seed_pred_lists[t].append(preds_s[t])
                    if tgts_t is None:
                        tgts_t, msks_t = tgts_s, msks_s  # same across seeds

                preds_t = {t: np.mean(seed_pred_lists[t], axis=0)
                           for t in test_idx}
                test_sr = compute_sharpe(preds_t, tgts_t, msks_t)
                print(f"    Test SR (ensemble of {tree_seeds}): {test_sr:+.3f}")
            else:
                # ── Deterministic models: single best model ──
                preds_t, tgts_t, msks_t = predict_cross_sectional(
                    model, data, test_idx, feat_indices, pca_model=pca
                )
                test_sr = compute_sharpe(preds_t, tgts_t, msks_t)
                print(f"    Test SR: {test_sr:+.3f}")

            all_results[feat_name] = {
                "preds": preds_t,
                "tgts": tgts_t,
                "msks": msks_t,
                "val_sr": val_sr,
                "test_sr": test_sr,
            }

        _save_results(model_name, all_results, data, dates, assets,
                      test_idx, cfg["output_dir"])

    # Save market portfolio results
    _save_market_results(market_results, data, dates, assets, test_idx,
                         cfg["output_dir"])

    return market_results


def compute_market_portfolio(data, test_idx):
    """Compute equal-weighted and value-weighted (VW) market portfolio.

    VW uses actual market cap if available (from market_cap.npz or btc_panel.npz).
    Run `python data_sources/fetch_market_cap.py` inside deep_learning_for_crypto/
    to generate market_cap.npz, then VW will be used as the primary benchmark.
    """
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "deep_learning_for_crypto", "datasets")

    # Try dedicated market_cap.npz first, then fall back to btc_panel.npz
    market_cap = None
    for mcap_path in [
        os.path.join(datasets_dir, "market_cap.npz"),
        os.path.join(datasets_dir, "btc_panel.npz"),
    ]:
        try:
            npz = np.load(mcap_path, allow_pickle=True)
            market_cap = npz.get("market_cap", None)
            if market_cap is not None:
                print(f"  Loaded market cap from: {os.path.basename(mcap_path)}")
                break
        except Exception:
            continue

    # Equal-weighted market portfolio
    ew_returns = []
    vw_returns = []
    for t in test_idx:
        returns = data[t, :, 0]
        mask = returns > UNK + 1
        if mask.sum() < 10:
            continue
        ew_returns.append(returns[mask].mean())

        if market_cap is not None:
            mcap = market_cap[t, :]
            valid = mask & (mcap > 0)
            if valid.sum() >= 10:
                w = mcap[valid] / mcap[valid].sum()
                vw_returns.append((returns[valid] * w).sum())

    ew = np.array(ew_returns)
    ew_sr = ew.mean() / (ew.std() + 1e-10) * np.sqrt(52)

    result = {
        "ew_returns": ew,
        "ew_mean": ew.mean() * 100,
        "ew_sr": ew_sr,
        "T_weeks": len(ew),
    }

    if vw_returns:
        vw = np.array(vw_returns)
        vw_sr = vw.mean() / (vw.std() + 1e-10) * np.sqrt(52)
        result["vw_returns"] = vw
        result["vw_mean"] = vw.mean() * 100
        result["vw_sr"] = vw_sr

    return result


def _save_results(model_name, all_results, data, dates, assets, test_idx, output_dir):
    """Save model results in NPZ format compatible with evaluate.py."""
    os.makedirs(output_dir, exist_ok=True)
    save_data = {}

    test_times_set = set()
    for feat_name, res in all_results.items():
        key = feat_name.replace("+", "plus_").replace(" ", "_")
        save_data[f"{key}_ensemble_sr"] = np.array(res["test_sr"])
        save_data[f"{key}_val_srs"] = np.array([res["val_sr"]])
        save_data[f"{key}_test_srs"] = np.array([res["test_sr"]])

        for t in sorted(res["preds"].keys()):
            save_data[f"{key}_pred_t{t}"] = res["preds"][t]
            save_data[f"{key}_tgt_t{t}"] = res["tgts"][t]
            save_data[f"{key}_mask_t{t}"] = res["msks"][t]
            test_times_set.add(t)

    save_data["test_times"] = np.array(sorted(test_times_set))
    save_data["dates"] = dates
    save_data["assets"] = assets

    path = os.path.join(output_dir, f"{model_name}_results.npz")
    np.savez(path, **save_data)
    print(f"  Saved: {path}")


def _save_market_results(market_results, data, dates, assets, test_idx, output_dir):
    """Save market portfolio results as NPZ."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "market_portfolio.npz")
    save_dict = {}
    for k, v in market_results.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            save_dict[k] = np.array(v)
    np.savez(path, **save_dict)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train traditional ML baselines for crypto ranking"
    )
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    print("=" * 60)
    print("  Traditional ML Baselines")
    print("  Models: OLS, ElasticNet, PCA, PLS, RF, GBT")
    print("=" * 60)

    # Load data
    data, dates, assets, variables = load_panel(cfg["data_path"])
    T, N, _ = data.shape
    print(f"Panel: T={T} weeks, N={N} assets, M={len(variables)} features")
    print(f"Date range: {dates[0]} → {dates[-1]}")

    # Detect Trump data start
    trump_indices = cfg.get("trump_feature_indices", [44, 45, 46, 47, 48])
    train_start_week = cfg.get("train_start_week", None)
    if train_start_week is None:
        train_start_week = detect_trump_start(data, trump_indices)
    cfg["_detected_train_start_week"] = train_start_week
    print(f"Trump data starts: week {train_start_week} ({dates[train_start_week]})")

    # Split
    train_idx, valid_idx, test_idx = make_splits(
        T, cfg.get("train_ratio", 0.7), cfg.get("valid_ratio", 0.15)
    )
    print(f"Split: train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    t0 = time.time()
    market_results = run_all_models(
        cfg, data, dates, assets, train_idx, valid_idx, test_idx
    )
    elapsed = time.time() - t0

    # Print market portfolio summary
    print(f"\n{'═'*60}")
    print(f"  Market Portfolio Benchmark")
    print(f"{'═'*60}")
    print(f"  EW Market: mean={market_results['ew_mean']:+.2f}%/week, "
          f"SR={market_results['ew_sr']:+.3f}, T={market_results['T_weeks']}")
    if "vw_sr" in market_results:
        print(f"  VW Market: mean={market_results['vw_mean']:+.2f}%/week, "
              f"SR={market_results['vw_sr']:+.3f}")
    else:
        print("  VW Market: N/A (no market_cap data in btc_panel.npz)")

    print(f"\n{'═'*60}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
