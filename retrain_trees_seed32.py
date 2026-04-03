"""
Retrain only Random Forest and Gradient Boosting with tree_ensemble_seeds=32.
Reuses all functions from train_traditional.py unchanged.
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from train_traditional import (
    prepare_flat_data, predict_cross_sectional, compute_sharpe,
    train_model, _build_tree_model_with_seed, _save_results,
    TREE_MODELS, MODEL_CONFIGS,
)
from data_loader import load_panel, make_splits, detect_trump_start

import numpy as np

def retrain_trees(cfg_path="config.json"):
    with open(cfg_path) as f:
        cfg = json.load(f)

    data, dates, assets, _ = load_panel(cfg["data_path"])
    T, N, _ = data.shape

    trump_indices = cfg.get("trump_feature_indices", [44, 45, 46, 47, 48])
    train_start_week = cfg.get("train_start_week", None)
    if train_start_week is None:
        train_start_week = detect_trump_start(data, trump_indices)
    cfg["_detected_train_start_week"] = train_start_week

    train_idx, valid_idx, test_idx = make_splits(
        T, cfg.get("train_ratio", 0.7), cfg.get("valid_ratio", 0.15)
    )
    print(f"Panel T={T} N={N} | train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)}")

    feat_configs = cfg["feature_configs"]
    trump_feat_set = set(cfg.get("trump_feature_indices", [44, 45, 46, 47, 48]))
    trad_cfg = cfg.get("traditional_models", {})
    tree_seeds = cfg.get("tree_ensemble_seeds", 32)

    for model_name in ["random_forest", "gradient_boosting"]:
        print(f"\n{'═'*60}")
        print(f"  Model: {model_name.upper()}  [ensemble={tree_seeds} seeds]")
        print(f"{'═'*60}")

        cfg_overrides = trad_cfg.get(model_name, {})
        all_results = {}

        for feat_name, feat_indices in feat_configs.items():
            has_trump = bool(trump_feat_set & set(feat_indices))
            effective_train_idx = (
                range(train_start_week, train_idx.stop)
                if has_trump and train_start_week > 0 else train_idx
            )
            print(f"\n  {feat_name} ({len(feat_indices)} features)")

            X_train, y_train, _ = prepare_flat_data(data, effective_train_idx, feat_indices)
            print(f"    Training samples: {len(y_train)}")

            model, pca, val_sr, best_params = train_model(
                model_name, X_train, y_train, data,
                valid_idx, test_idx, feat_indices, cfg_overrides
            )
            if model is None:
                print("    ✗ Training failed"); continue

            print(f"    Best params: {best_params}  Val SR: {val_sr:+.3f}")

            seed_pred_lists = {t: [] for t in test_idx}
            tgts_t = msks_t = None

            for seed in range(tree_seeds):
                m = _build_tree_model_with_seed(model_name, best_params, seed)
                m.fit(X_train, y_train)
                preds_s, tgts_s, msks_s = predict_cross_sectional(m, data, test_idx, feat_indices)
                for t in test_idx:
                    seed_pred_lists[t].append(preds_s[t])
                if tgts_t is None:
                    tgts_t, msks_t = tgts_s, msks_s

            preds_t = {t: np.mean(seed_pred_lists[t], axis=0) for t in test_idx}
            test_sr = compute_sharpe(preds_t, tgts_t, msks_t)
            print(f"    Test SR (ensemble of {tree_seeds}): {test_sr:+.3f}")

            all_results[feat_name] = {
                "preds": preds_t, "tgts": tgts_t, "msks": msks_t,
                "val_sr": val_sr, "test_sr": test_sr,
            }

        _save_results(model_name, all_results, data, dates, assets,
                      test_idx, cfg["output_dir"])

if __name__ == "__main__":
    t0 = time.time()
    retrain_trees()
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
