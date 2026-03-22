"""
data_loader.py — Load btc_panel.npz and create temporal sequences for LSTM/TFT.

Panel format: (T, N, M+1) where col 0 = next-week return, cols 1..M = features.
UNK sentinel: -99.99 (missing data marker).

v5: Added CrossSectionalDataset for ranking-based training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


UNK = -99.99


class CryptoSequenceDataset(Dataset):
    """Creates (lookback, n_features) sequences from panel data."""

    def __init__(self, data, time_indices, feature_indices, lookback=12):
        self.samples = []
        feat_cols = [f + 1 for f in feature_indices]

        for t in time_indices:
            if t < lookback:
                continue
            for n in range(data.shape[1]):
                target = data[t, n, 0]
                if target <= UNK + 1:
                    continue
                seq = data[t - lookback:t, n][:, feat_cols].copy()
                seq[seq <= UNK + 1] = 0.0
                self.samples.append((seq.astype(np.float32), np.float32(target), n))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target, asset_idx = self.samples[idx]
        return (torch.from_numpy(seq),
                torch.tensor(target),
                torch.tensor(asset_idx, dtype=torch.long))


class CrossSectionalDataset(Dataset):
    """
    Each item = all assets at one time step.

    Used for ranking-based training where the loss needs to compare
    all assets simultaneously at the same point in time.
    """

    def __init__(self, data, time_indices, feature_indices, lookback=8):
        self.batches = []
        feat_cols = [f + 1 for f in feature_indices]
        N = data.shape[1]

        for t in time_indices:
            if t < lookback:
                continue

            seqs = np.zeros((N, lookback, len(feature_indices)), dtype=np.float32)
            targets = np.zeros(N, dtype=np.float32)
            masks = np.zeros(N, dtype=bool)

            for n in range(N):
                seq = data[t - lookback:t, n][:, feat_cols].copy()
                seq[seq <= UNK + 1] = 0.0
                seqs[n] = seq

                ret = data[t, n, 0]
                valid = ret > UNK + 1
                targets[n] = float(ret) if valid else 0.0
                masks[n] = valid

            if masks.sum() >= 10:
                self.batches.append((seqs, targets, masks,
                                     np.arange(N, dtype=np.int64)))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        seqs, targets, masks, asset_idx = self.batches[idx]
        return (torch.from_numpy(seqs),
                torch.from_numpy(targets),
                torch.from_numpy(masks),
                torch.from_numpy(asset_idx))


def get_cross_sectional_data(data, time_indices, feature_indices, lookback=12):
    """Organise data by time step for cross-sectional evaluation."""
    feat_cols = [f + 1 for f in feature_indices]
    N = data.shape[1]
    results = {}

    for t in time_indices:
        if t < lookback:
            continue
        seqs, targets, masks = [], [], []
        for n in range(N):
            seq = data[t - lookback:t, n][:, feat_cols].copy()
            seq[seq <= UNK + 1] = 0.0
            seqs.append(seq.astype(np.float32))

            ret = data[t, n, 0]
            valid = ret > UNK + 1
            targets.append(float(ret) if valid else 0.0)
            masks.append(valid)

        results[t] = {
            'sequences': torch.from_numpy(np.stack(seqs)),
            'targets': np.array(targets, dtype=np.float32),
            'masks': np.array(masks, dtype=bool),
            'asset_indices': torch.arange(N, dtype=torch.long),
        }
    return results


def load_panel(npz_path):
    """Load the panel NPZ and return (data, dates, assets, variables)."""
    f = np.load(npz_path, allow_pickle=True)
    return f['data'], f['date'], f['wficn'], f['variable']


def detect_trump_start(data, trump_feature_indices, unk=UNK):
    """Find the first week where Trump features have valid (non-UNK) data."""
    feat_cols = [f + 1 for f in trump_feature_indices]  # +1 because col 0 = target
    for t in range(data.shape[0]):
        for n in range(data.shape[1]):
            vals = data[t, n, feat_cols]
            if np.all(vals > unk + 1):
                return t
    return 0


def make_splits(T, train_ratio=0.70, valid_ratio=0.15, train_start=0):
    """Chronological train / valid / test split."""
    t_train = int(T * train_ratio)
    t_valid = int(T * (train_ratio + valid_ratio))
    return range(train_start, t_train), range(t_train, t_valid), range(t_valid, T)
