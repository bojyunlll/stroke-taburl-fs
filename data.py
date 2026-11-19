"""
Data loading and preprocessing for the GSE69138 ischemic-stroke methylation cohort (n = 400).

Expected input: a CSV with probes as rows and samples as columns.
  * first column  ``TargetID``: CpG probe ID (e.g. cg00000029)
  * one row with TargetID ``infect``: subtype label (1 = SAD, 2 = LAA, 3 = CE)
  * remaining rows: beta values in [0, 1]
  * sample columns are named ``<SentrixBarcode>_<Position>[.suffix]``
    (e.g. ``9403904025_R01C01.AVG_Beta``); the barcode identifies the BeadChip.

Preprocessing (label-free, so it does not leak label information):
  1. drop probes with any missing value
  2. clip beta to [0.001, 0.999] and convert to M-values, M = log2(beta / (1 - beta))
  3. center every probe within each BeadChip (subtract the per-chip mean) to remove chip effects
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CLASS_NAMES = ["SAD", "LAA", "CE"]  # encoded labels 0 / 1 / 2


def m_value_chip_centered(beta, chips):
    """beta (samples x probes) -> M-values centered per BeadChip."""
    beta = np.clip(beta, 1e-3, 1 - 1e-3)
    x = np.log2(beta / (1 - beta)).astype(np.float32)
    for chip in np.unique(chips):
        idx = np.asarray(chips == chip)
        x[idx] -= x[idx].mean(axis=0)
    return x


def load_data(csv_path, cache_dir=None):
    """
    Returns (X, y, probes, samples):
      X       float32 array, samples x probes (preprocessed M-values)
      y       int64 labels 0/1/2 (SAD/LAA/CE)
      probes  CpG probe IDs (column names of X)
      samples sample IDs (row names of X)
    The preprocessed arrays are cached as .npy files in ``cache_dir`` (default: <csv dir>/_cache).
    """
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(csv_path)), "_cache")
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    paths = {n: os.path.join(cache_dir, f"{stem}_{n}.npy") for n in ["X", "y", "probes", "samples"]}
    if all(os.path.exists(p) for p in paths.values()):
        return (np.load(paths["X"]), np.load(paths["y"]),
                np.load(paths["probes"], allow_pickle=True), np.load(paths["samples"], allow_pickle=True))

    t0 = time.time()
    df = pd.read_csv(csv_path, engine="pyarrow").set_index("TargetID").T
    labels = df["infect"].astype(int)
    beta = df.drop(columns=["infect"]).astype(np.float32).dropna(axis=1)
    del df
    chips = np.asarray(beta.index.str.split("_").str[0])
    X = m_value_chip_centered(beta.to_numpy(), chips)
    y = LabelEncoder().fit_transform(labels).astype(np.int64)
    probes, samples = np.array(beta.columns), np.array(beta.index)

    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in [("X", X), ("y", y), ("probes", probes), ("samples", samples)]:
        np.save(paths[name], arr)
    print(f"Preprocessed data cached ({time.time() - t0:.0f}s): X = {X.shape}, "
          f"class counts = {np.bincount(y).tolist()}", flush=True)
    return X, y, probes, samples
