"""
data.py — Data loader for RNN training.

Loads the Zarr embedding store (produced by embed_quora.py) and the
Quora question-pairs CSV, returning parallel arrays of embedding pairs
and labels.
"""

import csv
import os
from typing import NamedTuple

import numpy as np
import zarr

ZARR_FILE = os.path.join(os.path.dirname(__file__), "..", "embeddings.zarr")
CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")


class PairData(NamedTuple):
    """Packed arrays for all usable question pairs."""
    emb1: np.ndarray      # (N, dim) float32
    emb2: np.ndarray      # (N, dim) float32
    labels: np.ndarray    # (N,) int64


def load_pairs(
    zarr_file: str = ZARR_FILE,
    csv_file: str = CSV_FILE,
    max_rows: int | None = None,
) -> PairData:
    """Load embeddings + CSV and return matched embedding pairs with labels."""

    print(f"[data] Loading zarr store: {zarr_file}", flush=True)
    store = zarr.open(zarr_file, mode="r")
    ids_arr = store["ids"][:].astype(np.int64)
    emb_arr = store["embeddings"][:].astype(np.float32)
    print(f"[data] embeddings shape: {emb_arr.shape}", flush=True)

    # Map question ID -> position in zarr arrays
    qid_to_pos = {int(qid): i for i, qid in enumerate(ids_arr)}

    print(f"[data] Reading CSV: {csv_file}", flush=True)
    emb1_list, emb2_list, label_list = [], [], []
    skipped = 0

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows is not None and len(label_list) >= max_rows:
                break
            try:
                qid1 = int(row["qid1"])
                qid2 = int(row["qid2"])
                label = int(row["is_duplicate"])
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue

            pos1 = qid_to_pos.get(qid1)
            pos2 = qid_to_pos.get(qid2)
            if pos1 is None or pos2 is None:
                skipped += 1
                continue

            emb1_list.append(pos1)
            emb2_list.append(pos2)
            label_list.append(label)

    # Gather embeddings by index (avoids copying one-by-one)
    idx1 = np.array(emb1_list, dtype=np.int64)
    idx2 = np.array(emb2_list, dtype=np.int64)

    print(f"[data] Loaded {len(label_list)} pairs, skipped {skipped}", flush=True)

    return PairData(
        emb1=emb_arr[idx1],
        emb2=emb_arr[idx2],
        labels=np.array(label_list, dtype=np.int64),
    )
