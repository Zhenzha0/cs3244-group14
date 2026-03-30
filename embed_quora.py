"""
embed_quora.py — Generate Qwen3-Embedding-4B embeddings for all unique questions.

Produces an embeddings.zarr store with:
  - ids:        (N,) int64   — sorted question IDs
  - texts:      (N,) str     — question strings
  - embeddings: (N, 2560) float32

This is the same embedding approach as Alan's repo, so the resulting
embeddings are interchangeable for fair model comparison.

Usage:
    python embed_quora.py                     # uses data/train.csv
    python embed_quora.py --csv data/train.csv --batch-size 64
"""

import argparse
import csv
import os
import time

import numpy as np
import zarr
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
OUTPUT_FILE = "embeddings.zarr"
BATCH_SIZE = 128


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Quora questions")
    parser.add_argument("--csv", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output zarr path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Collect unique questions by qid
    print("[INFO] Loading dataset...", flush=True)
    id_to_text: dict[int, str] = {}
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for id_col, text_col in [("qid1", "question1"), ("qid2", "question2")]:
                try:
                    qid = int(row[id_col])
                    text = row[text_col]
                    if qid not in id_to_text:
                        id_to_text[qid] = text
                except (KeyError, ValueError):
                    pass

    sorted_ids = sorted(id_to_text.keys())
    sorted_texts = [id_to_text[qid] for qid in sorted_ids]
    N = len(sorted_ids)
    print(f"[INFO] Unique questions: {N}", flush=True)

    # Load model
    print(f"[INFO] Loading model: {MODEL_NAME}", flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {dim}", flush=True)

    # Create zarr store
    store = zarr.open(args.output, mode="w")
    ids_arr = store.zeros(name="ids", shape=(N,), dtype="int64", chunks=(args.batch_size,))
    ids_arr[:] = np.array(sorted_ids, dtype=np.int64)

    texts_arr = store.create_array(name="texts", shape=(N,), dtype="str", chunks=(args.batch_size,))
    texts_arr[:] = sorted_texts

    emb_arr = store.zeros(name="embeddings", shape=(N, dim), dtype="float32", chunks=(args.batch_size, dim))

    # Encode in batches
    print(f"[INFO] Embedding {N} questions in batches of {args.batch_size}...", flush=True)
    t_start = time.time()
    last_log = t_start

    for i in range(0, N, args.batch_size):
        batch_texts = sorted_texts[i : i + args.batch_size]
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, prompt_name="query")
        emb_arr[i : i + len(batch_texts)] = embs

        now = time.time()
        done = i + len(batch_texts)
        elapsed = now - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (N - done) / rate if rate > 0 else 0

        if (now - last_log) >= 30 or i == 0:
            print(
                f"[PROGRESS] {done}/{N} ({done/N*100:.1f}%) | "
                f"Elapsed: {format_duration(elapsed)} | "
                f"ETA: {format_duration(eta)} | "
                f"Speed: {rate:.1f} q/s",
                flush=True,
            )
            last_log = now

    total = time.time() - t_start
    print(f"[DONE] Embedding complete in {format_duration(total)} ({N/total:.1f} q/s)", flush=True)
    print(f"Saved to {args.output}: ids={store['ids'].shape}, embeddings={store['embeddings'].shape}")


if __name__ == "__main__":
    main()
