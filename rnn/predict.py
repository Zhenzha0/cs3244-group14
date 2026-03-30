"""
predict.py — Generate Kaggle submission from the trained Siamese GRU model.

Embeds test questions with Qwen3-Embedding-4B, runs them through the trained
model, and outputs a CSV in Kaggle format (test_id, is_duplicate).

Memory-efficient: stores embeddings on disk via numpy memmap to avoid OOM.

Usage:
    python rnn/predict.py
    python rnn/predict.py --test-csv data/test.csv --model-path rnn/results/best_model.pt
"""

import argparse
import csv
import os
import tempfile

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from model import SiameseGRU


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Kaggle submission")
    p.add_argument("--test-csv", default=None, help="Path to test.csv")
    p.add_argument("--model-path", default=None, help="Path to best_model.pt")
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")

    test_csv = args.test_csv or os.path.join(repo_root, "data", "test.csv")
    model_path = args.model_path or os.path.join(script_dir, "results", "best_model.pt")
    output_path = args.output or os.path.join(script_dir, "results", "submission.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Device: {device}", flush=True)

    # Load test data
    print(f"[predict] Reading {test_csv}...", flush=True)
    test_ids = []
    questions1 = []
    questions2 = []
    with open(test_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(int(row["test_id"]))
            questions1.append(row["question1"] or "")
            questions2.append(row["question2"] or "")
    print(f"[predict] Loaded {len(test_ids)} test pairs", flush=True)

    # Collect unique questions and assign indices
    unique_questions = list(set(questions1 + questions2))
    q_to_idx = {q: i for i, q in enumerate(unique_questions)}
    N_unique = len(unique_questions)
    print(f"[predict] Unique questions to embed: {N_unique}", flush=True)

    # Load embedding model
    print(f"[predict] Loading embedding model: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(
        args.embedding_model,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    dim = embedder.get_sentence_embedding_dimension()
    print(f"[predict] Embedding dimension: {dim}", flush=True)

    # Embed into a memory-mapped file to avoid OOM
    tmpdir = tempfile.mkdtemp()
    emb_path = os.path.join(tmpdir, "embeddings.dat")
    emb_mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(N_unique, dim))

    print(f"[predict] Embedding {N_unique} unique questions (memmap)...", flush=True)
    embed_batch = 256
    for i in range(0, N_unique, embed_batch):
        batch = unique_questions[i:i + embed_batch]
        vecs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False, prompt_name="query")
        emb_mmap[i:i + len(batch)] = vecs

        if i % (embed_batch * 100) == 0 or i + embed_batch >= N_unique:
            pct = min((i + len(batch)) / N_unique * 100, 100)
            print(f"[predict] Embedded {i + len(batch)}/{N_unique} ({pct:.1f}%)", flush=True)

    emb_mmap.flush()
    print("[predict] Embedding complete.", flush=True)

    # Free the embedding model from GPU memory
    del embedder
    torch.cuda.empty_cache()

    # Build index arrays for pairs (int32, much smaller than full embeddings)
    idx1 = np.array([q_to_idx[q] for q in questions1], dtype=np.int32)
    idx2 = np.array([q_to_idx[q] for q in questions2], dtype=np.int32)

    # Free question lists
    del questions1, questions2, unique_questions, q_to_idx

    # Load trained model
    print(f"[predict] Loading model from {model_path}", flush=True)
    model = SiameseGRU(embedding_dim=dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Predict in batches, reading embeddings from memmap
    print("[predict] Running predictions...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred_batch = 1024

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "is_duplicate"])

        with torch.no_grad():
            for i in range(0, len(test_ids), pred_batch):
                batch_idx1 = idx1[i:i + pred_batch]
                batch_idx2 = idx2[i:i + pred_batch]

                e1 = torch.from_numpy(np.array(emb_mmap[batch_idx1])).to(device)
                e2 = torch.from_numpy(np.array(emb_mmap[batch_idx2])).to(device)

                logits = model(e1, e2)
                proba = torch.sigmoid(logits).cpu().numpy().flatten()

                for tid, prob in zip(test_ids[i:i + pred_batch], proba):
                    writer.writerow([tid, f"{prob:.6f}"])

                if i % (pred_batch * 100) == 0:
                    print(f"[predict] Predicted {i + len(batch_idx1)}/{len(test_ids)}", flush=True)

    # Cleanup
    del emb_mmap
    os.remove(emb_path)
    os.rmdir(tmpdir)

    print(f"[predict] Submission saved to {output_path}", flush=True)
    print(f"[predict] Total predictions: {len(test_ids)}", flush=True)


if __name__ == "__main__":
    main()
