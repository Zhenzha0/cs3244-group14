"""
predict.py — Generate Kaggle submission from the trained Siamese GRU model.

Embeds test questions with Qwen3-Embedding-4B, runs them through the trained
model, and outputs a CSV in Kaggle format (test_id, is_duplicate).

Usage:
    python rnn/predict.py
    python rnn/predict.py --test-csv data/test.csv --model-path rnn/results/best_model.pt
"""

import argparse
import csv
import os

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

    # Collect unique questions to avoid re-embedding duplicates
    all_questions = list(set(questions1 + questions2))
    print(f"[predict] Unique questions to embed: {len(all_questions)}", flush=True)

    # Embed
    print(f"[predict] Loading embedding model: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(
        args.embedding_model,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    dim = embedder.get_sentence_embedding_dimension()
    print(f"[predict] Embedding dimension: {dim}", flush=True)

    print(f"[predict] Embedding {len(all_questions)} unique questions...", flush=True)
    all_embeddings = embedder.encode(
        all_questions,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        prompt_name="query",
    )

    # Map question text -> embedding
    q_to_emb = {q: emb for q, emb in zip(all_questions, all_embeddings)}

    # Build embedding arrays for pairs
    print("[predict] Building pair embeddings...", flush=True)
    emb1 = np.array([q_to_emb[q] for q in questions1], dtype=np.float32)
    emb2 = np.array([q_to_emb[q] for q in questions2], dtype=np.float32)

    # Load trained model
    print(f"[predict] Loading model from {model_path}", flush=True)
    model = SiameseGRU(embedding_dim=dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Predict in batches
    print("[predict] Running predictions...", flush=True)
    all_proba = []
    with torch.no_grad():
        for i in range(0, len(test_ids), args.batch_size):
            e1 = torch.from_numpy(emb1[i:i + args.batch_size]).to(device)
            e2 = torch.from_numpy(emb2[i:i + args.batch_size]).to(device)
            logits = model(e1, e2)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            all_proba.append(proba)

    all_proba = np.concatenate(all_proba)

    # Write submission CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "is_duplicate"])
        for tid, prob in zip(test_ids, all_proba):
            writer.writerow([tid, f"{prob:.6f}"])

    print(f"[predict] Submission saved to {output_path}", flush=True)
    print(f"[predict] Total predictions: {len(all_proba)}", flush=True)
    print(f"[predict] Predicted duplicate rate: {(all_proba >= args.threshold).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
