"""
train.py — Training script for the Siamese GRU model.

Uses the same train/test split logic as Alan's experiment framework
(stratified, test_size=0.20, random_state=42) for fair comparison.

Usage:
    python rnn/train.py                          # full dataset
    python rnn/train.py --max-rows 50000         # smoke test
    python rnn/train.py --epochs 20 --lr 0.0005  # custom hyperparams
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from data import load_pairs
from model import SiameseGRU


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese GRU on Quora Question Pairs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-rows", type=int, default=None, help="Subsample for smoke tests")
    p.add_argument("--zarr", default=None, help="Path to embeddings.zarr")
    p.add_argument("--csv", default=None, help="Path to train.csv")
    p.add_argument("--results-dir", default=None, help="Where to save results")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for emb1, emb2, labels in loader:
        emb1 = emb1.to(device)
        emb2 = emb2.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(emb1, emb2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    all_proba, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for emb1, emb2, labels in loader:
        emb1 = emb1.to(device)
        emb2 = emb2.to(device)
        labels_dev = labels.to(device).float().unsqueeze(1)

        logits = model(emb1, emb2)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item()
        n_batches += 1

        proba = torch.sigmoid(logits).cpu().numpy().flatten()
        all_proba.append(proba)
        all_labels.append(labels.numpy())

    all_proba = np.concatenate(all_proba)
    all_labels = np.concatenate(all_labels)
    preds = (all_proba >= threshold).astype(int)

    return {
        "loss": total_loss / n_batches,
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds),
        "precision": precision_score(all_labels, preds),
        "recall": recall_score(all_labels, preds),
        "proba": all_proba,
        "labels": all_labels,
        "preds": preds,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}", flush=True)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(script_dir, "..")
    zarr_path = args.zarr or os.path.join(repo_root, "embeddings.zarr")
    csv_path = args.csv or os.path.join(repo_root, "data", "train.csv")
    results_dir = args.results_dir or os.path.join(script_dir, "results")

    # Load data
    pair_data = load_pairs(zarr_file=zarr_path, csv_file=csv_path, max_rows=args.max_rows)
    emb1, emb2, labels = pair_data.emb1, pair_data.emb2, pair_data.labels
    embedding_dim = emb1.shape[1]
    print(f"[train] Embedding dim: {embedding_dim}", flush=True)
    print(f"[train] Total pairs: {len(labels)}", flush=True)

    # Same split as Alan's: stratified, test_size=0.20, random_state=42
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    print(f"[train] Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(emb1[train_idx]),
        torch.from_numpy(emb2[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    test_ds = TensorDataset(
        torch.from_numpy(emb1[test_idx]),
        torch.from_numpy(emb2[test_idx]),
        torch.from_numpy(labels[test_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = SiameseGRU(
        embedding_dim=embedding_dim,
        chunk_size=args.chunk_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {total_params:,}", flush=True)
    print(model, flush=True)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_f1 = 0.0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, test_loader, criterion, device, args.threshold)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t_epoch
        print(
            f"[Epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"f1={val_metrics['f1']:.4f}  "
            f"prec={val_metrics['precision']:.4f}  "
            f"rec={val_metrics['recall']:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            os.makedirs(results_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))

    total_time = time.time() - t_start
    print(f"\n[train] Training complete in {total_time:.1f}s", flush=True)
    print(f"[train] Best F1: {best_f1:.4f} at epoch {best_epoch}", flush=True)

    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pt"), weights_only=True))
    final = evaluate(model, test_loader, criterion, device, args.threshold)

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    # Classification report
    report = classification_report(final["labels"], final["preds"], digits=4)
    cm = confusion_matrix(final["labels"], final["preds"])
    print(f"\n{report}", flush=True)
    print(f"Confusion matrix:\n{cm}", flush=True)

    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(f"Model: Siamese GRU\n")
        f.write(f"Best epoch: {best_epoch}/{args.epochs}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        f.write(f"Accuracy:  {final['accuracy']:.4f}\n")
        f.write(f"F1:        {final['f1']:.4f}\n")
        f.write(f"Precision: {final['precision']:.4f}\n")
        f.write(f"Recall:    {final['recall']:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion matrix:\n{cm}\n")
        f.write(f"\nTotal training time: {total_time:.1f}s\n")

    # Config for reproducibility
    config = {
        "model": "SiameseGRU",
        "embedding_dim": embedding_dim,
        "chunk_size": args.chunk_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "threshold": args.threshold,
        "test_size": args.test_size,
        "seed": args.seed,
        "max_rows": args.max_rows,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "accuracy": final["accuracy"],
        "f1": final["f1"],
        "precision": final["precision"],
        "recall": final["recall"],
        "training_time_s": total_time,
        "device": str(device),
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[train] Results saved to {results_dir}/", flush=True)


if __name__ == "__main__":
    main()
