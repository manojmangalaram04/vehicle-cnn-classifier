"""Evaluate the saved model on the independent TEST split only.

Usage:  python evaluate.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from utils.helpers import (IMG_SIZE, read_json, MODEL_PATH, OUTPUTS_DIR, TEST_DIR, ensure_dirs,
                           load_class_names, pretty, write_json)


def plot_history(hist: pd.DataFrame):
    for metric, fname, title in (("accuracy", "accuracy_graph.png", "Training vs Validation Accuracy"),
                                 ("loss", "loss_graph.png", "Training vs Validation Loss")):
        plt.figure(figsize=(7, 4.5))
        plt.plot(hist["epoch"] + 1, hist[metric], label=f"Training {metric}", lw=2)
        plt.plot(hist["epoch"] + 1, hist[f"val_{metric}"], label=f"Validation {metric}", lw=2)
        best = int(hist["val_loss"].idxmin())
        plt.axvline(best + 1, color="gray", ls="--", lw=1, label=f"Best epoch ({best + 1})")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / fname, dpi=150)
        plt.close()


def main():
    ensure_dirs()
    class_names = load_class_names()
    model = keras.models.load_model(MODEL_PATH)
    assert model.output_shape[-1] == len(class_names), "model outputs do not match class list"
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="categorical", class_names=class_names,
        color_mode="rgb", image_size=(IMG_SIZE, IMG_SIZE), batch_size=32, shuffle=False)

    probs = model.predict(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy().argmax(axis=1) for _, y in test_ds])
    y_pred = probs.argmax(axis=1)
    acc = float((y_true == y_pred).mean())
    # plain (un-smoothed) cross-entropy on the test set
    loss = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1))))

    # accuracy per data source (e.g. "existing photos" vs "kaggle"), parsed from file names
    by_source = {}
    src_labels = (read_json(OUTPUTS_DIR / "dataset_report.json") or {}).get("source_labels", [])
    src_idx = np.array([int(m.group(1)) if (m := re.search(r"_s(\d+)_", os.path.basename(p))) else -1
                        for p in test_ds.file_paths])
    for i, name in enumerate(src_labels):
        mask = src_idx == i
        if mask.sum():
            by_source[name] = {"n": int(mask.sum()), "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
                               "per_class_correct": {class_names[k]: f"{int(((y_true == k) & (y_pred == k) & mask).sum())}/{int(((y_true == k) & mask).sum())}"
                                                     for k in range(len(class_names))}}

    labels = [pretty(c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    rep_txt = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True, zero_division=0)

    plt.figure(figsize=(6.2, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    hist_path = OUTPUTS_DIR / "history.csv"
    if hist_path.exists():
        plot_history(pd.read_csv(hist_path))

    (OUTPUTS_DIR / "classification_report.txt").write_text(rep_txt, encoding="utf-8")
    write_json(OUTPUTS_DIR / "classification_report.json", rep)
    write_json(OUTPUTS_DIR / "metrics.json", {
        "test_accuracy": acc, "test_loss": loss, "num_test_images": int(len(y_true)),
        "weighted_precision": rep["weighted avg"]["precision"],
        "weighted_recall": rep["weighted avg"]["recall"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "macro_precision": rep["macro avg"]["precision"],
        "macro_recall": rep["macro avg"]["recall"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "accuracy_by_source": by_source,
        "confusion_matrix": cm.tolist(), "class_names": class_names})

    print(f"Test accuracy: {acc:.4f}   Test loss: {loss:.4f}   ({len(y_true)} images)")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    for name, d in by_source.items():
        print(f"  source '{name}': n={d['n']} accuracy={d['accuracy']:.4f}  {d['per_class_correct']}")
    print("\n" + rep_txt)


if __name__ == "__main__":
    main()
