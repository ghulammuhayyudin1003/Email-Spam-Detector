"""
evaluator.py
─────────────
Evaluation utilities: metrics, classification reports, confusion matrices,
and variance analysis aligned with your paper's Table 3 (std deviation).

Key insight from your paper (Section — Analysis of Model Robustness):
  Raw accuracy is insufficient. A model with 97% accuracy but std=0.01
  across 5 folds is LESS trustworthy than one with 96.5% and std=0.003.
  This module computes BOTH peak performance and variance.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# ── ANSI colour codes for terminal output ─────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def print_section(title: str) -> None:
    """Print a prominent section header to stdout."""
    width = 65
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def evaluate_on_test_set(
    model: Any,
    X_test,
    y_test,
    model_name: str = "Model",
    pos_label: str = "spam",
) -> dict:
    """
    Evaluate a fitted model on the held-out test set.

    Returns dict with: accuracy, precision, recall, f1, report, cm
    """
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec    = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1     = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)

    print_section(f"Test-Set Results — {model_name}")

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'─'*30}")

    for label, value in [
        ("Accuracy",       acc),
        ("Precision (spam)", prec),
        ("Recall (spam)",    rec),
        ("F1-Score (spam)",  f1),
    ]:
        color = GREEN if value >= 0.95 else YELLOW
        print(f"  {label:<20} {color}{value:>10.4f}{RESET}")

    print(f"\n{BOLD}  Full Classification Report:{RESET}")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n{BOLD}  Confusion Matrix:{RESET}")
    print(f"    (rows = actual class, cols = predicted class)\n")
    labels = sorted(set(y_test))
    header = "              " + "  ".join(f"{lbl:>8}" for lbl in labels)
    print(f"    {header}")
    for i, row_label in enumerate(labels):
        row_vals = "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        print(f"    {row_label:>12}  {row_vals}")

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "report":    report,
        "cm":        cm,
    }


def cross_validate_model(
    model: Any,
    X,
    y,
    model_name: str = "Model",
    n_splits: int = 5,
    scoring: str = "accuracy",
) -> dict:
    """
    Run stratified k-fold CV and report mean ± std.

    Mirrors Table 3 in your paper (Accuracy + Standard Deviation per model).
    Stratified k-fold preserves the spam/ham class ratio in every fold.
    """
    print_section(f"Cross-Validation — {model_name} ({n_splits}-Fold)")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    mean_score = scores.mean()
    std_score  = scores.std()

    print(f"\n  Fold scores: {[f'{s:.4f}' for s in scores]}")
    print(f"\n  {'Mean ' + scoring:<20} {mean_score:.4f}")

    std_color = GREEN if std_score <= 0.006 else (YELLOW if std_score <= 0.012 else RED)
    print(f"  {'Std deviation':<20} {std_color}{std_score:.4f}{RESET}", end="")

    if std_score <= 0.006:
        print(f"  {GREEN}← Highly stable (matches RF in your paper){RESET}")
    elif std_score <= 0.012:
        print(f"  {YELLOW}← Acceptable variance{RESET}")
    else:
        print(f"  {RED}← High variance — risk of overfitting{RESET}")

    return {"scores": scores, "mean": mean_score, "std": std_score}


def compare_models(results: dict) -> None:
    """Print a side-by-side comparison table for multiple models."""
    print_section("Model Comparison Summary")

    header = f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print(f"  {'─'*65}")

    sorted_items = sorted(
        results.items(), key=lambda x: x[1].get("f1", 0), reverse=True
    )

    for i, (name, metrics) in enumerate(sorted_items):
        prefix = f"{GREEN}★{RESET} " if i == 0 else "  "
        print(
            f"{prefix}{name:<23}"
            f" {metrics.get('accuracy', 0):>10.4f}"
            f" {metrics.get('precision', 0):>10.4f}"
            f" {metrics.get('recall', 0):>10.4f}"
            f" {metrics.get('f1', 0):>10.4f}"
        )

    best_name = sorted_items[0][0]
    print(f"\n  {GREEN}{BOLD}Best model by F1: {best_name}{RESET}")


def print_overfitting_check(
    train_accuracy: float,
    test_accuracy: float,
    model_name: str = "Model",
    threshold: float = 0.03,
) -> None:
    """
    Print a train vs. test accuracy gap analysis.

    A gap > 3% is the typical overfitting warning threshold.
    """
    gap = train_accuracy - test_accuracy
    gap_color = RED if gap > threshold else GREEN

    print(f"\n  {BOLD}Overfitting Check — {model_name}{RESET}")
    print(f"    Train Accuracy : {train_accuracy:.4f}")
    print(f"    Test  Accuracy : {test_accuracy:.4f}")
    print(f"    Gap            : {gap_color}{gap:+.4f}{RESET}", end="")

    if gap > threshold:
        print(f"  {RED}⚠  Gap > {threshold:.0%} — consider more regularisation.{RESET}")
    else:
        print(f"  {GREEN}✓  Low gap — model generalises well.{RESET}")
