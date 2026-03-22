"""# ============================================================
# EVALUATE.PY — Model Evaluation
# ============================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from config import LABEL_MAP


def evaluate_model(y_true, y_pred, model_name="Model"):


    #Print comprehensive evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'─' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─' * 50}")
    print(f"  Accuracy:  {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Real  Predicted Fake")
    print(f"  Actual Real     {cm[0][0]:>13}  {cm[0][1]:>14}")
    print(f"  Actual Fake     {cm[1][0]:>13}  {cm[1][1]:>14}")
    print(f"\n  Classification Report:")
    target_names = [LABEL_MAP[0], LABEL_MAP[1]]
    report = classification_report(y_true, y_pred, target_names=target_names)
    for line in report.split("\n"):
        print(f"  {line}")

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1_score": f1,
        "confusion_matrix": cm,
    }
"""

# ============================================================
# EVALUATE.PY — Model Evaluation
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from config import LABEL_MAP

# ── Output directory for saved curve images ──────────────────
CURVES_DIR = "eval_curves"
os.makedirs(CURVES_DIR, exist_ok=True)


# ── Internal helpers ─────────────────────────────────────────

def _apply_dark_style(fig, axes):
    """Apply a clean dark theme to a figure and its axes."""
    BG      = "#0f1117"
    PANEL   = "#1a1d27"
    GRID    = "#2a2d3a"
    TEXT    = "#e0e4f0"
    SUBTEXT = "#8890aa"

    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.8)


def _save(fig, path):
    """Save figure with tight layout and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Saved → {path}")


# ── ROC Curve ────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, model_name="Model", save_dir=CURVES_DIR):
    """
    Plot and save the ROC curve.

    Parameters
    ----------
    y_true  : array-like of int  — Ground-truth binary labels (0 / 1).
    y_prob  : array-like of float — Predicted probabilities for the positive
              class (label 1).  Shape: (n_samples,).
    model_name : str  — Used in the plot title and filename.
    save_dir   : str  — Directory where the PNG is written.

    Returns
    -------
    roc_auc : float
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    ACCENT  = "#4f8ef7"
    DIAG    = "#55607a"

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(fig, ax)

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color=DIAG, linewidth=1.2, linestyle="--",
            label="Random (AUC = 0.50)")

    # Shaded area under the curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)

    # ROC curve
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.2,
            label=f"{model_name}  (AUC = {roc_auc:.4f})")

    # Annotations
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    legend = ax.legend(loc="lower right", fontsize=9, framealpha=0.3,
                       facecolor="#1a1d27", edgecolor="#2a2d3a")
    for text in legend.get_texts():
        text.set_color("#e0e4f0")

    path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_roc_curve.png")
    _save(fig, path)
    return roc_auc


# ── Precision-Recall Curve ───────────────────────────────────

def plot_pr_curve(y_true, y_prob, model_name="Model", save_dir=CURVES_DIR):
    """
    Plot and save the Precision-Recall curve.

    Parameters
    ----------
    y_true  : array-like of int   — Ground-truth binary labels (0 / 1).
    y_prob  : array-like of float — Predicted probabilities for the positive
              class (label 1).  Shape: (n_samples,).
    model_name : str  — Used in the plot title and filename.
    save_dir   : str  — Directory where the PNG is written.

    Returns
    -------
    ap : float  — Average Precision score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    ACCENT   = "#f4845f"
    BASELINE = "#55607a"
    baseline = np.sum(y_true) / len(y_true)   # positive-class prevalence

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(fig, ax)

    # Baseline (no-skill classifier)
    ax.axhline(y=baseline, color=BASELINE, linewidth=1.2, linestyle="--",
               label=f"No-Skill  (AP = {baseline:.2f})")

    # Shaded area
    ax.fill_between(recall, precision, alpha=0.15, color=ACCENT)

    # PR curve (note: sklearn returns these in reverse threshold order)
    ax.plot(recall, precision, color=ACCENT, linewidth=2.2,
            label=f"{model_name}  (AP = {ap:.4f})")

    ax.set_xlabel("Recall",    fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.3,
                       facecolor="#1a1d27", edgecolor="#2a2d3a")
    for text in legend.get_texts():
        text.set_color("#e0e4f0")

    path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_pr_curve.png")
    _save(fig, path)
    return ap


# ── Combined dashboard (both curves side-by-side) ────────────

def plot_curves_dashboard(y_true, y_prob, model_name="Model", save_dir=CURVES_DIR):
    """
    Save a single PNG that shows the ROC curve and the PR curve
    side by side, plus a small stats panel.

    Returns
    -------
    dict with keys 'roc_auc' and 'average_precision'.
    """
    BG      = "#0f1117"
    PANEL   = "#1a1d27"
    GRID    = "#2a2d3a"
    TEXT    = "#e0e4f0"
    SUBTEXT = "#8890aa"
    BLUE    = "#4f8ef7"
    ORANGE  = "#f4845f"
    DIAG    = "#55607a"

    # ── compute curves ────────────────────────────────────────
    fpr, tpr, _       = roc_curve(y_true, y_prob)
    roc_auc            = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap                 = average_precision_score(y_true, y_prob)
    baseline           = np.sum(y_true) / len(y_true)

    # ── figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.2), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[5, 5, 2.6],
                            wspace=0.38, left=0.06, right=0.97,
                            top=0.88, bottom=0.14)

    ax_roc  = fig.add_subplot(gs[0])
    ax_pr   = fig.add_subplot(gs[1])
    ax_info = fig.add_subplot(gs[2])

    for ax in [ax_roc, ax_pr, ax_info]:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── ROC ───────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], color=DIAG, linewidth=1.2,
                linestyle="--", label="Random  (AUC = 0.50)")
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
    ax_roc.plot(fpr, tpr, color=BLUE, linewidth=2.2,
                label=f"AUC = {roc_auc:.4f}")
    ax_roc.set_xlabel("False Positive Rate", fontsize=9.5, color=TEXT)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=9.5, color=TEXT)
    ax_roc.set_title("ROC Curve", fontsize=11, fontweight="bold",
                     color=TEXT, pad=9)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.05])
    ax_roc.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax_roc.tick_params(colors=SUBTEXT, labelsize=8.5)
    leg = ax_roc.legend(loc="lower right", fontsize=8.5, framealpha=0.3,
                        facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # ── PR ────────────────────────────────────────────────────
    ax_pr.axhline(y=baseline, color=DIAG, linewidth=1.2, linestyle="--",
                  label=f"No-Skill  ({baseline:.2f})")
    ax_pr.fill_between(recall, precision, alpha=0.12, color=ORANGE)
    ax_pr.plot(recall, precision, color=ORANGE, linewidth=2.2,
               label=f"AP = {ap:.4f}")
    ax_pr.set_xlabel("Recall",    fontsize=9.5, color=TEXT)
    ax_pr.set_ylabel("Precision", fontsize=9.5, color=TEXT)
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold",
                    color=TEXT, pad=9)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.05])
    ax_pr.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax_pr.tick_params(colors=SUBTEXT, labelsize=8.5)
    leg2 = ax_pr.legend(loc="upper right", fontsize=8.5, framealpha=0.3,
                         facecolor=PANEL, edgecolor=GRID)
    for t in leg2.get_texts():
        t.set_color(TEXT)

    # ── Stats panel ───────────────────────────────────────────
    ax_info.axis("off")
    stats = [
        ("ROC AUC",    f"{roc_auc:.4f}"),
        ("Avg Prec",   f"{ap:.4f}"),
        ("Prevalence", f"{baseline:.4f}"),
        ("Samples",    f"{len(y_true):,}"),
        ("Positives",  f"{int(np.sum(y_true)):,}"),
    ]
    ax_info.set_title("Stats", fontsize=10, fontweight="bold",
                      color=TEXT, pad=6)
    y_pos = 0.88
    for label, value in stats:
        ax_info.text(0.08, y_pos, label, transform=ax_info.transAxes,
                     fontsize=9, color=SUBTEXT, va="top")
        ax_info.text(0.92, y_pos, value, transform=ax_info.transAxes,
                     fontsize=9.5, color=TEXT, va="top", ha="right",
                     fontweight="bold")
        y_pos -= 0.01
        ax_info.axhline(y=y_pos, xmin=0.04, xmax=0.96,
                        color=GRID, linewidth=0.5,
                        transform=ax_info.get_xaxis_transform())
        y_pos -= 0.15

    # ── Super-title ───────────────────────────────────────────
    fig.suptitle(f"{model_name} — Evaluation Curves",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.97)

    path = os.path.join(save_dir,
                        f"{model_name.replace(' ', '_')}_curves_dashboard.png")
    _save(fig, path)
    return {"roc_auc": roc_auc, "average_precision": ap}


# ── Loss Curve ───────────────────────────────────────────────

def plot_loss_curve(train_losses, val_losses=None, model_name="Model", save_dir=CURVES_DIR):
    """
    Plot and save the training (and optionally validation) loss curve.

    Parameters
    ----------
    train_losses : list of float — Loss value recorded after each epoch.
    val_losses   : list of float or None — Validation loss per epoch.
    model_name   : str  — Used in the plot title and filename.
    save_dir     : str  — Directory where the PNG is written.
    """
    GREEN  = "#3ecf8e"
    YELLOW = "#f9c74f"
    DIAG   = "#55607a"

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_dark_style(fig, ax)

    ax.plot(epochs, train_losses, color=GREEN, linewidth=2.2, marker="o",
            markersize=5, label="Train Loss")

    if val_losses is not None:
        ax.plot(epochs, val_losses, color=YELLOW, linewidth=2.2,
                marker="s", markersize=5, linestyle="--", label="Val Loss")

        # Mark best (lowest) validation loss
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val   = min(val_losses)
        ax.axvline(x=best_epoch, color=DIAG, linewidth=1.0, linestyle=":")
        ax.annotate(f" best\n ep {best_epoch}\n {best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.3, best_val),
                    fontsize=8, color=YELLOW, va="center")

    ax.set_xlabel("Epoch",  fontsize=10)
    ax.set_ylabel("Loss",   fontsize=10)
    ax.set_title("Loss Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs)

    legend = ax.legend(fontsize=9, framealpha=0.3,
                       facecolor="#1a1d27", edgecolor="#2a2d3a")
    for text in legend.get_texts():
        text.set_color("#e0e4f0")

    path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_loss_curve.png")
    _save(fig, path)


# ── Accuracy Curve ───────────────────────────────────────────

def plot_accuracy_curve(train_accs, val_accs=None, model_name="Model", save_dir=CURVES_DIR):
    """
    Plot and save the training (and optionally validation) accuracy curve.

    Parameters
    ----------
    train_accs : list of float — Accuracy value recorded after each epoch (0–1 or 0–100).
    val_accs   : list of float or None — Validation accuracy per epoch.
    model_name : str  — Used in the plot title and filename.
    save_dir   : str  — Directory where the PNG is written.
    """
    PURPLE = "#b185f7"
    CYAN   = "#50d8d7"
    DIAG   = "#55607a"

    # Normalise to 0–1 if values look like percentages
    def _norm(vals):
        arr = np.asarray(vals, dtype=float)
        return arr / 100.0 if arr.max() > 1.0 else arr

    train_accs = _norm(train_accs)
    epochs     = list(range(1, len(train_accs) + 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_dark_style(fig, ax)

    ax.plot(epochs, train_accs, color=PURPLE, linewidth=2.2, marker="o",
            markersize=5, label="Train Accuracy")

    if val_accs is not None:
        val_accs = _norm(val_accs)
        ax.plot(epochs, val_accs, color=CYAN, linewidth=2.2,
                marker="s", markersize=5, linestyle="--", label="Val Accuracy")

        # Mark best validation accuracy
        best_epoch = int(np.argmax(val_accs)) + 1
        best_val   = float(np.max(val_accs))
        ax.axvline(x=best_epoch, color=DIAG, linewidth=1.0, linestyle=":")
        ax.annotate(f" best\n ep {best_epoch}\n {best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.3, best_val - 0.03),
                    fontsize=8, color=CYAN, va="center")

    ax.set_xlabel("Epoch",    fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Accuracy Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs)
    ax.set_ylim([max(0, float(train_accs.min()) - 0.05), 1.05])

    legend = ax.legend(fontsize=9, framealpha=0.3,
                       facecolor="#1a1d27", edgecolor="#2a2d3a")
    for text in legend.get_texts():
        text.set_color("#e0e4f0")

    path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_accuracy_curve.png")
    _save(fig, path)


# ── Training dashboard (loss + accuracy side-by-side) ────────

def plot_training_curves(train_losses, train_accs,
                         val_losses=None, val_accs=None,
                         model_name="Model", save_dir=CURVES_DIR):
    """
    Save a single PNG with loss and accuracy curves side by side.

    Parameters
    ----------
    train_losses : list of float
    train_accs   : list of float  (0–1 or 0–100, auto-normalised)
    val_losses   : list of float or None
    val_accs     : list of float or None
    model_name   : str
    save_dir     : str
    """
    BG      = "#0f1117"
    PANEL   = "#1a1d27"
    GRID    = "#2a2d3a"
    TEXT    = "#e0e4f0"
    SUBTEXT = "#8890aa"
    GREEN   = "#3ecf8e"
    YELLOW  = "#f9c74f"
    PURPLE  = "#b185f7"
    CYAN    = "#50d8d7"
    DIAG    = "#55607a"

    def _norm(vals):
        arr = np.asarray(vals, dtype=float)
        return arr / 100.0 if arr.max() > 1.0 else arr

    train_accs_ = _norm(train_accs)
    val_accs_   = _norm(val_accs) if val_accs is not None else None
    epochs      = list(range(1, len(train_losses) + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5),
                                           facecolor=BG)
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.87, bottom=0.13)

    for ax in [ax_loss, ax_acc]:
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
        ax.tick_params(colors=SUBTEXT, labelsize=8.5)
        ax.set_xticks(epochs)

    # ── Loss ──────────────────────────────────────────────────
    ax_loss.plot(epochs, train_losses, color=GREEN, linewidth=2.2,
                 marker="o", markersize=5, label="Train Loss")
    if val_losses is not None:
        ax_loss.plot(epochs, val_losses, color=YELLOW, linewidth=2.2,
                     marker="s", markersize=5, linestyle="--", label="Val Loss")
        best_e = int(np.argmin(val_losses)) + 1
        ax_loss.axvline(x=best_e, color=DIAG, linewidth=1.0, linestyle=":")
        ax_loss.annotate(f" ep {best_e}",
                         xy=(best_e, min(val_losses)),
                         xytext=(best_e + 0.25, min(val_losses)),
                         fontsize=7.5, color=YELLOW)
    ax_loss.set_xlabel("Epoch", fontsize=9.5, color=TEXT)
    ax_loss.set_ylabel("Loss",  fontsize=9.5, color=TEXT)
    ax_loss.set_title("Loss Curve", fontsize=11, fontweight="bold", color=TEXT, pad=9)
    leg1 = ax_loss.legend(fontsize=8.5, framealpha=0.3,
                           facecolor=PANEL, edgecolor=GRID)
    for t in leg1.get_texts():
        t.set_color(TEXT)

    # ── Accuracy ──────────────────────────────────────────────
    ax_acc.plot(epochs, train_accs_, color=PURPLE, linewidth=2.2,
                marker="o", markersize=5, label="Train Accuracy")
    if val_accs_ is not None:
        ax_acc.plot(epochs, val_accs_, color=CYAN, linewidth=2.2,
                    marker="s", markersize=5, linestyle="--", label="Val Accuracy")
        best_e2 = int(np.argmax(val_accs_)) + 1
        ax_acc.axvline(x=best_e2, color=DIAG, linewidth=1.0, linestyle=":")
        ax_acc.annotate(f" ep {best_e2}",
                        xy=(best_e2, float(np.max(val_accs_))),
                        xytext=(best_e2 + 0.25, float(np.max(val_accs_)) - 0.03),
                        fontsize=7.5, color=CYAN)
    ax_acc.set_xlabel("Epoch",    fontsize=9.5, color=TEXT)
    ax_acc.set_ylabel("Accuracy", fontsize=9.5, color=TEXT)
    ax_acc.set_title("Accuracy Curve", fontsize=11, fontweight="bold", color=TEXT, pad=9)
    ax_acc.set_ylim([max(0, float(train_accs_.min()) - 0.05), 1.05])
    leg2 = ax_acc.legend(fontsize=8.5, framealpha=0.3,
                          facecolor=PANEL, edgecolor=GRID)
    for t in leg2.get_texts():
        t.set_color(TEXT)

    fig.suptitle(f"{model_name} — Training History",
                 fontsize=13, fontweight="bold", color=TEXT, y=0.97)

    path = os.path.join(save_dir,
                        f"{model_name.replace(' ', '_')}_training_curves.png")
    _save(fig, path)

    # Also save individuals
    plot_loss_curve(train_losses, val_losses, model_name, save_dir)
    plot_accuracy_curve(list(train_accs), val_accs, model_name, save_dir)


# ── Main evaluation function ─────────────────────────────────

def evaluate_model(y_true, y_pred, model_name="Model", y_prob=None,
                   train_losses=None, train_accs=None,
                   val_losses=None,   val_accs=None):
    """
    Print comprehensive evaluation metrics and save curve images.

    Parameters
    ----------
    y_true       : array-like of int   — Ground-truth labels.
    y_pred       : array-like of int   — Predicted labels.
    model_name   : str                 — Identifier shown in output.
    y_prob       : array-like of float or None
                   Predicted probabilities for the positive class (label 1).
                   When provided → ROC curve + PR curve images are saved.
    train_losses : list of float or None — Loss per epoch → Loss curve image.
    train_accs   : list of float or None — Accuracy per epoch → Accuracy curve image.
    val_losses   : list of float or None — Validation loss per epoch (optional).
    val_accs     : list of float or None — Validation accuracy per epoch (optional).

    Returns
    -------
    dict with accuracy, precision, recall, f1_score, confusion_matrix,
    and (if supplied) roc_auc, average_precision.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    f1   = f1_score(y_true, y_pred, average="weighted")
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'─' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─' * 50}")
    print(f"  Accuracy:  {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Real  Predicted Fake")
    print(f"  Actual Real     {cm[0][0]:>13}  {cm[0][1]:>14}")
    print(f"  Actual Fake     {cm[1][0]:>13}  {cm[1][1]:>14}")
    print(f"\n  Classification Report:")
    target_names = [LABEL_MAP[0], LABEL_MAP[1]]
    report = classification_report(y_true, y_pred, target_names=target_names)
    for line in report.split("\n"):
        print(f"  {line}")

    results = {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_score":         f1,
        "confusion_matrix": cm,
    }

    print(f"\n  Curve images → {os.path.abspath(CURVES_DIR)}/")

    # ── ROC + PR curves ───────────────────────────────────────
    if y_prob is not None:
        y_prob  = np.asarray(y_prob)
        y_true_ = np.asarray(y_true)

        curve_stats = plot_curves_dashboard(y_true_, y_prob, model_name)
        plot_roc_curve(y_true_, y_prob, model_name)
        plot_pr_curve(y_true_, y_prob, model_name)

        results["roc_auc"]           = curve_stats["roc_auc"]
        results["average_precision"] = curve_stats["average_precision"]
        print(f"  ROC AUC:           {results['roc_auc']:.4f}")
        print(f"  Average Precision: {results['average_precision']:.4f}")
    else:
        print("  ℹ  Pass y_prob= to generate ROC / PR curve images.")

    # ── Loss + Accuracy curves ────────────────────────────────
    if train_losses is not None and train_accs is not None:
        plot_training_curves(train_losses, train_accs,
                             val_losses, val_accs, model_name)
    elif train_losses is not None:
        plot_loss_curve(train_losses, val_losses, model_name)
    elif train_accs is not None:
        plot_accuracy_curve(train_accs, val_accs, model_name)
    else:
        print("  ℹ  Pass train_losses= / train_accs= to generate training curve images.")

    print(f"{'─' * 50}\n")
    return results