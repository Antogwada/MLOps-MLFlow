"""Utility helpers for plotting and metrics evaluation."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def plot_roc_curve(y_true, y_prob, output_path: str = "artifacts/roc_curve.png") -> float:
    """Génère et sauvegarde la courbe ROC."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Spécificité)")
    plt.ylabel("True Positive Rate (Sensibilité)")
    plt.title("Receiver Operating Characteristic (ROC) - Telco Churn")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return roc_auc


def plot_pr_curve(y_true, y_prob, output_path: str = "artifacts/pr_curve.png") -> float:
    """Génère et sauvegarde la courbe Précision-Rappel."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, color="purple", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Rappel (Recall)")
    plt.ylabel("Précision")
    plt.title("Precision-Recall Curve - Telco Churn")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, output_path: str = "artifacts/confusion_matrix.png"):
    """Génère et sauvegarde la matrice de confusion."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn (0)", "Churn (1)"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix - Telco Churn")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
