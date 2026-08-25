"""
Evaluation Suite & Performance Metrics for Quantum Transfer Learning
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- Equations 12-15: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- Multiclass Confusion Matrix, Macro/Weighted Metrics, and Stage-wise breakdowns
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quantum.config import DR_STAGE_LABELS


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int = 5
) -> Dict[str, Union[float, Dict[int, Dict[str, float]], np.ndarray]]:
    """
    Computes all standard performance evaluation metrics from Section 4.7 of the paper:
    - Overall Accuracy (Eq. 12)
    - Macro / Weighted Precision (Eq. 13)
    - Macro / Weighted Recall/Sensitivity (Eq. 14)
    - Macro / Weighted Specificity
    - Macro / Weighted F1-score (Eq. 15)
    - Stage-wise breakdown
    - 5x5 Confusion Matrix
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Accuracy
    total_samples = len(y_true)
    correct_samples = np.sum(y_true == y_pred)
    accuracy = float(correct_samples / total_samples) if total_samples > 0 else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class metrics (One-vs-Rest)
    per_class_metrics = {}
    specificities = []

    for c in range(num_classes):
        tp = float(cm[c, c])
        fn = float(np.sum(cm[c, :]) - tp)
        fp = float(np.sum(cm[:, c]) - tp)
        tn = float(total_samples - (tp + fn + fp))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[c] = {
            "label": DR_STAGE_LABELS.get(c, f"Stage {c}"),
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1,
            "support": int(np.sum(cm[c, :]))
        }
        specificities.append(spec)

    # Scikit-learn macro / weighted calculations
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    specificity_macro = float(np.mean(specificities))

    return {
        "accuracy": accuracy,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "specificity_macro": specificity_macro,
        "f1_score_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_score_weighted": float(f1_weighted),
        "per_class": per_class_metrics,
        "confusion_matrix": cm
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    criterion: Optional[nn.Module] = None
) -> Tuple[Dict[str, Union[float, Dict[int, Dict[str, float]], np.ndarray]], float]:
    """
    Evaluates the hybrid quantum model over a validation/test dataloader.
    Returns computed metrics and mean loss.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_expvals = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, q_expvals, _ = model(inputs)

            if criterion:
                loss = criterion(logits, targets)
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_expvals.extend(q_expvals.cpu().tolist())
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["avg_loss"] = avg_loss
    metrics["quantum_expvals"] = np.array(all_expvals)

    return metrics, avg_loss


def format_metrics_table(metrics: Dict) -> str:
    """
    Formats the evaluation results into a clean markdown table.
    """
    lines = [
        "### 📊 Quantum Transfer Learning Evaluation Results",
        "",
        f"- **Overall Accuracy**: `{metrics['accuracy'] * 100:.2f}%`",
        f"- **Macro Precision**: `{metrics['precision_macro'] * 100:.2f}%`",
        f"- **Macro Recall (Sensitivity)**: `{metrics['recall_macro'] * 100:.2f}%`",
        f"- **Macro Specificity**: `{metrics['specificity_macro'] * 100:.2f}%`",
        f"- **Macro F1-Score**: `{metrics['f1_score_macro'] * 100:.2f}%`",
        "",
        "#### Stage-wise Performance Breakdown",
        "",
        "| Stage | Diagnosis | Precision | Sensitivity | Specificity | F1-Score | Support |",
        "|---|---|---|---|---|---|---|"
    ]

    for c, data in metrics["per_class"].items():
        lines.append(
            f"| {c} | {data['label']} | {data['precision']*100:.2f}% | {data['recall_sensitivity']*100:.2f}% | "
            f"{data['specificity']*100:.2f}% | {data['f1_score']*100:.2f}% | {data['support']} |"
        )

    return "\n".join(lines)
