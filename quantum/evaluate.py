"""
Evaluation Suite & Performance Metrics for Quantum Transfer Learning
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- Equations 12-15: Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score
- Multiclass Confusion Matrix, Macro/Weighted Metrics, and Stage-wise breakdowns
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main():
    import argparse
    import os
    import sys
    from pathlib import Path

    # Add project root to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from quantum.config import ModelConfig
    from quantum.dataset import get_dataloaders, resolve_dataset_path
    from quantum.models import QuantumTransferLearningDR
    from quantum.train import run_inference_on_test_images

    parser = argparse.ArgumentParser(description="Evaluate Quantum Transfer Learning Model on 5-Stage Retinal Dataset & test_images")
    parser.add_argument("--checkpoint", type=str, default="quantum/quantum_dr_model.pt", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--dataset-dir", type=str, default="aptos2019-blindness-detection", help="Path to APTOS / fundus dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation loader")
    parser.add_argument("--output-csv", type=str, default="quantum/submission.csv", help="Output path for test_images predictions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [EVALUATION] Active Device: {device.upper()}")

    # 1. Resolve dataset directory
    resolved_dir = resolve_dataset_path(args.dataset_dir)
    print(f"📁 Dataset Directory: {resolved_dir}")

    # 2. Instantiate Model
    model = QuantumTransferLearningDR(config=ModelConfig())

    # 3. Load checkpoint if exists
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(current_dir, checkpoint_path)
        if not os.path.exists(checkpoint_path) and os.path.exists(alt_path):
            checkpoint_path = alt_path

    if os.path.exists(checkpoint_path):
        print(f"📦 Loading weights from checkpoint: {checkpoint_path}")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Evaluating with initial weights.")

    model.to(device)

    # 4. Evaluate on Validation Set (Stages 0 to 4)
    print("\n🔄 Loading validation set (Stages 0 to 4)...")
    try:
        _, val_loader = get_dataloaders(
            dataset_dir=resolved_dir,
            batch_size=args.batch_size,
            apply_graham=True
        )
        print(f"📊 Validation Batches: {len(val_loader)}")
        metrics, avg_loss = evaluate_model(model, val_loader, device=device)

        print("\n" + "=" * 80)
        print("🎯 VALIDATION SET EVALUATION METRICS (Stages 0 to 4):")
        print(format_metrics_table(metrics))
        print("=" * 80)
        print("\nConfusion Matrix (5x5):")
        print(metrics["confusion_matrix"])
    except Exception as e:
        print(f"⚠️ Validation set evaluation skipped: {e}")

    # 5. Run inference on test_images/ folder
    test_img_dir = os.path.join(resolved_dir, "test_images")
    if os.path.exists(test_img_dir):
        print("\n" + "=" * 80)
        print(f"🔍 Evaluating and generating predictions for all images in {test_img_dir}...")
        print("=" * 80)
        device_obj = torch.device(device)
        run_inference_on_test_images(
            model=model,
            dataset_dir=resolved_dir,
            device=device_obj,
            output_csv=args.output_csv
        )
    else:
        print(f"ℹ️ No test_images folder found at {test_img_dir}")


if __name__ == "__main__":
    main()

