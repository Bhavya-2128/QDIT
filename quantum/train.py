"""
Training Pipeline for Quantum Transfer Learning in Diabetic Retinopathy Detection
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- CrossEntropyLoss (Section 4.6)
- Adam Optimizer (Section 4.6)
- StepLR Learning Rate Scheduler with decay factor every 10 epochs (Section 4.6)
- Checkpoint management and training history tracking
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from quantum.config import (
    BackboneType,
    DR_STAGE_LABELS,
    EmbeddingGateType,
    EntanglingGateType,
    ModelConfig,
    QuantumCircuitConfig,
    TrainingConfig,
)
from quantum.dataset import create_synthetic_fundus_dataset, get_dataloaders
from quantum.evaluate import evaluate_model, format_metrics_table
from quantum.models import QuantumTransferLearningDR


class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for multi-class Diabetic Retinopathy classification.
    Focuses gradient updates on hard borderline stages (Mild & Moderate DR) to maximize macro F1-score.
    """
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, List[float]], Dict]:
    """
    Executes the training and validation loops with GPU mixed-precision,
    differential optimizer learning rates, and Focal Loss / Label Smoothing.
    """
    use_cuda = torch.cuda.is_available() and config.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        try:
            # Pre-flight CUDA sanity check to verify device compute capability (e.g. catches P100 sm_60 incompatibility)
            test_tensor = torch.zeros((1, 1, 3, 3), device=device)
            test_conv = nn.Conv2d(1, 1, 1).to(device)
            _ = test_conv(test_tensor)
            del test_tensor, test_conv
            torch.cuda.empty_cache()

            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda
            print(f"🚀 [FULL GPU MODE] Active Device: {gpu_name} ({gpu_vram:.1f} GB VRAM) | CUDA: {cuda_version}")
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            err_msg = str(e)
            print("\n" + "!" * 80)
            print("⚠️  CUDA INCOMPATIBILITY DETECTED (cudaErrorNoKernelImageForDevice):")
            print(f"   {err_msg}")
            print("\n📋 HOW TO FIX ON KAGGLE:")
            print("   1. Switch Accelerator: In Kaggle's right sidebar under 'Notebook options',")
            print("      set Accelerator to 'GPU T4 x2' (Tesla T4) instead of 'GPU P100'.")
            print("   2. Avoid overwriting PyTorch: Do NOT run `pip install torchvision` or `pip install torch`.")
            print("      Only run: `!pip install -q pennylane kagglehub`")
            print("   3. Restart Session: Click 'Session' -> 'Restart Session' in Kaggle's top menu.")
            print("!" * 80 + "\n")
            print("🔄 Falling back to CPU device to prevent crashes...")
            device = torch.device("cpu")
            use_cuda = False
    else:
        print("💻 Running on CPU device.")

    model.to(device)

    # 1. Loss function: Focal Loss or Cross-Entropy with Label Smoothing
    loss_type = getattr(config, 'loss_type', 'focal')
    label_smoothing = getattr(config, 'label_smoothing', 0.05)
    focal_gamma = getattr(config, 'focal_gamma', 1.5)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"🎯 Objective: Focal Loss (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    elif loss_type == "label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"🎯 Objective: CrossEntropyLoss (label_smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("🎯 Objective: Standard CrossEntropyLoss")

    # 2. Differential Optimizer: Lower LR for backbone fine-tuning, higher LR for quantum head
    backbone_params = []
    quantum_head_params = []

    if hasattr(model, 'backbone') and hasattr(model, 'dressed_quantum_net'):
        for param in model.backbone.parameters():
            if param.requires_grad:
                backbone_params.append(param)
        for param in model.dressed_quantum_net.parameters():
            if param.requires_grad:
                quantum_head_params.append(param)
    else:
        for param in model.parameters():
            if param.requires_grad:
                quantum_head_params.append(param)

    backbone_lr = config.learning_rate * getattr(config, 'backbone_lr_ratio', 0.1)
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': config.weight_decay
        })
        print(f"🔧 Backbone fine-tuning active ({sum(p.numel() for p in backbone_params):,} params, LR={backbone_lr:.1e})")
    if quantum_head_params:
        param_groups.append({
            'params': quantum_head_params,
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay
        })
        print(f"⚛️  Dressed Quantum Head active ({sum(p.numel() for p in quantum_head_params):,} params, LR={config.learning_rate:.1e})")

    optimizer = torch.optim.Adam(param_groups)

    # 3. Learning rate scheduler: StepLR decreasing LR by gamma every 10 epochs (Section 4.6)
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_decay_step,
        gamma=config.lr_decay_gamma
    )

    # Automatic mixed precision scaler for CUDA
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_metrics = {}
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0

        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass through hybrid CQ model with mixed precision on GPU
            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits, _, _ = model(inputs)
                loss = criterion(logits, targets)

            # Backward pass & quantum variational parameter updates
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track statistics
            preds = torch.argmax(logits, dim=-1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data).item()
            total_train_samples += inputs.size(0)

        # Step LR scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        epoch_train_loss = running_loss / max(total_train_samples, 1)
        epoch_train_acc = running_corrects / max(total_train_samples, 1)

        # Validation evaluation
        val_metrics, val_loss = evaluate_model(model, val_loader, device=str(device), criterion=criterion)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_score_macro"]

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | "
            f"Val F1: {val_f1*100:5.2f}% | Time: {epoch_duration:.2f}s"
        )

        # Save checkpoint if best accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "val_metrics": val_metrics,
                    "config": config
                }, save_path)

    total_training_time = time.time() - start_time
    print("=" * 70)
    print(f"✅ Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f} min). Best Val Accuracy: {best_val_acc*100:.2f}%")
    if save_path:
        print(f"💾 Best model weights saved to: {save_path}")

    return model, history, best_metrics


def plot_learning_curves(history: Dict[str, List[float]], output_path: str = "training_curves.png") -> None:
    """Plots and saves loss, accuracy, and Macro F1 curves across epochs."""
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(14, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
        plt.plot(epochs_range, history["val_loss"], 'r-s', label='Val Loss')
        plt.title('Training & Validation Loss (Focal Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        # Accuracy & Macro F1 Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Acc (%)')
        plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], 'g-s', label='Val Acc (%)')
        if "val_f1" in history:
            plt.plot(epochs_range, [f * 100 for f in history["val_f1"]], 'm--', label='Val Macro F1 (%)')
        plt.title('Validation Accuracy & Macro F1 (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Score (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"📊 Training curves saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save plot curves: {e}")


def run_inference_on_test_images(
    model: nn.Module,
    dataset_dir: str,
    device: torch.device,
    output_csv: str = "submission.csv"
) -> Optional[pd.DataFrame]:
    """Runs model inference on test_images/ and outputs submission.csv."""
    import pandas as pd
    from PIL import Image
    from quantum.dataset import get_fundus_transforms, resolve_dataset_path

    resolved = resolve_dataset_path(dataset_dir)
    test_csv_path = os.path.join(resolved, "test.csv")
    test_img_dir = os.path.join(resolved, "test_images")

    if not os.path.exists(test_img_dir):
        print(f"ℹ️  No test_images directory found at {test_img_dir}. Skipping test inference.")
        return None

    print("\n" + "=" * 80)
    print("🔍 EVALUATING & RUNNING INFERENCE ON test_images/...")
    print("=" * 80)

    test_transform = get_fundus_transforms(image_size=(224, 224), is_training=False, apply_graham=True)
    model.eval()

    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
    else:
        # Construct dataframe from image files
        img_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        test_df = pd.DataFrame({'id_code': [Path(f).stem for f in img_files]})

    predictions = []
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for i, row in test_df.iterrows():
            img_id = str(row['id_code'])
            name = img_id if img_id.endswith(('.png', '.jpg', '.jpeg')) else f"{img_id}.png"
            p = os.path.join(test_img_dir, name)

            if not os.path.exists(p):
                for ext in ['.jpg', '.jpeg', '.tiff']:
                    p_alt = os.path.join(test_img_dir, f"{img_id}{ext}")
                    if os.path.exists(p_alt):
                        p = p_alt
                        break

            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                tensor = test_transform(img).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _, _ = model(tensor)
                pred = torch.argmax(logits, dim=-1).item()
                predictions.append(pred)
            else:
                predictions.append(0)

            if (i + 1) % 500 == 0 or (i + 1) == len(test_df):
                print(f"   Processed [{i+1}/{len(test_df)}] test images...")

    test_df['diagnosis'] = predictions
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    test_df[['id_code', 'diagnosis']].to_csv(output_csv, index=False)
    print(f"\n✅ Generated test predictions saved to: {output_csv}")
    print("\nPredicted Class Distribution on Test Set:")
    print(test_df['diagnosis'].value_counts().sort_index())
    print("\nSample Predictions:")
    print(test_df[['id_code', 'diagnosis']].head(10))
    return test_df


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Quantum Model on APTOS 2019 Retinal Scans")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=[b.value for b in BackboneType])
    parser.add_argument("--embedding", type=str, default="hadamard", choices=[e.value for e in EmbeddingGateType])
    parser.add_argument("--entangling", type=str, default="cnot", choices=[e.value for e in EntanglingGateType])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--q-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to aptos2019-blindness-detection directory")
    parser.add_argument("--fine-tune-backbone", action="store_true", default=True)
    parser.add_argument("--loss-type", type=str, default="focal", choices=["focal", "label_smoothing", "ce"])
    parser.add_argument("--save-path", type=str, default="quantum_dr_model.pt")
    parser.add_argument("--use-synthetic", action="store_true", default=False)
    args = parser.parse_args()

    quantum_dir = os.path.dirname(os.path.abspath(__file__))

    # Auto-resolve dataset directory
    from quantum.dataset import resolve_dataset_path
    dataset_dir = resolve_dataset_path(args.dataset_dir)
    print(f"📁 Dataset Directory: {dataset_dir}")

    # Build configurations
    circuit_cfg = QuantumCircuitConfig(
        n_qubits=args.n_qubits,
        q_depth=args.q_depth,
        embedding_gate=EmbeddingGateType(args.embedding),
        entangling_gate=EntanglingGateType(args.entangling)
    )

    model_cfg = ModelConfig(
        backbone=BackboneType(args.backbone),
        quantum_circuit=circuit_cfg
    )

    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_type=args.loss_type,
        backbone_lr_ratio=0.1 if args.fine_tune_backbone else 0.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. Load DataLoaders (Stratified 80/20 train/val split with Graham filter)
    print("\n🔄 Loading retinal images from train_images/...")
    train_loader, val_loader = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=train_cfg.batch_size,
        apply_graham=True
    )
    print(f"📊 Training Batches: {len(train_loader)} | Validation Batches: {len(val_loader)}")

    # 2. Instantiate hybrid CQ model
    print(f"\n⚛️  Instantiating Quantum Transfer Learning DR Model (Backbone: {args.backbone}, Qubits: {args.n_qubits})...")
    model = QuantumTransferLearningDR(config=model_cfg)

    # 3. Train Model from scratch
    save_file = os.path.join(quantum_dir, args.save_path) if not os.path.isabs(args.save_path) else args.save_path
    trained_model, history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        save_path=save_file
    )

    # 4. Display Final Evaluation Metrics Table
    print("\n" + "=" * 80)
    print("🎯 FINAL VALIDATION METRICS:")
    print(format_metrics_table(best_metrics))
    print("=" * 80)

    # 5. Plot and save learning curves
    plot_file = os.path.join(quantum_dir, "training_curves.png")
    plot_learning_curves(history, output_path=plot_file)

    # 6. Run Inference & Testing on test_images/ -> submission.csv
    device_obj = torch.device(train_cfg.device)
    submission_file = os.path.join(quantum_dir, "submission.csv")
    run_inference_on_test_images(trained_model, dataset_dir, device_obj, output_csv=submission_file)


if __name__ == "__main__":
    main()

