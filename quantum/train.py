"""
Training Pipeline for Quantum Transfer Learning in Diabetic Retinopathy Detection
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- CrossEntropyLoss (Section 4.6)
- Adam Optimizer (Section 4.6)
- StepLR Learning Rate Scheduler with decay factor every 10 epochs (Section 4.6)
- Checkpoint management and training history tracking
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict, List, Optional, Tuple
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, List[float]], Dict]:
    """
    Executes the training and validation loops as specified in Section 4.6 of the paper.
    Fully optimized for NVIDIA GPU accelerators (T4, P100, V100, A100, RTX).
    """
    use_cuda = torch.cuda.is_available() and config.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
        print(f"🚀 [FULL GPU MODE] Active Device: {gpu_name} ({gpu_vram:.1f} GB VRAM) | CUDA: {cuda_version}")
        torch.backends.cudnn.benchmark = True
    else:
        print("💻 Running on CPU device.")

    model.to(device)

    # 1. Loss function: Cross-Entropy Loss (Section 4.6)
    criterion = nn.CrossEntropyLoss()

    # 2. Optimizer: Adam optimizer updating only trainable (quantum + dressed head) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

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

    print(f"📦 Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 70)

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
    print(f"✅ Training completed in {total_training_time:.2f}s. Best Val Accuracy: {best_val_acc*100:.2f}%")
    if save_path:
        print(f"💾 Best model weights saved to: {save_path}")

    return model, history, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Quantum Transfer Learning Model for Diabetic Retinopathy")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=[b.value for b in BackboneType])
    parser.add_argument("--embedding", type=str, default="hadamard", choices=[e.value for e in EmbeddingGateType])
    parser.add_argument("--entangling", type=str, default="cnot", choices=[e.value for e in EntanglingGateType])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--q-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset-dir", type=str, default=None, help="Local dataset path or folder")
    parser.add_argument("--dataset-kaggle", type=str, default="bhavyasanghavi2348/data-qdit", help="Kaggle dataset slug e.g. bhavyasanghavi2348/data-qdit")
    parser.add_argument("--use-synthetic", action="store_true", default=False)
    parser.add_argument("--use-pennylane", action="store_true", default=False)
    parser.add_argument("--save-path", type=str, default="quantum_dr_model.pt")
    args = parser.parse_args()

    # Determine dataset directory (checks local, kaggle input, kagglehub, or synthetic fallback)
    dataset_source = args.dataset_dir or args.dataset_kaggle
    if args.use_synthetic:
        dataset_source = None

    dataset_dir = None
    if dataset_source:
        from quantum.dataset import resolve_dataset_path
        dataset_dir = resolve_dataset_path(dataset_source)

    if dataset_dir is None or not os.path.exists(dataset_dir):
        print("⚡ Creating synthetic fundus dataset for immediate demonstration & training...")
        dataset_dir = create_synthetic_fundus_dataset(
            output_dir="./data/synthetic_fundus",
            samples_per_stage=25
        )

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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load DataLoaders
    train_loader, val_loader = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=train_cfg.batch_size,
        apply_graham=train_cfg.apply_graham_filter
    )

    # Instantiate hybrid CQ model
    model = QuantumTransferLearningDR(
        config=model_cfg,
        use_pennylane=args.use_pennylane
    )

    # Train
    _, history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        save_path=args.save_path
    )

    print("\n" + format_metrics_table(best_metrics))


if __name__ == "__main__":
    main()
