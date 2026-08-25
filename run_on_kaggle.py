"""
Kaggle Automated Training & Benchmark Runner
Quantum Transfer Learning for Diabetic Retinopathy Detection (arXiv:2405.01734v1)

This script can be executed directly inside a Kaggle notebook or environment.
It automatically locates or downloads dataset 'bhavyasanghavi2348/data-qdit',
trains the hybrid quantum model, evaluates performance metrics, and saves model weights.
"""

import os
import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantum.config import (
    BackboneType,
    DR_STAGE_LABELS,
    EmbeddingGateType,
    EntanglingGateType,
    ModelConfig,
    QuantumCircuitConfig,
    TrainingConfig,
)
from quantum.dataset import get_dataloaders, resolve_dataset_path
from quantum.evaluate import format_metrics_table
from quantum.models import QuantumTransferLearningDR
from quantum.train import train_model


def run_kaggle_experiment(
    dataset_slug: str = "bhavyasanghavi2348/data-qdit",
    backbone: str = "resnet18",
    embedding_gate: str = "hadamard",
    entangling_gate: str = "cnot",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    save_path: str = "/kaggle/working/quantum_dr_model.pt"
):
    print("=" * 80)
    print("🌟 QDIT: Quantum Transfer Learning for Diabetic Retinopathy Detection")
    print("   Paper: arXiv:2405.01734v1")
    print("=" * 80)

    # 1. Resolve Dataset
    print(f"\n📂 Resolving Kaggle dataset: {dataset_slug}")
    dataset_dir = resolve_dataset_path(dataset_slug)
    print(f"📁 Dataset location: {dataset_dir}")

    # If running outside kaggle, save to local folder
    if not os.path.exists("/kaggle/working"):
        save_path = "quantum_dr_model.pt"

    # 2. Build Configurations
    circuit_cfg = QuantumCircuitConfig(
        n_qubits=4,
        q_depth=4,
        embedding_gate=EmbeddingGateType(embedding_gate),
        entangling_gate=EntanglingGateType(entangling_gate)
    )

    model_cfg = ModelConfig(
        backbone=BackboneType(backbone),
        quantum_circuit=circuit_cfg
    )

    train_cfg = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Create DataLoaders
    print("\n🔄 Initializing Graham-filtered Fundus DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=train_cfg.batch_size,
        image_size=(224, 224),
        apply_graham=True
    )
    print(f"📊 Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 4. Instantiate Hybrid CQ Model
    print(f"\n⚛️  Instantiating Hybrid Model: Backbone={backbone} | Gate={embedding_gate}+{entangling_gate} | Qubits=4")
    model = QuantumTransferLearningDR(config=model_cfg)

    # 5. Train Model
    trained_model, history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        save_path=save_path
    )

    # 6. Output Final Results
    print("\n" + "=" * 80)
    print("🎯 FINAL EVALUATION RESULTS:")
    print(format_metrics_table(best_metrics))
    print("=" * 80)

    return trained_model, history, best_metrics


if __name__ == "__main__":
    run_kaggle_experiment()
