"""
Reproduction & Benchmarking Script for Experiments in Paper
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Reproduces:
- Table 4: Backbone comparison (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Inception V3)
- Table 5: Quantum gate variations (H+CNOT, S+H+CNOT, S†+H+CNOT, RX+CNOT, H+CZ, H+SWAP, H+CRX, RX+CRX)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import torch

from quantum.config import TABLE_4_CONFIGS, TABLE_5_CONFIGS, TrainingConfig
from quantum.dataset import create_synthetic_fundus_dataset, get_dataloaders
from quantum.models import QuantumTransferLearningDR
from quantum.train import train_model


# Reference values published in arXiv:2405.01734v1
PAPER_TABLE_4_REFERENCE = {
    "resnet18": {"accuracy": 97.6, "f1_score": 97.9},
    "resnet34": {"accuracy": 97.6, "f1_score": 97.9},
    "resnet50": {"accuracy": 97.6, "f1_score": 97.9},
    "resnet101": {"accuracy": 97.8, "f1_score": 98.0},
    "resnet152": {"accuracy": 98.2, "f1_score": 98.3},
    "inception_v3": {"accuracy": 98.5, "f1_score": 98.4},
}

PAPER_TABLE_5_REFERENCE = {
    "s_hadamard_cnot": {"name": "S(Phase) + Hadamard & CNOT", "accuracy": 92.1, "f1_score": 92.4},
    "s_dagger_hadamard_cnot": {"name": "S†(Dagger) + Hadamard & CNOT", "accuracy": 92.3, "f1_score": 92.6},
    "rx_cnot": {"name": "RX & CNOT", "accuracy": 97.5, "f1_score": 97.9},
    "hadamard_cz": {"name": "Hadamard & CZ", "accuracy": 95.2, "f1_score": 95.4},
    "hadamard_swap": {"name": "Hadamard & SWAP", "accuracy": 94.8, "f1_score": 94.3},
    "hadamard_crx": {"name": "Hadamard & CRX", "accuracy": 97.8, "f1_score": 97.9},
    "rx_crx": {"name": "RX & CRX", "accuracy": 98.1, "f1_score": 98.4},
}


def benchmark_table_4(dataset_dir: str, epochs: int = 5, batch_size: int = 16, device: str = "cuda"):
    """
    Runs benchmarks across the 6 pre-trained CNN backbones with Hadamard & CNOT circuit.
    """
    print("\n" + "=" * 80)
    print("🔬 EXPERIMENT 1: Reproducing Table 4 - Performance Analysis across Classical Backbones")
    print("=" * 80)

    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    results = []

    for name, model_cfg in TABLE_4_CONFIGS.items():
        print(f"\n--- Testing Backbone: {model_cfg.backbone.value} ---")
        img_size = (299, 299) if model_cfg.backbone.value == "inception_v3" else (224, 224)

        train_loader, val_loader = get_dataloaders(
            dataset_dir=dataset_dir,
            batch_size=train_cfg.batch_size,
            image_size=img_size,
            apply_graham=True
        )

        model = QuantumTransferLearningDR(config=model_cfg)
        _, _, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg
        )

        ref = PAPER_TABLE_4_REFERENCE.get(name, {"accuracy": "-", "f1_score": "-"})
        results.append({
            "Backbone Model": model_cfg.backbone.value,
            "Run Accuracy (%)": f"{best_metrics['accuracy']*100:.2f}",
            "Run F1-Score (%)": f"{best_metrics['f1_score_macro']*100:.2f}",
            "Paper Reported Acc (%)": ref["accuracy"],
            "Paper Reported F1 (%)": ref["f1_score"]
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📊 Table 4 Benchmark Summary:")
    print(df.to_markdown(index=False))
    return df


def benchmark_table_5(dataset_dir: str, epochs: int = 5, batch_size: int = 16, device: str = "cuda"):
    """
    Runs benchmarks across the 7 quantum gate combinations using ResNet-18 backbone.
    """
    print("\n" + "=" * 80)
    print("🔬 EXPERIMENT 2: Reproducing Table 5 - Quantum Gate Variations (ResNet-18 Backbone)")
    print("=" * 80)

    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )

    train_loader, val_loader = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=train_cfg.batch_size,
        image_size=(224, 224),
        apply_graham=True
    )

    results = []

    for name, model_cfg in TABLE_5_CONFIGS.items():
        ref_info = PAPER_TABLE_5_REFERENCE.get(name, {"name": name, "accuracy": "-", "f1_score": "-"})
        print(f"\n--- Testing Quantum Circuit Configuration: {ref_info['name']} ---")

        model = QuantumTransferLearningDR(config=model_cfg)
        _, _, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg
        )

        results.append({
            "Quantum Circuit Configuration": ref_info["name"],
            "Run Accuracy (%)": f"{best_metrics['accuracy']*100:.2f}",
            "Run F1-Score (%)": f"{best_metrics['f1_score_macro']*100:.2f}",
            "Paper Reported Acc (%)": ref_info["accuracy"],
            "Paper Reported F1 (%)": ref_info["f1_score"]
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📊 Table 5 Benchmark Summary:")
    print(df.to_markdown(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Reproduce paper experiments and generate comparison tables")
    parser.add_argument("--table", type=int, default=4, choices=[4, 5, 0], help="4 for Table 4, 5 for Table 5, 0 for Both")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataset-dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if dataset_dir is None or not os.path.exists(dataset_dir):
        print("⚡ Creating synthetic fundus dataset for reproducible benchmarking...")
        dataset_dir = create_synthetic_fundus_dataset(
            output_dir="./data/synthetic_fundus",
            samples_per_stage=20
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.table in [4, 0]:
        benchmark_table_4(dataset_dir, epochs=args.epochs, batch_size=args.batch_size, device=device)

    if args.table in [5, 0]:
        benchmark_table_5(dataset_dir, epochs=args.epochs, batch_size=args.batch_size, device=device)


if __name__ == "__main__":
    main()
