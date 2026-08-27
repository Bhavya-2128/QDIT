"""
Kaggle Automated Training & Benchmark Runner
Quantum Transfer Learning for Diabetic Retinopathy Detection (arXiv:2405.01734v1)

This script can be executed directly inside a Kaggle notebook or environment.
It automatically locates or downloads dataset 'bhavyasanghavi2348/data-qdit',
trains the hybrid quantum model, evaluates performance metrics, and saves model weights.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
    dataset_slug: str = "aptos2019-blindness-detection",
    backbone: str = "resnet18",
    embedding_gate: str = "hadamard",
    entangling_gate: str = "cnot",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    fine_tune_backbone: bool = True,
    loss_type: str = "focal",
    save_path: str = "/kaggle/working/quantum_dr_model.pt"
):
    print("=" * 80)
    print("🌟 QDIT: Quantum Transfer Learning for Diabetic Retinopathy Detection")
    print("   Paper: arXiv:2405.01734v1 (Enhanced Multi-Format & Deep CQ Architecture)")
    print("=" * 80)

    if torch.cuda.is_available():
        try:
            test_t = torch.zeros((1, 1, 3, 3), device="cuda")
            test_c = torch.nn.Conv2d(1, 1, 1).to("cuda")
            _ = test_c(test_t)
            del test_t, test_c
            torch.cuda.empty_cache()

            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"🚀 [KAGGLE GPU ACCELERATION ENABLED] Using {gpu_name} ({gpu_vram:.1f} GB VRAM)")
        except Exception as e:
            print("\n" + "!" * 80)
            print("⚠️  CUDA INCOMPATIBILITY DETECTED ON KAGGLE:")
            print(f"   {e}")
            print("\n📋 QUICK FIX:")
            print("   1. Switch Accelerator to 'GPU T4 x2' in Kaggle Notebook Options (P100 sm_60 is not supported by newer PyTorch).")
            print("   2. Avoid `pip install torchvision` which breaks Kaggle's pre-installed CUDA PyTorch.")
            print("   3. Restart Session via Session -> Restart Session.")
            print("!" * 80 + "\n")
    else:
        print("⚠️  [NOTICE] Running on CPU. For faster training, enable 'GPU T4 x2' in Kaggle Notebook Settings.")

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
        quantum_circuit=circuit_cfg,
        freeze_backbone=not fine_tune_backbone,
        unfreeze_last_n_layers=1 if fine_tune_backbone else 0,
        enhanced_projection=True
    )

    train_cfg = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_type=loss_type,
        backbone_lr_ratio=0.1 if fine_tune_backbone else 0.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3. Create DataLoaders
    print("\n🔄 Initializing Graham-filtered Fundus DataLoaders (with multi-format auto-cropping & contrast normalization)...")
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

    # 7. Plot and Save Curves
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
        plt.plot(epochs_range, history["val_loss"], 'r-s', label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Focal Loss')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Acc')
        plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], 'g-s', label='Val Acc')
        plt.plot(epochs_range, [f * 100 for f in history["val_f1"]], 'm--', label='Val Macro F1')
        plt.title('Validation Accuracy & Macro F1 (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Score (%)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_path = "/kaggle/working/training_curves.png" if os.path.exists("/kaggle/working") else "training_curves.png"
        plt.savefig(plot_path, dpi=300)
        print(f"📊 Training curves saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️ Could not save plot: {e}")

    # 8. Automated Test Inference on test_images/
    test_csv_path = os.path.join(dataset_dir, "test.csv")
    test_img_dir = os.path.join(dataset_dir, "test_images")

    if os.path.exists(test_csv_path) and os.path.exists(test_img_dir):
        from PIL import Image
        from quantum.dataset import get_fundus_transforms
        import pandas as pd

        print("\n" + "=" * 80)
        print("🔍 RUNNING INFERENCE ON test_images/ (generating submission.csv)")
        print("=" * 80)

        test_df = pd.read_csv(test_csv_path)
        test_transform = get_fundus_transforms(image_size=(224, 224), is_training=False, apply_graham=True)

        if save_path and os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=train_cfg.device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)

        model.eval()
        device_torch = torch.device(train_cfg.device)
        predictions = []

        with torch.no_grad():
            for i, row in test_df.iterrows():
                img_name = str(row['id_code'])
                if not img_name.endswith('.png'):
                    img_name += '.png'
                img_path = os.path.join(test_img_dir, img_name)

                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    tensor = test_transform(img).unsqueeze(0).to(device_torch)
                    logits, _, _ = model(tensor)
                    pred = torch.argmax(logits, dim=-1).item()
                    predictions.append(pred)
                else:
                    predictions.append(0)

                if (i + 1) % 500 == 0 or (i + 1) == len(test_df):
                    print(f"   Processed [{i+1}/{len(test_df)}] test scans...")

        test_df['diagnosis'] = predictions
        sub_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
        submission_path = os.path.join(sub_dir, "submission.csv")
        test_df[['id_code', 'diagnosis']].to_csv(submission_path, index=False)
        print(f"\n✅ Generated submission file: {submission_path}")
        print(test_df[['id_code', 'diagnosis']].head(10))

    return trained_model, history, best_metrics


def run_all_paper_combinations(
    dataset_slug: str = "bhavyasanghavi2348/data-qdit",
    experiment_type: str = "all",  # "table4", "table5", or "all"
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    fine_tune_backbone: bool = True,
    loss_type: str = "focal",
    output_dir: str = "/kaggle/working"
) -> pd.DataFrame:
    """
    Sequentially runs every single backbone and quantum gate combination published in the paper:
    1. Table 4: Backbone comparison (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Inception-V3)
    2. Table 5: Quantum gate variations (H+CNOT, S+CNOT, S†+CNOT, RX+CNOT, RY+CNOT, H+CZ, H+SWAP, H+CRX, RX+CRX, RY+CRY, H+CRZ)
    """
    import pandas as pd
    from quantum.config import TABLE_4_CONFIGS, TABLE_5_CONFIGS
    from quantum.reproduce_paper_experiments import PAPER_TABLE_4_REFERENCE, PAPER_TABLE_5_REFERENCE

    print("=" * 90)
    print(f"🌟 RUNNING ALL PAPER EXPERIMENTS SEQUENTIALLY: mode={experiment_type.upper()}")
    print("=" * 90)

    dataset_dir = resolve_dataset_path(dataset_slug)
    print(f"📁 Dataset location: {dataset_dir}")

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    summary_records = []

    # -------------------------------------------------------------
    # 1. Run Table 4 (Backbone Architectures with Hadamard + CNOT)
    # -------------------------------------------------------------
    if experiment_type in ["table4", "all"]:
        print("\n" + "=" * 90)
        print("🔬 TABLE 4: Benchmarking 6 Pretrained Backbones with Hadamard + CNOT")
        print("=" * 90)

        for name, model_cfg in TABLE_4_CONFIGS.items():
            print(f"\n▶️ [Table 4 - {name.upper()}] Starting training for Backbone: {model_cfg.backbone.value}...")
            img_size = (299, 299) if model_cfg.backbone.value == "inception_v3" else (224, 224)

            train_loader, val_loader = get_dataloaders(
                dataset_dir=dataset_dir,
                batch_size=batch_size,
                image_size=img_size,
                apply_graham=True
            )

            # Update model config with fine-tuning & enhanced projection
            model_cfg.freeze_backbone = not fine_tune_backbone
            model_cfg.unfreeze_last_n_layers = 1 if fine_tune_backbone else 0
            model_cfg.enhanced_projection = True

            train_cfg = TrainingConfig(
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                loss_type=loss_type,
                backbone_lr_ratio=0.1 if fine_tune_backbone else 0.0,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            model = QuantumTransferLearningDR(config=model_cfg)
            ckpt_path = os.path.join(output_dir, "checkpoints", f"best_table4_{name}.pt")

            _, _, best_metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_cfg,
                save_path=ckpt_path
            )

            ref = PAPER_TABLE_4_REFERENCE.get(name, {"accuracy": "-", "f1_score": "-"})
            record = {
                "Experiment": "Table 4 (Backbone)",
                "Configuration": model_cfg.backbone.value.upper(),
                "Embedding": "Hadamard",
                "Entangling": "CNOT",
                "Accuracy (%)": round(best_metrics["accuracy"] * 100, 2),
                "F1-Score (%)": round(best_metrics["f1_score_macro"] * 100, 2),
                "Sensitivity (%)": round(best_metrics["recall_macro"] * 100, 2),
                "Specificity (%)": round(best_metrics["specificity_macro"] * 100, 2),
                "Paper Acc (%)": ref["accuracy"],
                "Paper F1 (%)": ref["f1_score"]
            }
            summary_records.append(record)
            print(f"✅ Completed Table 4 - {name.upper()}: Acc={record['Accuracy (%)']}%, F1={record['F1-Score (%)']}%")

    # -------------------------------------------------------------
    # 2. Run Table 5 (Quantum Gate Variations with ResNet-18)
    # -------------------------------------------------------------
    if experiment_type in ["table5", "all"]:
        print("\n" + "=" * 90)
        print("🔬 TABLE 5: Benchmarking 11 Quantum Embedding & Entangling Gate Variations")
        print("=" * 90)

        # Standard 224x224 DataLoader for Table 5
        train_loader, val_loader = get_dataloaders(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            image_size=(224, 224),
            apply_graham=True
        )

        for name, model_cfg in TABLE_5_CONFIGS.items():
            emb = model_cfg.quantum_circuit.embedding_gate.value
            ent = model_cfg.quantum_circuit.entangling_gate.value
            print(f"\n▶️ [Table 5 - {name.upper()}] Starting Gate Variation: Embedding={emb.upper()} | Entangling={ent.upper()}...")

            model_cfg.freeze_backbone = not fine_tune_backbone
            model_cfg.unfreeze_last_n_layers = 1 if fine_tune_backbone else 0
            model_cfg.enhanced_projection = True

            train_cfg = TrainingConfig(
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                loss_type=loss_type,
                backbone_lr_ratio=0.1 if fine_tune_backbone else 0.0,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            model = QuantumTransferLearningDR(config=model_cfg)
            ckpt_path = os.path.join(output_dir, "checkpoints", f"best_table5_{name}.pt")

            _, _, best_metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=train_cfg,
                save_path=ckpt_path
            )

            ref = PAPER_TABLE_5_REFERENCE.get(name, {"accuracy": "-", "f1_score": "-"})
            record = {
                "Experiment": "Table 5 (Gates)",
                "Configuration": f"{emb.upper()} + {ent.upper()}",
                "Embedding": emb.upper(),
                "Entangling": ent.upper(),
                "Accuracy (%)": round(best_metrics["accuracy"] * 100, 2),
                "F1-Score (%)": round(best_metrics["f1_score_macro"] * 100, 2),
                "Sensitivity (%)": round(best_metrics["recall_macro"] * 100, 2),
                "Specificity (%)": round(best_metrics["specificity_macro"] * 100, 2),
                "Paper Acc (%)": ref["accuracy"],
                "Paper F1 (%)": ref["f1_score"]
            }
            summary_records.append(record)
            print(f"✅ Completed Table 5 - {name.upper()}: Acc={record['Accuracy (%)']}%, F1={record['F1-Score (%)']}%")

    # -------------------------------------------------------------
    # 3. Final Tabular Report Generation
    # -------------------------------------------------------------
    summary_df = pd.DataFrame(summary_records)
    csv_path = os.path.join(output_dir, "paper_experiment_results.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("🏆 ALL PAPER EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"📁 Results CSV saved to: {csv_path}")
    print("=" * 90)
    print(summary_df.to_markdown(index=False))

    return summary_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QDIT Kaggle Benchmark Runner")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "table4", "table5", "all"])
    parser.add_argument("--dataset", type=str, default="aptos2019-blindness-detection")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "single":
        run_kaggle_experiment(dataset_slug=args.dataset, epochs=args.epochs, batch_size=args.batch_size)
    else:
        run_all_paper_combinations(dataset_slug=args.dataset, experiment_type=args.mode, epochs=args.epochs, batch_size=args.batch_size)
