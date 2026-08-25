"""
Quantum Transfer Learning for Diabetic Retinopathy Detection (QDIT)
Implementation of arXiv:2405.01734v1
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path if not present
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quantum.config import (
    BackboneType,
    DR_STAGE_LABELS,
    EmbeddingGateType,
    EntanglingGateType,
    ModelConfig,
    QuantumCircuitConfig,
    TABLE_4_CONFIGS,
    TABLE_5_CONFIGS,
    TrainingConfig,
)
from quantum.circuits import create_quantum_circuit_qnode, draw_circuit
from quantum.dataset import (
    DiabeticRetinopathyDataset,
    apply_graham_gaussian_filter,
    create_synthetic_fundus_dataset,
    generate_synthetic_fundus_image,
    get_dataloaders,
    get_fundus_transforms,
)
from quantum.evaluate import compute_metrics, evaluate_model, format_metrics_table
from quantum.infer import QuantumDRPredictor
from quantum.models import DressedQuantumNet, FastTorchQuantumCircuit, QuantumTransferLearningDR
from quantum.train import train_model

__all__ = [
    "QuantumTransferLearningDR",
    "DressedQuantumNet",
    "FastTorchQuantumCircuit",
    "QuantumCircuitConfig",
    "ModelConfig",
    "TrainingConfig",
    "BackboneType",
    "EmbeddingGateType",
    "EntanglingGateType",
    "TABLE_4_CONFIGS",
    "TABLE_5_CONFIGS",
    "DR_STAGE_LABELS",
    "create_quantum_circuit_qnode",
    "draw_circuit",
    "DiabeticRetinopathyDataset",
    "apply_graham_gaussian_filter",
    "generate_synthetic_fundus_image",
    "create_synthetic_fundus_dataset",
    "get_dataloaders",
    "get_fundus_transforms",
    "compute_metrics",
    "evaluate_model",
    "format_metrics_table",
    "QuantumDRPredictor",
    "train_model",
]
