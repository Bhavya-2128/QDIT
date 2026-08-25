"""
Configuration and Hyperparameter Settings for Quantum Transfer Learning
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class BackboneType(str, Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    INCEPTION_V3 = "inception_v3"


class EmbeddingGateType(str, Enum):
    HADAMARD = "hadamard"           # H + RY(x)
    S_PHASE = "s_phase"             # S + H + RY(x)
    S_DAGGER = "s_dagger"           # S_dagger + H + RY(x)
    RX = "rx"                       # RX(x)
    RY = "ry"                       # RY(x)


class EntanglingGateType(str, Enum):
    CNOT = "cnot"                   # Standard 2-qubit CNOT
    CZ = "cz"                       # Controlled-Z gate
    SWAP = "swap"                   # SWAP gate
    CRX = "crx"                     # Controlled-RX rotation gate
    CRY = "cry"                     # Controlled-RY rotation gate
    CRZ = "crz"                     # Controlled-RZ rotation gate


@dataclass
class QuantumCircuitConfig:
    n_qubits: int = 4
    q_depth: int = 4
    embedding_gate: EmbeddingGateType = EmbeddingGateType.HADAMARD
    entangling_gate: EntanglingGateType = EntanglingGateType.CNOT
    device_name: str = "default.qubit"
    diff_method: str = "backprop"  # "backprop" or "parameter-shift"
    shots: Optional[int] = None    # None for exact state-vector simulation


@dataclass
class ModelConfig:
    backbone: BackboneType = BackboneType.RESNET18
    pretrained: bool = True
    freeze_backbone: bool = True
    num_classes: int = 5
    quantum_circuit: QuantumCircuitConfig = field(default_factory=QuantumCircuitConfig)
    dropout_rate: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    lr_decay_step: int = 10
    lr_decay_gamma: float = 0.5
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    image_size: Tuple[int, int] = (224, 224)
    apply_graham_filter: bool = True
    num_workers: int = 2


# --- Preset configurations reproducing Table 4 (Backbone comparison with H + CNOT) ---
TABLE_4_CONFIGS = {
    "resnet18": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "resnet34": ModelConfig(
        backbone=BackboneType.RESNET34,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "resnet50": ModelConfig(
        backbone=BackboneType.RESNET50,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "resnet101": ModelConfig(
        backbone=BackboneType.RESNET101,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "resnet152": ModelConfig(
        backbone=BackboneType.RESNET152,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "inception_v3": ModelConfig(
        backbone=BackboneType.INCEPTION_V3,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
}

# --- Preset configurations reproducing Table 5 (Quantum Gates variations with ResNet18) ---
TABLE_5_CONFIGS = {
    "hadamard_cnot": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "s_hadamard_cnot": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.S_PHASE,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "s_dagger_hadamard_cnot": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.S_DAGGER,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "rx_cnot": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.RX,
            entangling_gate=EntanglingGateType.CNOT
        )
    ),
    "hadamard_cz": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CZ
        )
    ),
    "hadamard_swap": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.SWAP
        )
    ),
    "hadamard_crx": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CRX
        )
    ),
    "rx_crx": ModelConfig(
        backbone=BackboneType.RESNET18,
        quantum_circuit=QuantumCircuitConfig(
            embedding_gate=EmbeddingGateType.RX,
            entangling_gate=EntanglingGateType.CRX
        )
    ),
}

DR_STAGE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}
