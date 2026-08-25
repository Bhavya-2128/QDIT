"""
Neural Network Models for Quantum Transfer Learning
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements:
- DressedQuantumNet (Classical Pre-layer -> tanh -> pi/2 scale -> Quantum VQC -> Classical Post-layer)
- QuantumTransferLearningDR (Pretrained Backbone + DressedQuantumNet)
- Fast PyTorch Differentiable Quantum Simulator for high-throughput GPU/CPU training
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torchvision.models as models

from quantum.circuits import create_quantum_circuit_qnode
from quantum.config import (
    BackboneType,
    EmbeddingGateType,
    EntanglingGateType,
    ModelConfig,
    QuantumCircuitConfig,
)


class FastTorchQuantumCircuit(nn.Module):
    """
    High-performance, pure PyTorch state-vector simulator implementing the exact
    quantum circuit described in the paper. Fully batched and differentiable on GPU/CPU.
    """

    def __init__(self, config: QuantumCircuitConfig):
        super().__init__()
        self.n_qubits = config.n_qubits
        self.q_depth = config.q_depth
        self.embedding_type = config.embedding_gate
        self.entangling_type = config.entangling_gate

        # Trainable quantum variational weights: shape (q_depth, n_qubits)
        self.weights = nn.Parameter(
            torch.randn(self.q_depth, self.n_qubits) * 0.1
        )

        # Standard quantum gates
        self._init_quantum_gate_matrices()

    def _init_quantum_gate_matrices(self):
        # 1-Qubit matrices registered as GPU buffers
        self.register_buffer("H", (1.0 / math.sqrt(2.0)) * torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64))
        self.register_buffer("S", torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex64))
        self.register_buffer("S_dagger", torch.tensor([[1.0, 0.0], [0.0, -1.0j]], dtype=torch.complex64))
        self.register_buffer("I1", torch.eye(2, dtype=torch.complex64))
        self.register_buffer("sigma_z", torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64))

        # 2-Qubit matrices registered as GPU buffers
        self.register_buffer("cnot_matrix", torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64))

        self.register_buffer("cz_matrix", torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex64))

        self.register_buffer("swap_matrix", torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.complex64))

    def _rx_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """Rx(theta) = cos(theta/2) I - i sin(theta/2) sigma_x"""
        cos = torch.cos(theta / 2.0).to(torch.complex64)
        sin = torch.sin(theta / 2.0).to(torch.complex64)
        row1 = torch.stack([cos, -1.0j * sin], dim=-1)
        row2 = torch.stack([-1.0j * sin, cos], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def _ry_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]"""
        cos = torch.cos(theta / 2.0).to(torch.complex64)
        sin = torch.sin(theta / 2.0).to(torch.complex64)
        row1 = torch.stack([cos, -sin], dim=-1)
        row2 = torch.stack([sin, cos], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def _ry_single(self, theta: torch.Tensor) -> torch.Tensor:
        """Single-angle Ry rotation gate constructed differentiably."""
        cos = torch.cos(theta / 2.0).to(torch.complex64)
        sin = torch.sin(theta / 2.0).to(torch.complex64)
        row1 = torch.stack([cos, -sin])
        row2 = torch.stack([sin, cos])
        return torch.stack([row1, row2])

    def _make_crx(self, theta: torch.Tensor) -> torch.Tensor:
        """Controlled-RX gate constructed differentiably."""
        cos = torch.cos(theta / 2.0).to(torch.complex64)
        sin = torch.sin(theta / 2.0).to(torch.complex64)
        zero = torch.zeros((), dtype=torch.complex64, device=theta.device)
        one = torch.ones((), dtype=torch.complex64, device=theta.device)
        r0 = torch.stack([one, zero, zero, zero])
        r1 = torch.stack([zero, one, zero, zero])
        r2 = torch.stack([zero, zero, cos, -1.0j * sin])
        r3 = torch.stack([zero, zero, -1.0j * sin, cos])
        return torch.stack([r0, r1, r2, r3])

    def _make_cry(self, theta: torch.Tensor) -> torch.Tensor:
        """Controlled-RY gate constructed differentiably."""
        cos = torch.cos(theta / 2.0).to(torch.complex64)
        sin = torch.sin(theta / 2.0).to(torch.complex64)
        zero = torch.zeros((), dtype=torch.complex64, device=theta.device)
        one = torch.ones((), dtype=torch.complex64, device=theta.device)
        r0 = torch.stack([one, zero, zero, zero])
        r1 = torch.stack([zero, one, zero, zero])
        r2 = torch.stack([zero, zero, cos, -sin])
        r3 = torch.stack([zero, zero, sin, cos])
        return torch.stack([r0, r1, r2, r3])

    def _make_crz(self, theta: torch.Tensor) -> torch.Tensor:
        """Controlled-RZ gate constructed differentiably."""
        phase_pos = torch.exp(-1.0j * theta / 2.0)
        phase_neg = torch.exp(1.0j * theta / 2.0)
        zero = torch.zeros((), dtype=torch.complex64, device=theta.device)
        one = torch.ones((), dtype=torch.complex64, device=theta.device)
        r0 = torch.stack([one, zero, zero, zero])
        r1 = torch.stack([zero, one, zero, zero])
        r2 = torch.stack([zero, zero, phase_pos, zero])
        r3 = torch.stack([zero, zero, zero, phase_neg])
        return torch.stack([r0, r1, r2, r3])

    def _apply_1q_gate(self, state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        subs = list('abcdefghij'[:self.n_qubits + 1])
        target_char = subs[qubit + 1]
        out_subs = subs.copy()
        out_subs[qubit + 1] = 'z'
        eqn = f"z{target_char},{''.join(subs)}->{''.join(out_subs)}"
        return torch.einsum(eqn, gate, state)

    def _apply_batched_1q_gate(self, state: torch.Tensor, batched_gate: torch.Tensor, qubit: int) -> torch.Tensor:
        subs = list('abcdefghij'[:self.n_qubits + 1])
        target_char = subs[qubit + 1]
        out_subs = subs.copy()
        out_subs[qubit + 1] = 'z'
        eqn = f"az{target_char},{''.join(subs)}->{''.join(out_subs)}"
        return torch.einsum(eqn, batched_gate, state)

    def _apply_2q_gate(self, state: torch.Tensor, gate_4x4: torch.Tensor, q0: int, q1: int) -> torch.Tensor:
        g_tensor = gate_4x4.view(2, 2, 2, 2)
        subs = list('abcdefghij'[:self.n_qubits + 1])
        t0 = subs[q0 + 1]
        t1 = subs[q1 + 1]
        out_subs = subs.copy()
        out_subs[q0 + 1] = 'y'
        out_subs[q1 + 1] = 'z'
        eqn = f"yz{t0}{t1},{''.join(subs)}->{''.join(out_subs)}"
        return torch.einsum(eqn, g_tensor, state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched quantum forward pass.
        Args:
            x: Scaled inputs of shape (batch_size, n_qubits).
        Returns:
            Expectation values of Pauli-Z on all qubits of shape (batch_size, n_qubits).
        """
        batch_size = x.shape[0]
        device = x.device

        # Initial state |0000>
        state = torch.zeros(batch_size, *(2 for _ in range(self.n_qubits)), dtype=torch.complex64, device=device)
        state[(slice(None),) + (0,) * self.n_qubits] = 1.0 + 0.0j

        # 1. State preparation / Embedding Layer
        for q in range(self.n_qubits):
            if self.embedding_type == EmbeddingGateType.HADAMARD:
                state = self._apply_1q_gate(state, self.H, q)
                ry_gate = self._ry_matrix(x[:, q])
                state = self._apply_batched_1q_gate(state, ry_gate, q)
            elif self.embedding_type == EmbeddingGateType.S_PHASE:
                state = self._apply_1q_gate(state, self.S, q)
                state = self._apply_1q_gate(state, self.H, q)
                ry_gate = self._ry_matrix(x[:, q])
                state = self._apply_batched_1q_gate(state, ry_gate, q)
            elif self.embedding_type == EmbeddingGateType.S_DAGGER:
                state = self._apply_1q_gate(state, self.S_dagger, q)
                state = self._apply_1q_gate(state, self.H, q)
                ry_gate = self._ry_matrix(x[:, q])
                state = self._apply_batched_1q_gate(state, ry_gate, q)
            elif self.embedding_type == EmbeddingGateType.RX:
                rx_gate = self._rx_matrix(x[:, q])
                state = self._apply_batched_1q_gate(state, rx_gate, q)
            elif self.embedding_type == EmbeddingGateType.RY:
                ry_gate = self._ry_matrix(x[:, q])
                state = self._apply_batched_1q_gate(state, ry_gate, q)

        # 2. Variational Layers
        for layer in range(self.q_depth):
            # Entangling gates
            for q in range(self.n_qubits):
                target_q = (q + 1) % self.n_qubits
                if self.entangling_type == EntanglingGateType.CNOT:
                    state = self._apply_2q_gate(state, self.cnot_matrix, q, target_q)
                elif self.entangling_type == EntanglingGateType.CZ:
                    state = self._apply_2q_gate(state, self.cz_matrix, q, target_q)
                elif self.entangling_type == EntanglingGateType.SWAP:
                    state = self._apply_2q_gate(state, self.swap_matrix, q, target_q)
                elif self.entangling_type == EntanglingGateType.CRX:
                    crx = self._make_crx(self.weights[layer, q])
                    state = self._apply_2q_gate(state, crx, q, target_q)
                elif self.entangling_type == EntanglingGateType.CRY:
                    cry = self._make_cry(self.weights[layer, q])
                    state = self._apply_2q_gate(state, cry, q, target_q)
                elif self.entangling_type == EntanglingGateType.CRZ:
                    crz = self._make_crz(self.weights[layer, q])
                    state = self._apply_2q_gate(state, crz, q, target_q)

            # Parameterized RY rotations
            for q in range(self.n_qubits):
                angle = self.weights[layer, q]
                ry_fixed = self._ry_single(angle)
                state = self._apply_1q_gate(state, ry_fixed, q)

        # 3. Measurement Layer: Pauli-Z expectation values
        expvals = []
        for q in range(self.n_qubits):
            z_state = self._apply_1q_gate(state, self.sigma_z, q)
            # Expectation value: Re(<state | Z | state>)
            dims = tuple(range(1, self.n_qubits + 1))
            expval = torch.real(torch.sum(torch.conj(state) * z_state, dim=dims))
            expvals.append(expval)

        return torch.stack(expvals, dim=-1)


class DressedQuantumNet(nn.Module):
    """
    Dressed Quantum Circuit Module as specified in Section 4.4 of the paper:
    1. Classical Pre-processing layer: nn.Linear(in_features, n_qubits)
    2. Classical Activation function: torch.tanh
    3. Constant scaling: np.pi / 2.0
    4. Variational Quantum Circuit: 4 qubits with trainable weights
    5. Classical Post-processing layer: nn.Linear(n_qubits, num_classes)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        circuit_config: Optional[QuantumCircuitConfig] = None,
        use_pennylane: bool = False
    ):
        super().__init__()
        self.circuit_config = circuit_config or QuantumCircuitConfig()
        self.in_features = in_features
        self.num_classes = num_classes
        self.n_qubits = self.circuit_config.n_qubits
        self.q_depth = self.circuit_config.q_depth
        self.use_pennylane = use_pennylane

        # Step 1: Classical Pre-processing layer
        self.pre_net = nn.Linear(self.in_features, self.n_qubits)

        # Step 4: Quantum Layer
        if self.use_pennylane:
            # PennyLane QNode setup
            self.qnode, self.dev = create_quantum_circuit_qnode(self.circuit_config)
            self.q_weights = nn.Parameter(
                torch.randn(self.q_depth, self.n_qubits) * 0.1
            )
        else:
            # Fast PyTorch batched state-vector quantum layer
            self.quantum_circuit = FastTorchQuantumCircuit(self.circuit_config)

        # Step 5: Classical Post-processing layer
        self.post_net = nn.Linear(self.n_qubits, self.num_classes)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dressed quantum network.
        Args:
            features: Abstract feature tensor from classical backbone of shape (batch_size, in_features).
        Returns:
            Tuple of:
              - logits: Output logits of shape (batch_size, num_classes).
              - q_out: Pauli-Z expectation values of shape (batch_size, n_qubits).
        """
        # Step 1, 2 & 3: Linear -> Tanh -> pi/2 scaling
        pre_out = self.pre_net(features)
        scaled_inputs = (np.pi / 2.0) * torch.tanh(pre_out)

        # Step 4: Quantum Circuit processing
        if self.use_pennylane:
            batch_size = scaled_inputs.shape[0]
            q_out_list = []
            for i in range(batch_size):
                res = self.qnode(scaled_inputs[i], self.q_weights)
                q_out_list.append(torch.stack(res))
            q_out = torch.stack(q_out_list, dim=0).to(features.device, dtype=torch.float32)
        else:
            q_out = self.quantum_circuit(scaled_inputs).to(dtype=torch.float32)

        # Step 5: Classical Post-processing layer
        logits = self.post_net(q_out)
        return logits, q_out


class QuantumTransferLearningDR(nn.Module):
    """
    Complete Classic-to-Quantum (CQ) Transfer Learning architecture for
    Diabetic Retinopathy detection (Section 3.5 & 3.6).

    Components:
    - Backbone Network A': Truncated pre-trained CNN (ResNet18-152, Inception-V3) with frozen weights.
    - Dressed Quantum Circuit B: Trainable quantum neural classifier.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        use_pennylane: bool = False
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.use_pennylane = use_pennylane

        # 1. Instantiate Classical Feature Extractor Backbone A'
        self.backbone, self.feature_dim = self._build_backbone(
            self.config.backbone,
            self.config.pretrained,
            self.config.freeze_backbone
        )

        # 2. Instantiate Dressed Quantum Classifier Head B
        self.dressed_quantum_net = DressedQuantumNet(
            in_features=self.feature_dim,
            num_classes=self.config.num_classes,
            circuit_config=self.config.quantum_circuit,
            use_pennylane=self.use_pennylane
        )

    def _build_backbone(
        self,
        backbone_type: BackboneType,
        pretrained: bool,
        freeze: bool
    ) -> Tuple[nn.Module, int]:
        """Loads torchvision CNN backbone, truncates final FC layer, and freezes parameters."""
        weights = "DEFAULT" if pretrained else None

        if backbone_type == BackboneType.RESNET18:
            model = models.resnet18(weights=weights)
            feature_dim = model.fc.in_features  # 512
            model.fc = nn.Identity()

        elif backbone_type == BackboneType.RESNET34:
            model = models.resnet34(weights=weights)
            feature_dim = model.fc.in_features  # 512
            model.fc = nn.Identity()

        elif backbone_type == BackboneType.RESNET50:
            model = models.resnet50(weights=weights)
            feature_dim = model.fc.in_features  # 2048
            model.fc = nn.Identity()

        elif backbone_type == BackboneType.RESNET101:
            model = models.resnet101(weights=weights)
            feature_dim = model.fc.in_features  # 2048
            model.fc = nn.Identity()

        elif backbone_type == BackboneType.RESNET152:
            model = models.resnet152(weights=weights)
            feature_dim = model.fc.in_features  # 2048
            model.fc = nn.Identity()

        elif backbone_type == BackboneType.INCEPTION_V3:
            model = models.inception_v3(weights=weights, aux_logits=False)
            feature_dim = model.fc.in_features  # 2048
            model.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # Freeze classical backbone parameters
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        return model, feature_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass:
        Image -> Frozen Backbone A' -> Feature Vector -> Dressed Quantum Circuit B -> Logits
        """
        # Feature extraction through Classical Network A'
        features = self.backbone(x)
        if hasattr(features, 'logits'):
            features = features.logits

        # Quantum classification through Dressed Quantum Circuit B
        logits, q_expvals = self.dressed_quantum_net(features)

        return logits, q_expvals, features
