"""
Variational Quantum Circuits for Quantum Transfer Learning
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)

Implements parameterized quantum circuits, embedding layers (Hadamard, S, S†, RX, RY),
entangling patterns (CNOT, CZ, SWAP, CRX, CRY, CRZ), and Pauli-Z measurements.
"""

from typing import Callable, List, Optional
import numpy as np
import pennylane as qml
import torch

from quantum.config import EmbeddingGateType, EntanglingGateType, QuantumCircuitConfig


def apply_embedding_layer(
    inputs: torch.Tensor,
    n_qubits: int,
    embedding_type: EmbeddingGateType = EmbeddingGateType.HADAMARD
) -> None:
    """
    Applies the quantum embedding layer to encode classical features into quantum states.

    Args:
        inputs: 1D tensor of classical features scaled to [-pi/2, pi/2] or [0, 2pi].
        n_qubits: Number of qubits.
        embedding_type: Type of embedding gate from paper (Hadamard, S-Phase, S†, RX, RY).
    """
    if embedding_type == EmbeddingGateType.HADAMARD:
        # Default paper architecture: H gate creates superposition, RY rotates by input feature
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

    elif embedding_type == EmbeddingGateType.S_PHASE:
        # Table 5: S-Phase gate followed by Hadamard and RY
        for i in range(n_qubits):
            qml.S(wires=i)
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

    elif embedding_type == EmbeddingGateType.S_DAGGER:
        # Table 5: S† (conjugate transpose / adjoint S) followed by Hadamard and RY
        for i in range(n_qubits):
            qml.adjoint(qml.S)(wires=i)
            qml.Hadamard(wires=i)
            qml.RY(inputs[i], wires=i)

    elif embedding_type == EmbeddingGateType.RX:
        # Table 5: Direct RX rotation embedding
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

    elif embedding_type == EmbeddingGateType.RY:
        # Direct RY rotation embedding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

    else:
        raise ValueError(f"Unsupported embedding gate type: {embedding_type}")


def apply_entangling_layer(
    n_qubits: int,
    entangling_type: EntanglingGateType = EntanglingGateType.CNOT,
    weights_slice: Optional[torch.Tensor] = None
) -> None:
    """
    Applies entangling 2-qubit gates across adjacent qubits in a circular/ring topology.

    Args:
        n_qubits: Number of qubits in the circuit.
        entangling_type: Gate type (CNOT, CZ, SWAP, CRX, CRY, CRZ).
        weights_slice: Optional tensor of weights for parameterized controlled rotations.
    """
    for i in range(n_qubits):
        control = i
        target = (i + 1) % n_qubits

        if entangling_type == EntanglingGateType.CNOT:
            qml.CNOT(wires=[control, target])

        elif entangling_type == EntanglingGateType.CZ:
            qml.CZ(wires=[control, target])

        elif entangling_type == EntanglingGateType.SWAP:
            qml.SWAP(wires=[control, target])

        elif entangling_type == EntanglingGateType.CRX:
            angle = weights_slice[i] if weights_slice is not None else 0.0
            qml.CRX(angle, wires=[control, target])

        elif entangling_type == EntanglingGateType.CRY:
            angle = weights_slice[i] if weights_slice is not None else 0.0
            qml.CRY(angle, wires=[control, target])

        elif entangling_type == EntanglingGateType.CRZ:
            angle = weights_slice[i] if weights_slice is not None else 0.0
            qml.CRZ(angle, wires=[control, target])

        else:
            raise ValueError(f"Unsupported entangling gate type: {entangling_type}")


def apply_variational_layer(
    weights: torch.Tensor,
    n_qubits: int,
    q_depth: int,
    entangling_type: EntanglingGateType = EntanglingGateType.CNOT
) -> None:
    """
    Applies q_depth repeated variational layers consisting of entangling gates and trainable RY rotations.

    Args:
        weights: Tensor of shape (q_depth, n_qubits) containing trainable rotation parameters.
        n_qubits: Number of qubits.
        q_depth: Depth (number of repeated layers) of the circuit.
        entangling_type: Entangling gate type.
    """
    for layer in range(q_depth):
        # 1. Multi-qubit entangling layer
        apply_entangling_layer(
            n_qubits=n_qubits,
            entangling_type=entangling_type,
            weights_slice=weights[layer]
        )

        # 2. Trainable single-qubit rotation layer around Y-axis (or parameterized rotation)
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)


def create_quantum_circuit_qnode(config: QuantumCircuitConfig) -> Tuple[Callable, qml.Device]:
    """
    Creates and returns a PennyLane QNode and its associated simulation device.

    Args:
        config: QuantumCircuitConfig with n_qubits, q_depth, gate types, and backend device.

    Returns:
        Tuple of (qnode_function, dev).
    """
    dev = qml.device(config.device_name, wires=config.n_qubits, shots=config.shots)

    @qml.qnode(dev, interface="torch", diff_method=config.diff_method)
    def circuit_qnode(inputs: torch.Tensor, weights: torch.Tensor):
        """
        Executes the quantum circuit for a single input sample.

        Args:
            inputs: Tensor of shape (n_qubits,) representing preprocessed classical features.
            weights: Tensor of shape (q_depth, n_qubits) representing variational angles.

        Returns:
            List of Pauli-Z expectation values for each qubit.
        """
        # Step 1: Quantum state initialization and input embedding
        apply_embedding_layer(
            inputs=inputs,
            n_qubits=config.n_qubits,
            embedding_type=config.embedding_gate
        )

        # Step 2: Variational processing with entangling & rotation layers
        apply_variational_layer(
            weights=weights,
            n_qubits=config.n_qubits,
            q_depth=config.q_depth,
            entangling_type=config.entangling_gate
        )

        # Step 3: Measurement layer - Expectation value of Pauli-Z operator on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(config.n_qubits)]

    return circuit_qnode, dev


def draw_circuit(config: Optional[QuantumCircuitConfig] = None) -> str:
    """
    Returns an ASCII/Unicode diagram of the quantum circuit.
    """
    if config is None:
        config = QuantumCircuitConfig()

    qnode, _ = create_quantum_circuit_qnode(config)
    sample_inputs = torch.zeros(config.n_qubits)
    sample_weights = torch.zeros((config.q_depth, config.n_qubits))

    return qml.draw(qnode)(sample_inputs, sample_weights)
