"""
Unit & Integration Test Suite for Quantum Transfer Learning Architecture
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)
"""

import sys
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from quantum.circuits import create_quantum_circuit_qnode, draw_circuit
from quantum.config import (
    BackboneType,
    EmbeddingGateType,
    EntanglingGateType,
    ModelConfig,
    QuantumCircuitConfig,
    TrainingConfig,
)
from quantum.dataset import (
    apply_graham_gaussian_filter,
    create_synthetic_fundus_dataset,
    generate_synthetic_fundus_image,
    get_dataloaders,
    get_fundus_transforms,
)
from quantum.evaluate import compute_metrics
from quantum.models import DressedQuantumNet, FastTorchQuantumCircuit, QuantumTransferLearningDR
from quantum.train import train_model


class TestQuantumCircuits(unittest.TestCase):
    """Tests parameterized quantum circuit construction, gate application, and measurements."""

    def test_pennylane_qnode_execution(self):
        config = QuantumCircuitConfig(
            n_qubits=4,
            q_depth=2,
            embedding_gate=EmbeddingGateType.HADAMARD,
            entangling_gate=EntanglingGateType.CNOT
        )
        qnode, dev = create_quantum_circuit_qnode(config)

        inputs = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
        weights = torch.zeros((2, 4), dtype=torch.float32)

        expvals = qnode(inputs, weights)
        self.assertEqual(len(expvals), 4)
        for val in expvals:
            # Expectation values of Pauli-Z must be bounded in [-1.0, 1.0]
            val_float = float(val)
            self.assertGreaterEqual(val_float, -1.0 - 1e-5)
            self.assertLessEqual(val_float, 1.0 + 1e-5)

    def test_all_embedding_and_entangling_gate_combinations(self):
        for emb in [EmbeddingGateType.HADAMARD, EmbeddingGateType.S_PHASE, EmbeddingGateType.S_DAGGER, EmbeddingGateType.RX]:
            for ent in [EntanglingGateType.CNOT, EntanglingGateType.CZ, EntanglingGateType.SWAP, EntanglingGateType.CRX]:
                config = QuantumCircuitConfig(
                    n_qubits=4,
                    q_depth=2,
                    embedding_gate=emb,
                    entangling_gate=ent
                )
                fast_sim = FastTorchQuantumCircuit(config)
                sample_inputs = torch.randn(2, 4)
                out = fast_sim(sample_inputs)
                self.assertEqual(out.shape, (2, 4))
                # Pauli-Z expectation values must lie in [-1, 1]
                self.assertTrue(torch.all(out >= -1.0001))
                self.assertTrue(torch.all(out <= 1.0001))

    def test_circuit_drawing(self):
        circuit_str = draw_circuit()
        self.assertIsInstance(circuit_str, str)
        self.assertTrue(len(circuit_str) > 0)


class TestQuantumModels(unittest.TestCase):
    """Tests DressedQuantumNet, FastTorchQuantumCircuit, and hybrid CQ architecture."""

    def test_dressed_quantum_net_forward_and_backward(self):
        batch_size = 4
        in_features = 512
        num_classes = 5

        dressed_net = DressedQuantumNet(
            in_features=in_features,
            num_classes=num_classes,
            circuit_config=QuantumCircuitConfig(n_qubits=4, q_depth=2),
            use_pennylane=False
        )

        dummy_features = torch.randn(batch_size, in_features, requires_grad=True)
        logits, q_expvals = dressed_net(dummy_features)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(q_expvals.shape, (batch_size, 4))

        # Test backpropagation
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(dummy_features.grad)
        self.assertTrue(any(p.grad is not None for p in dressed_net.pre_net.parameters()))
        self.assertTrue(any(p.grad is not None for p in dressed_net.post_net.parameters()))

    def test_pennylane_dressed_net_forward_and_backward(self):
        batch_size = 2
        in_features = 128
        num_classes = 5

        dressed_net = DressedQuantumNet(
            in_features=in_features,
            num_classes=num_classes,
            circuit_config=QuantumCircuitConfig(n_qubits=4, q_depth=2),
            use_pennylane=True
        )

        dummy_features = torch.randn(batch_size, in_features, requires_grad=True)
        logits, q_expvals = dressed_net(dummy_features)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(q_expvals.shape, (batch_size, 4))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(dressed_net.q_weights.grad)

    def test_hybrid_cq_model_backbone_freezing(self):
        # 1. Test 100% frozen backbone
        frozen_model = QuantumTransferLearningDR(
            config=ModelConfig(
                backbone=BackboneType.RESNET18,
                pretrained=False,
                freeze_backbone=True,
                unfreeze_last_n_layers=0
            )
        )
        for param in frozen_model.backbone.parameters():
            self.assertFalse(param.requires_grad)

        # 2. Test selective layer4 fine-tuning
        model = QuantumTransferLearningDR(
            config=ModelConfig(
                backbone=BackboneType.RESNET18,
                pretrained=False,
                freeze_backbone=True,
                unfreeze_last_n_layers=1
            )
        )
        layer4_trainable = any(p.requires_grad for name, p in model.backbone.named_parameters() if "layer4" in name)
        early_conv_frozen = all(not p.requires_grad for name, p in model.backbone.named_parameters() if name.startswith("conv1") or name.startswith("layer1"))
        self.assertTrue(layer4_trainable)
        self.assertTrue(early_conv_frozen)

        # Check dressed quantum net weights are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertGreater(len(trainable_params), 0)

        # End-to-end forward pass
        dummy_images = torch.randn(2, 3, 224, 224)
        logits, q_expvals, features = model(dummy_images)

        self.assertEqual(logits.shape, (2, 5))
        self.assertEqual(q_expvals.shape, (2, 4))
        self.assertEqual(features.shape, (2, 512))


class TestDatasetAndPreprocessing(unittest.TestCase):
    """Tests retinal image preprocessing and synthetic dataset generation."""

    def test_graham_gaussian_filter(self):
        sample_img = Image.new("RGB", (100, 100), (150, 50, 20))
        filtered = apply_graham_gaussian_filter(sample_img)
        self.assertEqual(filtered.size, (100, 100))
        self.assertIsInstance(filtered, Image.Image)

    def test_synthetic_dataset_and_dataloaders(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_synthetic_fundus_dataset(
                output_dir=tmp_dir,
                samples_per_stage=3,
                image_size=(64, 64)
            )

            train_loader, val_loader = get_dataloaders(
                dataset_dir=dataset_dir,
                batch_size=4,
                image_size=(64, 64),
                num_workers=0
            )

            batch_x, batch_y = next(iter(train_loader))
            self.assertEqual(batch_x.ndim, 4)
            self.assertEqual(batch_x.shape[1], 3)
            self.assertEqual(len(batch_y), batch_x.shape[0])


class TestMetricsAndEvaluation(unittest.TestCase):
    """Tests evaluation metrics against Equations 12-15."""

    def test_compute_metrics_accuracy_precision_recall_f1(self):
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 0])  # 9/10 correct

        metrics = compute_metrics(y_true, y_pred, num_classes=5)

        self.assertAlmostEqual(metrics["accuracy"], 0.90)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("specificity_macro", metrics)
        self.assertIn("f1_score_macro", metrics)
        self.assertEqual(metrics["confusion_matrix"].shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
