"""
Inference Module & Clinical Diagnostic API for Quantum Transfer Learning DR Detection
Paper: "Diabetic Retinopathy Detection Using Quantum Transfer Learning" (arXiv:2405.01734v1)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from quantum.config import DR_STAGE_LABELS, ModelConfig
from quantum.dataset import generate_synthetic_fundus_image, get_fundus_transforms
from quantum.models import QuantumTransferLearningDR

CLINICAL_DESCRIPTIONS = {
    0: "Normal retinal fundus with no signs of diabetic retinopathy. Recommendation: Routine annual screening.",
    1: "Mild Non-Proliferative DR: Microaneurysms present. Recommendation: Follow-up in 6 to 12 months with glycemic control.",
    2: "Moderate Non-Proliferative DR: Multiple microaneurysms, blot hemorrhages, and hard exudates. Recommendation: Retinal specialist evaluation within 3 to 6 months.",
    3: "Severe Non-Proliferative DR: Significant vascular abnormalities, cotton-wool ischemic spots, venous beading. Recommendation: Prompt ophthalmologist consultation within 2 to 4 weeks.",
    4: "Proliferative DR: Active neovascularization, vitreous/preretinal hemorrhage, high risk of vision loss. Recommendation: Urgent vitreoretinal referral for panretinal photocoagulation (PRP) or anti-VEGF therapy."
}


class QuantumDRPredictor:
    """
    High-level API for running Diabetic Retinopathy stage diagnosis using the Quantum Transfer Learning model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        device: Optional[str] = None,
        use_pennylane: bool = False
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_config = model_config or ModelConfig()
        self.use_pennylane = use_pennylane

        self.model = QuantumTransferLearningDR(
            config=self.model_config,
            use_pennylane=self.use_pennylane
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_fundus_transforms(
            image_size=(224, 224),
            is_training=False,
            apply_graham=True
        )

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor]
    ) -> Dict:
        """
        Runs quantum transfer learning prediction on an input retinal image.

        Args:
            image_input: File path, PIL Image, or preprocessed PyTorch Tensor.

        Returns:
            Dict containing predicted_stage, class_probabilities, quantum_state, and clinical_guidance.
        """
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
            tensor_input = self.transform(pil_image).unsqueeze(0).to(self.device)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            tensor_input = self.transform(pil_image).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 3:
                tensor_input = image_input.unsqueeze(0).to(self.device)
            else:
                tensor_input = image_input.to(self.device)
        else:
            raise TypeError("Unsupported image input type.")

        with torch.no_grad():
            logits, q_expvals, _ = self.model(tensor_input)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            q_vals = q_expvals.squeeze(0).cpu().numpy()

        pred_stage = int(np.argmax(probs))
        confidence = float(probs[pred_stage])

        stage_probabilities = {
            DR_STAGE_LABELS[i]: float(probs[i]) for i in range(len(probs))
        }

        quantum_pauli_z_expectations = {
            f"Qubit_{i} (<Z_{i}>)": float(q_vals[i]) for i in range(len(q_vals))
        }

        return {
            "predicted_stage": pred_stage,
            "stage_name": DR_STAGE_LABELS[pred_stage],
            "confidence": confidence,
            "stage_probabilities": stage_probabilities,
            "quantum_pauli_z_expectations": quantum_pauli_z_expectations,
            "clinical_diagnosis": CLINICAL_DESCRIPTIONS[pred_stage]
        }


def main():
    parser = argparse.ArgumentParser(description="Run Quantum Transfer Learning inference on a retina scan")
    parser.add_argument("--image", type=str, default=None, help="Path to retinal fundus image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--stage-demo", type=int, default=2, help="Generate synthetic fundus image of specified stage")
    args = parser.parse_args()

    predictor = QuantumDRPredictor(checkpoint_path=args.checkpoint)

    if args.image and Path(args.image).exists():
        print(f"🔍 Analyzing image: {args.image}")
        result = predictor.predict(args.image)
    else:
        print(f"✨ Generating and diagnosing sample Stage {args.stage_demo} fundus scan...")
        sample_img = generate_synthetic_fundus_image(stage=args.stage_demo)
        result = predictor.predict(sample_img)

    print("\n" + "=" * 60)
    print(f"🩺 DIAGNOSIS RESULT: Stage {result['predicted_stage']} - {result['stage_name']}")
    print(f"🎯 Model Confidence: {result['confidence'] * 100:.2f}%")
    print("-" * 60)
    print("📈 Stage Probabilities:")
    for stage, prob in result["stage_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {stage:<18}: {prob * 100:5.2f}% {bar}")
    print("-" * 60)
    print("⚛️  Quantum Pauli-Z Expectation Values (<Z_i> in [-1, 1]):")
    for qubit, val in result["quantum_pauli_z_expectations"].items():
        print(f"  {qubit}: {val:+.4f}")
    print("-" * 60)
    print(f"📋 Clinical Guidance:\n  {result['clinical_diagnosis']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
