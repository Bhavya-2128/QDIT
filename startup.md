# 🚀 QDIT: Startup & Execution Guide

Diabetic Retinopathy Detection using **Quantum Transfer Learning** (arXiv:[2405.01734v1](https://arxiv.org/abs/2405.01734)).

This guide provides end-to-end instructions to train on the complete 5-stage dataset (`train_images/`), evaluate model performance across stages 0 to 4, generate predictions on the `test_images/` folder, run single-image clinical diagnosis, and reproduce paper benchmarks.

---

## 📑 Table of Contents

1. [Dataset Structure (Stages 0 to 4)](#1-dataset-structure-stages-0-to-4)
2. [Environment & Setup](#2-environment--setup)
3. [Quick Verification (Unit Tests)](#3-quick-verification-unit-tests)
4. [Training on Whole Dataset (Stages 0 to 4)](#4-training-on-whole-dataset-stages-0-to-4)
5. [Evaluating & Testing on `test_images/`](#5-evaluating--testing-on-test_images)
6. [Single Image Diagnostic Inference](#6-single-image-diagnostic-inference)
7. [Reproducing Paper Experiments (Tables 4 & 5)](#7-reproducing-paper-experiments-tables-4--5)
8. [Kaggle & Cloud GPU Execution](#8-kaggle--cloud-gpu-execution)
9. [Codebase Architecture & File Reference](#9-codebase-architecture--file-reference)
10. [Troubleshooting & FAQs](#10-troubleshooting--faqs)

---

## 1. Dataset Structure (Stages 0 to 4)

The project uses the **APTOS 2019 Blindness Detection** dataset located in [`aptos2019-blindness-detection/`](aptos2019-blindness-detection):

```
aptos2019-blindness-detection/
├── train.csv                # 3,662 labeled images with id_code and diagnosis (0 - 4)
├── train_images/            # 3,662 retinal fundus PNG scans
├── test.csv                 # 1,928 test image IDs
├── test_images/             # 1,928 test fundus scans for model evaluation & submission
└── sample_submission.csv    # Submission format reference
```

### Diabetic Retinopathy Severity Grading
| Stage | Clinical Diagnosis | Description | Train Samples |
|:---:|:---|:---|:---:|
| **0** | **No DR** | Normal retina with no diabetic lesions | 1,805 |
| **1** | **Mild Non-Proliferative DR** | Microaneurysms only | 370 |
| **2** | **Moderate Non-Proliferative DR** | Microaneurysms, hemorrhages, hard exudates | 999 |
| **3** | **Severe Non-Proliferative DR** | Cotton-wool ischemic spots, venous beading | 193 |
| **4** | **Proliferative DR** | Neovascularization & preretinal hemorrhage | 295 |
| **Total** | | | **3,662** |

---

## 2. Environment & Setup

### Prerequisites
- Python 3.10+ (Tested with PyTorch 2.13 and Python 3.14)
- CUDA-compatible GPU (e.g., NVIDIA RTX / Tesla T4 / P100 / A100) or CPU

### Activating the Virtual Environment
```bash
# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Hardware Verification
```bash
python -c "import torch, pennylane as qml; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"} | PennyLane: {qml.__version__}')"
```

---

## 3. Quick Verification (Unit Tests)

Run the full automated test suite to verify PennyLane quantum circuits, classical backbones, Graham filters, training loops, and inference APIs:

```bash
python -m unittest quantum/test_quantum_model.py
```

**Expected output:**
```
Ran 9 tests in ~3.7s
OK
```

---

## 4. Training on Whole Dataset (Stages 0 to 4)

The training pipeline ([`quantum/train.py`](quantum/train.py)) trains the Classic-to-Quantum transfer learning model on all 3,662 fundus scans across stages 0 to 4 using Focal Loss / Label Smoothing and Adam optimizer.

### A. Full Training on Local GPU / CPU
```bash
# Train default model (ResNet-18 + 4-Qubit Hadamard/CNOT circuit, 30 epochs)
python quantum/train.py --dataset-dir aptos2019-blindness-detection --epochs 30 --batch-size 32 --lr 0.001
```

### B. Fast Training (5 Epochs)
```bash
python quantum/train.py --dataset-dir aptos2019-blindness-detection --epochs 5 --batch-size 32
```

### C. Advanced Quantum Gates & Backbones
```bash
# ResNet-50 backbone with RX embedding and CRX entangling gates
python quantum/train.py --backbone resnet50 --embedding rx --entangling crx --epochs 25 --batch-size 16

# Inception-V3 backbone with S-Phase embedding and CZ entanglers
python quantum/train.py --backbone inception_v3 --embedding s_phase --entangling cz --epochs 20 --batch-size 16
```

### Training Pipeline Workflow:
1. **Preprocessing**: Automatically applies Ben Graham's Gaussian blur filter ($\alpha=4, \beta=-4, \gamma=128$) and circular FOV cropping.
2. **Stratified Split**: Splits 3,662 images into 80% train (2,929 images) and 20% validation (733 images) preserving class balance across all 5 stages.
3. **Quantum Forward Pass**: Classical backbone extracts 512-dim features $\rightarrow$ Linear pre-layer reduces to 4 dimensions $\rightarrow$ 4-qubit parameterized quantum circuit executes $\rightarrow$ Pauli-$Z$ expectations $\langle Z_i \rangle \in [-1, 1]$ measured $\rightarrow$ Linear post-layer classifies into 5 DR stages.
4. **Automatic Evaluation**: Evaluates on validation set and immediately runs inference on all 1,928 images in `test_images/`.
5. **Artifacts Saved**:
   - Best Checkpoint: `quantum/quantum_dr_model.pt`
   - Learning Curves Plot: `quantum/training_curves.png`
   - Test Inferences: `quantum/submission.csv`

---

## 5. Evaluating & Testing on `test_images/`

Use [`quantum/evaluate.py`](quantum/evaluate.py) to evaluate a trained checkpoint on stages 0 to 4 and generate predictions on the `test_images/` folder:

### A. Standalone Evaluation & Prediction Generation
```bash
python quantum/evaluate.py --checkpoint quantum/quantum_dr_model.pt --dataset-dir aptos2019-blindness-detection --output-csv quantum/submission.csv
```

### Output Metrics Generated:
- Overall Accuracy (Eq. 12)
- Macro Precision (Eq. 13)
- Macro Recall / Sensitivity (Eq. 14)
- Macro Specificity
- Macro F1-Score (Eq. 15)
- Stage-by-Stage Performance Table
- 5x5 Multi-class Confusion Matrix
- `quantum/submission.csv` containing predicted stages (0 to 4) for all 1,928 test images

---

## 6. Single Image Diagnostic Inference

Use [`quantum/infer.py`](quantum/infer.py) to run clinical diagnostic inference on any retinal fundus image:

### A. Diagnosing a Real Fundus Scan from `train_images/` or `test_images/`
```bash
python quantum/infer.py --image aptos2019-blindness-detection/train_images/000c1434d8d7.png --checkpoint quantum/quantum_dr_model.pt
```

### B. Diagnosing a Synthetic Fundus Scan (Stages 0 to 4)
```bash
# Diagnose synthetic stage 3 (Severe DR) scan:
python quantum/infer.py --stage-demo 3
```

### C. Python Inference in Your Own Code
```python
from quantum.infer import QuantumDRPredictor

# 1. Initialize predictor
predictor = QuantumDRPredictor(checkpoint_path="quantum/quantum_dr_model.pt")

# 2. Run prediction on any image
result = predictor.predict("aptos2019-blindness-detection/train_images/000c1434d8d7.png")

# 3. Access clinical output
print(f"Predicted Stage: {result['predicted_stage']} - {result['stage_name']}")
print(f"Confidence: {result['confidence'] * 100:.2f}%")
print(f"Pauli-Z Expectations: {result['quantum_pauli_z_expectations']}")
print(f"Clinical Guidance: {result['clinical_diagnosis']}")
```

---

## 7. Reproducing Paper Experiments (Tables 4 & 5)

Reproduce the benchmark tables from the research paper (arXiv:2405.01734v1) using [`quantum/reproduce_paper_experiments.py`](quantum/reproduce_paper_experiments.py):

### A. Reproduce Table 4 (Backbone Architectures: ResNet-18/34/50/101/152 & Inception-V3)
```bash
python quantum/reproduce_paper_experiments.py --table 4 --epochs 5 --dataset-dir aptos2019-blindness-detection
```

### B. Reproduce Table 5 (Quantum Gate Presets: $H$, $S$, $S^\dagger$, $R_X$, $R_Y$, $\text{CNOT}$, $\text{CZ}$, $\text{SWAP}$, $\text{CR}_X$)
```bash
python quantum/reproduce_paper_experiments.py --table 5 --epochs 5 --dataset-dir aptos2019-blindness-detection
```

---

## 8. Kaggle & Cloud GPU Execution

To train on Kaggle using dataset `bhavyasanghavi2348/data-qdit`:

1. Open a new Kaggle Notebook.
2. Select **Accelerator**: **GPU T4 x2** or **P100**.
3. Run the following cells (or open [`kaggle_qdit_runner.ipynb`](kaggle_qdit_runner.ipynb)):

```python
# 1. Clone repo & install PennyLane
!git clone https://github.com/Bhavya-2128/QDIT.git
%cd QDIT
!pip install -q pennylane kagglehub

# 2. Download Kaggle dataset
import kagglehub
dataset_path = kagglehub.dataset_download("bhavyasanghavi2348/data-qdit")

# 3. Train on dataset and evaluate on test_images
!python quantum/train.py --dataset-dir {dataset_path} --epochs 30 --batch-size 32
```

---

## 9. Codebase Architecture & File Reference

```
QDIT/
├── startup.md                       # Complete startup, training & evaluation guide
├── README.md                        # Project summary
├── dataset_info.md                  # Comprehensive dataset sources & paper annotations
├── aptos2019-blindness-detection/   # Local APTOS 2019 dataset
│   ├── train.csv                    # 3,662 image IDs & diagnosis labels (0 - 4)
│   ├── train_images/                # 3,662 fundus image files
│   ├── test.csv                     # 1,928 test image IDs
│   └── test_images/                 # 1,928 test fundus scans
├── kaggle_qdit_runner.ipynb         # Turnkey notebook for Kaggle GPU
├── codes/                           # Exploratory CNN & quantum notebooks
├── paper/                           # Research papers & PDFs
└── quantum/                         # Core Quantum Transfer Learning Package
    ├── __init__.py                  # Package exports
    ├── config.py                    # Backbones, gate presets, and hyperparameters
    ├── circuits.py                  # PennyLane parameterized quantum circuits & gates
    ├── models.py                    # Classic-to-Quantum transfer learning architecture
    ├── dataset.py                   # Ben Graham filter, data loaders, APTOS dataset
    ├── train.py                     # Training loop, Focal Loss, StepLR, checkpointing
    ├── evaluate.py                  # Standalone evaluator & metrics calculation
    ├── infer.py                     # Single-image clinical diagnostic inference API
    ├── reproduce_paper_experiments.py # Benchmarks for Tables 4 & 5
    ├── test_quantum_model.py        # Automated test suite
    ├── training_curves.png          # Saved loss & accuracy plots
    ├── quantum_dr_model.pt          # Trained model weights checkpoint
    └── submission.csv               # Test set predictions for 1,928 images
```

---

## 10. Troubleshooting & FAQs

### Q: How to resolve CUDA out of memory errors?
**A:** Reduce `--batch-size` to `16` or `8`:
```bash
python quantum/train.py --batch-size 16
```

### Q: Where are the predictions for `test_images/` saved?
**A:** Predictions are automatically saved to `quantum/submission.csv` with columns `id_code` and `diagnosis` (0 to 4).

### Q: What is the significance of Pauli-$Z$ values?
**A:** Each qubit produces an expectation value $\langle Z_i \rangle = \langle \psi | \sigma_z^{(i)} | \psi \rangle \in [-1, 1]$. These 4 continuous quantum observables encode the macroscopic visual features from the classical backbone into a compact quantum state representation for final stage classification.
