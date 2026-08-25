# ⚛️ Quantum Transfer Learning for Diabetic Retinopathy Detection (arXiv:2405.01734v1)

A complete, production-grade PyTorch and PennyLane implementation of the quantum machine learning algorithm described in the research paper:

> **"Diabetic Retinopathy Detection Using Quantum Transfer Learning"**  
> *Preprint / arXiv:2405.01734v1*  
> PDF available at: [`quantum/2405.01734v1.pdf`](file:///home/bhavya_2128/QDIT/quantum/2405.01734v1.pdf)

---

## 📑 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Mathematical Formulations](#-mathematical-formulations)
3. [Project Structure](#-project-structure)
4. [Installation & Requirements](#-installation--requirements)
5. [Running on Kaggle](#-running-on-kaggle)
6. [Local & Server Execution](#-local--server-execution)
7. [Paper Experiments Reproduction (Tables 4 & 5)](#-paper-experiments-reproduction)
8. [API & Inference Reference](#-api--inference-reference)

---

## 🧠 Architecture Overview

The system uses a **Classic-to-Quantum (CQ) Transfer Learning** paradigm to classify retina fundus images into 5 severity stages:
- `0 - No DR`
- `1 - Mild DR`
- `2 - Moderate DR`
- `3 - Severe DR`
- `4 - Proliferative DR`

```
                                  [ Input Fundus Image (224x224x3) ]
                                                  │
                                                  ▼
                               [ Gaussian / Graham Preprocessing ]
                                                  │
                                                  ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │    Classical Backbone (Frozen weights, e.g. ResNet18)    │
                     │    Maps image -> 512 / 2048 abstract feature vector      │
                     └────────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │            DRESSED QUANTUM CIRCUIT (Trainable)           │
                     │                                                          │
                     │ 1. Linear Pre-layer: (512 -> 4)                          │
                     │ 2. Tanh Activation & pi/2 Scaling                        │
                     │ 3. Quantum Circuit (4 Qubits):                           │
                     │    - Embedding: H / S / S† / RX / RY                     │
                     │    - Variational Layers: Entanglers + RY(θ)              │
                     │      (CNOT / CZ / SWAP / CRX / CRY / CRZ)                │
                     │    - Measurement: PauliZ Expectation <Z_i> (4 dims)      │
                     │ 4. Linear Post-layer: (4 -> 5 DR Classes)                │
                     └────────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                           [ Class Probabilities & Severity Stage (0 - 4) ]
```

---

## 📐 Mathematical Formulations

### 1. Quantum Gates & Operations

- **Hadamard Gate ($H$)**:
  $$H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1 \\ 1 & -1\end{bmatrix}$$

- **Phase Gate ($S$) & Conjugate Transpose ($S^\dagger$)**:
  $$S = \begin{bmatrix}1 & 0 \\ 0 & i\end{bmatrix}, \quad S^\dagger = \begin{bmatrix}1 & 0 \\ 0 & -i\end{bmatrix}$$

- **Rotation Gates ($R_x, R_y$)**:
  $$R_x(\theta) = \begin{bmatrix}\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{bmatrix}, \quad R_y(\theta) = \begin{bmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{bmatrix}$$

- **Entangling Gates (CNOT, CZ, SWAP, Controlled Rotations)**:
  $$\text{CNOT} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0\end{bmatrix}, \quad \text{CZ} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1\end{bmatrix}, \quad \text{SWAP} = \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}$$

- **Measurement Layer**:
  Local expectation values of Pauli-Z operator on each qubit:
  $$\langle Z_i \rangle = \langle \psi | \sigma_z^{(i)} | \psi \rangle \in [-1, 1]$$

### 2. Performance Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \quad (\text{Eq. 12})$$
$$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Eq. 13})$$
$$\text{Recall / Sensitivity} = \frac{TP}{TP + FN} \quad (\text{Eq. 14})$$
$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad (\text{Eq. 15})$$

---

## 📁 Project Structure

```
QDIT/
├── kaggle_qdit_runner.ipynb         # Turnkey Jupyter notebook for Kaggle execution
├── run_on_kaggle.py                 # Automated script for training on Kaggle environment
├── quantum/
│   ├── __init__.py                  # Package exports
│   ├── 2405.01734v1.pdf             # Original research paper
│   ├── config.py                    # Configurations, Backbones, and Gate presets
│   ├── circuits.py                  # PennyLane parameterized quantum circuits
│   ├── models.py                    # DressedQuantumNet, FastTorchQuantumCircuit, Hybrid CQ model
│   ├── dataset.py                   # Ben Graham filter, Kaggle/APTOS loader, synthetic generator
│   ├── train.py                     # Adam optimizer, StepLR, CrossEntropyLoss training pipeline
│   ├── evaluate.py                  # Multi-class accuracy, precision, recall, specificity, F1
│   ├── infer.py                     # Single-image clinical diagnostic inference API
│   ├── reproduce_paper_experiments.py # Benchmarks for Tables 4 and 5
│   ├── test_quantum_model.py        # Comprehensive unittest suite
│   └── README.md                    # Detailed documentation
└── README.md                        # Root overview
```

---

## 💻 Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/Bhavya-2128/QDIT.git
cd QDIT

# Create environment and install dependencies
pip install torch torchvision pennylane kagglehub scikit-learn pandas matplotlib pillow
```

---

## ☁️ Running on Kaggle

You can run this project directly on Kaggle with access to your dataset `bhavyasanghavi2348/data-qdit`:

### Option A: Via Kaggle Notebook (1-Click)
1. Open a new Kaggle Notebook (enable **GPU T4 x2** or **P100** accelerator).
2. Upload and open [`kaggle_qdit_runner.ipynb`](file:///home/bhavya_2128/QDIT/kaggle_qdit_runner.ipynb) or paste the cells:
```python
!git clone https://github.com/Bhavya-2128/QDIT.git
%cd QDIT
!pip install -q pennylane kagglehub

import kagglehub
path = kagglehub.dataset_download("bhavyasanghavi2348/data-qdit")

from run_on_kaggle import run_kaggle_experiment
run_kaggle_experiment(dataset_slug="bhavyasanghavi2348/data-qdit", epochs=30)
```

### Option B: Via Command Line in Kaggle
```bash
python run_on_kaggle.py
```

---

## 🚀 Local & Server Execution

### 1. Training
```bash
# Train default model (ResNet18 + Hadamard & CNOT, 4 Qubits)
python quantum/train.py --epochs 30 --batch-size 16 --lr 0.001

# Train with Kaggle dataset directly downloaded via kagglehub
python quantum/train.py --dataset-kaggle bhavyasanghavi2348/data-qdit --epochs 30

# Train with different quantum gates and backbones
python quantum/train.py --backbone resnet50 --embedding rx --entangling crx --epochs 25
```

### 2. Single Image Inference & Diagnosis
```bash
# Run on real image
python quantum/infer.py --image path/to/retina.png --checkpoint quantum_dr_model.pt

# Run demonstration on synthetic stage 3 fundus scan
python quantum/infer.py --stage-demo 3
```

### 3. Unit Tests
```bash
python -m unittest quantum/test_quantum_model.py
```

---

## 📊 Paper Experiments Reproduction

### Table 4: Backbone Architecture Comparison (Hadamard & CNOT)
| Backbone Model | Paper Reported Acc (%) | Paper Reported F1 (%) |
|---|---|---|
| **ResNet-18** | 97.6% | 97.9% |
| **ResNet-34** | 97.6% | 97.9% |
| **ResNet-50** | 97.6% | 97.9% |
| **ResNet-101** | 97.8% | 98.0% |
| **ResNet-152** | 98.2% | 98.3% |
| **Inception-V3** | 98.5% | 98.4% |

### Table 5: Quantum Gate Variations (ResNet-18 Backbone)
| Quantum Circuit Configuration | Paper Reported Acc (%) | Paper Reported F1 (%) |
|---|---|---|
| $S\text{(Phase)} + H + \text{CNOT}$ | 92.1% | 92.4% |
| $S^\dagger\text{(Dagger)} + H + \text{CNOT}$ | 92.3% | 92.6% |
| $R_x + \text{CNOT}$ | 97.5% | 97.9% |
| $H + \text{CZ}$ | 95.2% | 95.4% |
| $H + \text{SWAP}$ | 94.8% | 94.3% |
| $H + \text{CR}_x$ | 97.8% | 97.9% |
| $R_x + \text{CR}_x$ | 98.1% | 98.4% |

To run the automated benchmarks reproducing Tables 4 & 5:
```bash
python quantum/reproduce_paper_experiments.py --table 4 --epochs 5
python quantum/reproduce_paper_experiments.py --table 5 --epochs 5
```
