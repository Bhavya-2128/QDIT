# 🚀 Step-by-Step Guide: Running QDIT on Kaggle with GPU Acceleration

This comprehensive guide walks you through setting up, training, and running inference for the **Quantum Transfer Learning for Diabetic Retinopathy (QDIT)** model directly in a **Kaggle Notebook** using free NVIDIA GPU accelerators (Tesla T4 x2 or P100) and your dataset [`bhavyasanghavi2348/data-qdit`](https://www.kaggle.com/datasets/bhavyasanghavi2348/data-qdit).

---

## 📑 Table of Contents
1. [Prerequisites & Kaggle Setup](#1-prerequisites--kaggle-setup)
2. [Method 1: One-Click Execution (Recommended)](#2-method-1-one-click-execution-recommended)
3. [Method 2: Interactive Step-by-Step Notebook](#3-method-2-interactive-step-by-step-notebook)
4. [Method 3: Custom Command-Line Training](#4-method-3-custom-command-line-training)
5. [How to Download Trained Model Weights](#5-how-to-download-trained-model-weights)
6. [Running Inference on New Retinal Images](#6-running-inference-on-new-retinal-images)
7. [Troubleshooting & FAQs](#7-troubleshooting--faqs)

---

## 1. Prerequisites & Kaggle Setup

### Step 1.1: Create a New Kaggle Notebook
1. Go to [Kaggle.com](https://www.kaggle.com/) and log into your account.
2. Click the **+ Create** button in the left sidebar and select **New Notebook**.

### Step 1.2: Enable GPU and Internet Access (Critical ⚠️)
On the right-hand panel of your Kaggle notebook under **Notebook options / Settings**:

1. **Accelerator**: Click the dropdown and select **GPU T4 x2** (or **GPU P100**).
2. **Internet**: Toggle the switch to **Internet On** (required to clone GitHub repo and download pre-trained weights).

---

## 2. Method 1: One-Click Execution (Recommended)

In a new code cell in your Kaggle notebook, paste and run the following code:

```python
# -------------------------------------------------------------
# 1. Clone the QDIT repository from GitHub
# -------------------------------------------------------------
!git clone https://github.com/Bhavya-2128/QDIT.git
%cd QDIT

# -------------------------------------------------------------
# 2. Install PennyLane & Kagglehub dependencies
# -------------------------------------------------------------
!pip install -q pennylane kagglehub torchvision scikit-learn pandas matplotlib pillow

# -------------------------------------------------------------
# 3. Option A: Train Single Selected Configuration
# -------------------------------------------------------------
from run_on_kaggle import run_kaggle_experiment

trained_model, history, metrics = run_kaggle_experiment(
    dataset_slug="bhavyasanghavi2348/data-qdit",
    backbone="resnet18",            # Options: resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3
    embedding_gate="hadamard",      # Options: hadamard, s_phase, s_dagger, rx, ry
    entangling_gate="cnot",         # Options: cnot, cz, swap, crx, cry, crz
    epochs=30,                      # Training epochs
    batch_size=32,                  # Fast GPU batch size
    learning_rate=1e-3,             # Initial learning rate
    fine_tune_backbone=True,        # Fine-tune layer4 for high DR lesion sensitivity
    loss_type="focal",              # Focal loss to maximize F1-score
    save_path="/kaggle/working/quantum_dr_model.pt"
)

# -------------------------------------------------------------
# 4. Option B: Run ALL Paper Combinations (Table 4 & Table 5) Sequentially
# -------------------------------------------------------------
# from run_on_kaggle import run_all_paper_combinations
#
# summary_table = run_all_paper_combinations(
#     dataset_slug="bhavyasanghavi2348/data-qdit",
#     experiment_type="all",        # "table4" (6 Backbones), "table5" (11 Gate Combos), or "all"
#     epochs=15,                    # Epochs per model
#     batch_size=32,
#     fine_tune_backbone=True,
#     loss_type="focal"
# )
```

---

## 3. Method 2: Interactive Step-by-Step Notebook

If you prefer an interactive notebook with visual loss curves, confusion matrices, and circuit diagrams:

### Cell 1: Setup & Clone
```python
!git clone https://github.com/Bhavya-2128/QDIT.git
%cd QDIT
!pip install -q pennylane kagglehub torchvision scikit-learn pandas matplotlib pillow
```

### Cell 2: Verify GPU & Download Dataset
```python
import kagglehub
import torch

print("🚀 CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")

# Download dataset
dataset_path = kagglehub.dataset_download("bhavyasanghavi2348/data-qdit")
print("📁 Dataset location:", dataset_path)
```

### Cell 3: Visualize the 4-Qubit Variational Quantum Circuit
```python
from quantum.circuits import draw_circuit
from quantum.config import QuantumCircuitConfig, EmbeddingGateType, EntanglingGateType

circuit_cfg = QuantumCircuitConfig(
    n_qubits=4,
    q_depth=4,
    embedding_gate=EmbeddingGateType.HADAMARD,
    entangling_gate=EntanglingGateType.CNOT
)
print("⚛️  4-Qubit Variational Quantum Circuit Architecture:")
print(draw_circuit(circuit_cfg))
```

### Cell 4: Train Hybrid CQ Model
```python
from quantum.config import ModelConfig, TrainingConfig, BackboneType
from quantum.dataset import get_dataloaders
from quantum.models import QuantumTransferLearningDR
from quantum.train import train_model
from quantum.evaluate import format_metrics_table

# Configure Model
model_cfg = ModelConfig(
    backbone=BackboneType.RESNET18,
    quantum_circuit=circuit_cfg
)

train_cfg = TrainingConfig(
    batch_size=16,
    epochs=30,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Load fundus images with Ben Graham Gaussian filter
train_loader, val_loader = get_dataloaders(
    dataset_dir=dataset_path,
    batch_size=train_cfg.batch_size,
    image_size=(224, 224),
    apply_graham=True
)

# Initialize and train
model = QuantumTransferLearningDR(config=model_cfg)
trained_model, history, best_metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=train_cfg,
    save_path="/kaggle/working/quantum_dr_model.pt"
)

print(format_metrics_table(best_metrics))
```

### Cell 5: Plot Training Loss & Accuracy Curves
```python
import matplotlib.pyplot as plt

epochs_range = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
plt.plot(epochs_range, history["val_loss"], 'r-s', label='Val Loss')
plt.title('Quantum Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Accuracy')
plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], 'g-s', label='Val Accuracy')
plt.title('Classification Accuracy (%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 4. Method 3: Custom Command-Line Training

You can customize training directly using command-line arguments:

```bash
# Train ResNet-50 backbone with RX embedding and CRX entangling gates (Paper Table 5)
!python quantum/train.py \
    --dataset-kaggle bhavyasanghavi2348/data-qdit \
    --backbone resnet50 \
    --embedding rx \
    --entangling crx \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --save-path /kaggle/working/quantum_dr_resnet50.pt
```

---

## 5. How to Download Trained Model Weights

After training finishes, the best model weights are saved to `/kaggle/working/quantum_dr_model.pt`.

To download the file:
1. In the right-hand panel of Kaggle, locate the **Output** section.
2. Click the `/kaggle/working/` folder.
3. Find `quantum_dr_model.pt`, click the three vertical dots (`⋮`), and select **Download**.

---

## 6. Running Inference on New Retinal Images

To test the trained model on an image:

```python
from quantum.infer import QuantumDRPredictor

# Initialize predictor with saved checkpoint
predictor = QuantumDRPredictor(checkpoint_path="/kaggle/working/quantum_dr_model.pt")

# Pass image path (or PIL Image)
result = predictor.predict("path/to/retina_image.png")

print(f"🩺 Stage: {result['predicted_stage']} - {result['stage_name']}")
print(f"🎯 Model Confidence: {result['confidence'] * 100:.2f}%")
print("⚛️  Quantum Pauli-Z Expectation Values (<Z_i>):", result["quantum_pauli_z_expectations"])
print("📋 Clinical Guidance:", result["clinical_diagnosis"])
```

---

## 7. Troubleshooting & FAQs

| Issue | Solution |
|---|---|
| **`CUDA out of memory`** | Reduce batch size to `8` or `16` using `--batch-size 8`. |
| **`Permission denied / Connection error downloading weights`** | Ensure the **Internet** switch is toggled to **ON** in the right-side notebook settings. |
| **`ModuleNotFoundError: No module named 'pennylane'`** | Run `!pip install -q pennylane kagglehub` at the beginning of the notebook. |
| **`FileNotFoundError: No images found`** | Ensure `bhavyasanghavi2348/data-qdit` is specified properly, or attach the dataset directly via **+ Add Input** in the top right corner. |
