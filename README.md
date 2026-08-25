# QDIT: Quantum Transfer Learning for Diabetic Retinopathy Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-blue.svg)](https://pennylane.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.01734-b31b1b.svg)](https://arxiv.org/abs/2405.01734)

A complete implementation of the hybrid quantum-classical transfer learning algorithm for 5-stage Diabetic Retinopathy detection based on the research paper **"Diabetic Retinopathy Detection Using Quantum Transfer Learning"** (`quantum/2405.01734v1.pdf`).

---

## ⚡ Quick Start on Kaggle

Run directly on Kaggle with dataset [`bhavyasanghavi2348/data-qdit`](https://www.kaggle.com/datasets/bhavyasanghavi2348/data-qdit):

```python
# 1. In a Kaggle GPU notebook:
!git clone https://github.com/Bhavya-2128/QDIT.git
%cd QDIT
!pip install -q pennylane kagglehub

# 2. Train on the Kaggle dataset:
!python run_on_kaggle.py
```

Or open [`kaggle_qdit_runner.ipynb`](kaggle_qdit_runner.ipynb) in Kaggle!

📖 **For detailed step-by-step instructions with screenshots and troubleshooting, see the [Kaggle Execution Guide](KAGGLE_EXECUTION_GUIDE.md).**

---

## 🔬 Local Quick Start

```bash
# 1. Run unit test suite
python -m unittest quantum/test_quantum_model.py

# 2. Train with default configuration (ResNet18 + Hadamard/CNOT)
python quantum/train.py --epochs 30 --batch-size 16

# 3. Predict on a fundus image
python quantum/infer.py --stage-demo 2

# 4. Reproduce Paper Benchmark Tables
python quantum/reproduce_paper_experiments.py --table 4
```

For complete documentation, mathematical formulations, and gate specifications, see [`quantum/README.md`](quantum/README.md).