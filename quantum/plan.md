# Implementation Plan: Quantum Transfer Learning for DR Detection

Based on the paper "Diabetic Retinopathy Detection Using Quantum Transfer Learning", I will implement the hybrid classical-quantum models in the `quantum` directory using PyTorch and PennyLane.

## 1. Directory Structure and Files
I will create the following files in the `quantum/` directory:
*   **`dataset.py`**: Handles loading the dataset from `train&test/` (which contains `0, 1, 2, 3, 4` for train, and `test/0, 1, 2, 3, 4` for testing). Includes resizing (224x224), center cropping, and normalization.
*   **`model.py`**: Defines the `HybridQuantumResNet18` class and the `DressedQuantumNet`. The quantum circuit will be configurable to support all 7 gate variations specified in the paper.
*   **`train.py`**: Contains the training and validation loops, Adam optimizer setup, learning rate scheduler, and cross-entropy loss function.
*   **`main.py`**: A script to initialize and train any of the 7 models easily via command-line arguments.

## 2. Model Architecture Details

### Classical Feature Extractor
*   We will use `torchvision.models.resnet18(pretrained=True)`.
*   We will freeze all classical layers by setting `requires_grad = False`.
*   We will replace the final fully-connected layer (`fc`) with our custom `DressedQuantumNet`.

### Dressed Quantum Network
This module will consist of:
1.  **Pre-processing:** `nn.Linear(512, 4)` -> `torch.tanh` -> scaling by $\pi/2$.
2.  **Quantum Circuit (PennyLane):**
    *   **4 Qubits** initialized.
    *   **Embedding Layer:** Configurable to use different combinations (e.g., `Hadamard`, `S` followed by `H`, `RX`, etc.) to encode the 4 classical inputs.
    *   **Variational Layer:** Iterated `q_depth` times. Contains an entangling block (e.g., `CNOT`, `CZ`, `SWAP`, `CRX`) and a trainable rotation block (`RY` gates).
    *   **Measurement:** Returns the expectation value of `PauliZ` for each of the 4 qubits.
3.  **Post-processing:** `nn.Linear(4, 5)` to map the quantum output to the 5 DR classes.

## 3. The 7 Model Variations
We will implement the following configurations for the Quantum Circuit as per Table 5 of the paper:
1.  **S(Phase) - Hadamard & CNOT**: Embedding uses `S` then `H`, entangling uses `CNOT`.
2.  **S†(Dagger) - Hadamard & CNOT**: Embedding uses `S_dagger` then `H`, entangling uses `CNOT`.
3.  **Hadamard & SWAP**: Embedding uses `H`, entangling uses `SWAP`.
4.  **Hadamard & CZ**: Embedding uses `H`, entangling uses `CZ`.
5.  **RX & CNOT**: Embedding uses `RX`, entangling uses `CNOT`.
6.  **Hadamard & CRX**: Embedding uses `H`, entangling uses `CRX`.
7.  **RX & CRX**: Embedding uses `RX`, entangling uses `CRX`.

## 4. Verification Plan
*   Run a quick sanity check script to ensure the PyTorch+PennyLane forward pass works for all 7 variations.
*   Verify the data loader successfully reads the folders inside `train&test/`.

> [!IMPORTANT]
> **Requirements:** 
> To run this code, you will need the `pennylane` package installed in your Python environment (`pip install pennylane`). 
