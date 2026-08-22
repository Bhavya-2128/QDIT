# Quantum Transfer Learning Model Implementation

I have successfully implemented the hybrid classical-quantum models according to the specifications in the paper and our implementation plan. 

Here is a breakdown of what was created in the `quantum` directory:

### 1. Dataset Preprocessing (`dataset.py`)
I implemented a robust data loader using `torchvision.datasets.ImageFolder`.
*   **Transformations**: Images are resized to `224x224` and normalized using standard ImageNet mean/std as required by the ResNet architecture.
*   **Directory Structure Fix**: I added custom logic to filter out the `test` directory from the training class folders so that PyTorch `ImageFolder` correctly identifies the 5 DR classes (0-4) from your `train&test/` directory.

### 2. Hybrid Quantum Architecture (`model.py`)
This file houses the core architecture consisting of two main classes:
*   **`DressedQuantumNet`**: A `torch.nn.Module` containing the PennyLane `QNode`. It incorporates:
    *   **Classical Pre-processing**: Shrinks the 512 ResNet features down to 4 parameters.
    *   **Quantum Circuit**: A flexible 4-qubit circuit with `q_depth` variational layers. The circuit dynamically switches its **Embedding Layer** (H, S-H, RX) and **Entangling Layer** (CNOT, CZ, SWAP, CRX) based on the provided configuration.
    *   **Classical Post-processing**: Maps the 4 quantum measurements (Pauli-Z) to the 5 output classes.
*   **`HybridQuantumResNet18`**: Wraps `torchvision.models.resnet18(pretrained=True)`. It automatically freezes all classical weights and replaces the final fully connected layer with our `DressedQuantumNet`.

### 3. Training Loop (`train.py`)
Implemented a standard PyTorch training loop that:
*   Uses `CrossEntropyLoss` and the `Adam` optimizer.
*   Utilizes a learning rate scheduler (`StepLR`) to decay the learning rate as suggested in the paper.
*   Only trains the quantum parameters (the classical ResNet is frozen).

### 4. Main Execution Script (`main.py`)
A unified command-line interface to easily train any of the 7 model variations from the paper.

## How to Run
You can run any of the 7 model variations using the `main.py` script from your terminal.

**Example 1**: Run Variation 1 (S-Hadamard & CNOT) with default settings (15 epochs, batch size 16):
```bash
python main.py --variation 1 --data_dir "../train&test"
```

**Example 2**: Run Variation 7 (RX & CRX - which achieved 98.1% accuracy in the paper) with custom epochs:
```bash
python main.py --variation 7 --epochs 25
```

> [!TIP]
> **Performance**
> Because the classical ResNet18 layers are frozen and the quantum circuit only has `4 * q_depth` parameters (default 24 parameters), the training process should be extremely fast per epoch once the classical features are extracted!
