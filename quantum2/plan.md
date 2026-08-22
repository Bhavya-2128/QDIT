# Implementation Plan: FedQCNN

Based on the paper "FedQCNN: A Privacy-Preserving Federated Quantum Convolutional Neural Network for Retinal Image Classification", I will implement the hybrid classical-quantum federated architecture in the `quantum2` directory.

## 1. Directory Structure
I will create the following files in the `quantum2/` directory:
*   **`dataset.py`**: Handles dataset loading from your `train&test/` folder. It will preprocess the images to match the paper: extract the green channel, and resize to 16x16 pixels. *(Note: The paper used 16x16 patches extracted from large images. To apply this to your full-image dataset seamlessly, we will resize the images to 16x16).*
*   **`model.py`**: Defines the Classical CNN feature extractor (188 parameters) and the Parameterised Quantum Circuit (PQC) with 16 qubits.
*   **`federated_train.py`**: Simulates the Federated Learning environment. It will split your training data among `n` simulated local clients (edge nodes), train them locally using Quantum Gradient Descent, and aggregate their weights (simulating AirComp weighted average) at a global server.
*   **`main.py`**: The entry point to configure the number of federated clients, epochs, and start the training process.

## 2. Model Architecture Details

### Classical CNN Feature Extractor
As explicitly stated in the paper (Figure 6), the classical network takes a `16x16x1` input (green channel) and extracts 16 features:
1.  **Conv2D:** 4 filters, 3x3 kernel $\rightarrow$ Output `14x14x4` (40 params)
2.  **MaxPooling2D:** 2x2 pool $\rightarrow$ Output `7x7x4`
3.  **Conv2D:** 4 filters, 3x3 kernel $\rightarrow$ Output `5x5x4` (148 params)
4.  **MaxPooling2D:** 2x2 pool $\rightarrow$ Output `2x2x4`
5.  **Flatten:** $\rightarrow$ Output `16` features.

### Quantum Neural Network (QNN)
A 16-qubit quantum circuit using PennyLane:
1.  **Data Encoding:** The 16 classical features are angle-encoded into the 16 qubits using $R_X$ gates.
2.  **Entangling Layer:** Implements a "star topology" entanglement where qubit 0 ($q_0$) acts as the central node. It applies 15 $R_{XX}$ coupling gates and 15 $R_{ZZ}$ coupling gates between $q_0$ and all other qubits to capture spatial correlations.
3.  **Variational Layer:** Applies trainable $R_Z$ rotations on each of the 16 qubits (providing the 16 learnable parameters mentioned in the paper).
4.  **Measurement & Post-Processing:** The paper describes measuring the Pauli-Z operator for binary classification (healthy vs. unhealthy). Since your dataset has 5 classes (0-4), we will measure the expectation values of all 16 qubits and apply a final classical linear layer `nn.Linear(16, 5)` to output the 5 DR classes.

## 3. Federated Learning (FL) Simulation
Instead of a standard training loop, `federated_train.py` will:
1.  Divide the `train_loader` into $N$ distinct local datasets (e.g., $N=3$ hospitals).
2.  Initialize a Global Model.
3.  **Communication Round:**
    *   Broadcast global weights to $N$ local models.
    *   Train each local model for $E$ local epochs using the Adam optimizer.
    *   Aggregate the local weights back into the Global Model using a weighted average (simulating the Over-the-Air Computation / AirComp).
