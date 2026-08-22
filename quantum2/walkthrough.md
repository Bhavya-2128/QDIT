# FedQCNN Model Implementation Walkthrough

I have successfully built the complete federated hybrid classical-quantum framework as specified in the "FedQCNN" paper. 

Here is a breakdown of what was implemented in your `quantum2` directory:

### 1. Preprocessing and Data Splitting (`dataset.py`)
To mimic the paper's preprocessing of retinal images while supporting your 5-class DR dataset:
*   I created a custom `GreenChannelTransform` that isolates only the green color channel.
*   The images are resized to $16 \times 16$ pixels to strictly match the input size expected by the paper's custom CNN architecture.
*   Because this is a federated learning setup, the training dataset is dynamically split into multiple non-overlapping subsets based on the `--clients` argument you provide (simulating multiple hospitals/nodes).

### 2. Hybrid Model Architecture (`model.py`)
I implemented the exact architecture highlighted in Figure 6 of the paper:
*   **`ClassicalCNN`**: A lightweight PyTorch CNN that compresses the $16 \times 16 \times 1$ image into a $4 \times 4 \times 4$ volume, and then flattens it into exactly 16 feature parameters. This network has exactly **188 parameters**, matching the paper!
*   **`QuantumCircuit`**: A 16-qubit `PennyLane` circuit.
    *   It uses `qml.RX` to angle-encode the 16 features from the CNN.
    *   It utilizes a **Star Topology** for entanglement, applying 15 `qml.IsingXX` and 15 `qml.IsingZZ` coupling gates between qubit 0 and the rest.
    *   It applies a variational layer with `qml.RZ` gates providing the **16 learnable parameters**.
*   **`FedQCNN`**: The wrapper class joining them together and adding the final mapping to your 5 target classes.

### 3. Federated Learning Engine (`federated_train.py`)
This file is the core of the federated framework simulating Over-The-Air Computation (AirComp).
*   It handles the broadcast of the Global Model weights to the local edge clients.
*   Each local client individually runs **Quantum Gradient Descent** (via PyTorch + PennyLane) using the Adam Optimizer to train its local `FedQCNN`.
*   At the end of a round, it simulates the AirComp parameter aggregation by taking the **weighted average** of the parameters across all clients to update the Global Model.

### 4. Running the Simulation (`main.py`)
The unified entry script to run the federated training!

**How to run it:**
Open your terminal and run the following command to start a federated training simulation with 3 clients, 5 global communication rounds, and 2 local epochs per client:
```bash
python main.py --data_dir "../train&test" --clients 3 --rounds 5 --local_epochs 2
```

> [!TIP]
> **Federated Workflow**
> When you run the script, you'll see it first distribute the dataset. Then for each global round, it will train Client 1, Client 2, and Client 3 sequentially. After they all finish their local epochs, it aggregates the weights and tests the new global model against your test set!
