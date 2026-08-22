import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class ClassicalCNN(nn.Module):
    """
    Classical CNN matching the parameters described in Figure 6 of the paper.
    Input: (None, 1, 16, 16)
    Output: 16 features
    """
    def __init__(self):
        super().__init__()
        # Conv2D_6: 4 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2D_7: 4 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        # Optional: scale classical output to be suitable for quantum angle encoding [-pi, pi]
        x = torch.tanh(x) * np.pi
        return x

class QuantumCircuit(nn.Module):
    """
    16-Qubit Parameterised Quantum Circuit (PQC)
    Uses RX encoding, Star-topology IsingXX/ZZ entanglement, and RZ learnable rotations.
    """
    def __init__(self, n_qubits=16):
        super().__init__()
        self.n_qubits = n_qubits
        
        # 16 learnable parameters as per the paper's PQC design
        self.q_weights = nn.Parameter(torch.randn(self.n_qubits))
        
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")
        
    def circuit(self, inputs, weights):
        # 1. Data Encoding: Angle encoding using RX
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)
            
        # 2. Entangling Layer: 15 XX and 15 ZZ coupling gates
        # "entangling one qubit (q0) with all others makes q0 a central node"
        for i in range(1, self.n_qubits):
            qml.IsingXX(np.pi/2, wires=[0, i])
            
        for i in range(1, self.n_qubits):
            qml.IsingZZ(np.pi/2, wires=[0, i])
            
        # 3. Variational Layer: Trainable parameters
        for i in range(self.n_qubits):
            qml.RZ(weights[i], wires=i)
            
        # 4. Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
    def forward(self, x):
        q_out = torch.zeros(x.shape[0], self.n_qubits).to(x.device)
        for i in range(x.shape[0]):
            res = self.qnode(x[i], self.q_weights)
            q_out[i] = torch.stack(res) if isinstance(res, (list, tuple)) else res
        return q_out

class FedQCNN(nn.Module):
    """
    The full hybrid model.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = ClassicalCNN()
        self.qnn = QuantumCircuit()
        
        # Final post-processing layer to map the 16 quantum measurements to 5 classes
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        features = self.cnn(x)
        q_features = self.qnn(features)
        out = self.fc(q_features)
        return out
