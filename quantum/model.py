import torch
import torch.nn as nn
import pennylane as qml
from torchvision import models
import numpy as np

class DressedQuantumNet(nn.Module):
    def __init__(self, n_qubits=4, q_depth=6, embed_type='H', entangle_type='CNOT'):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.embed_type = embed_type
        self.entangle_type = entangle_type
        
        # Classical pre-processing: 512 -> 4
        self.pre_net = nn.Linear(512, n_qubits)
        
        # Classical post-processing: 4 -> 5
        self.post_net = nn.Linear(n_qubits, 5)
        
        # Quantum device (Using lightning.qubit for fast C++ execution)
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        # Trainable weights for quantum circuit
        # Size is (q_depth, n_qubits)
        self.q_weights = nn.Parameter(torch.randn(q_depth, n_qubits))
        
        # Define the QNode
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        
    def quantum_circuit(self, inputs, weights):
        # 1. Embedding Layer
        for i in range(self.n_qubits):
            if self.embed_type == 'H':
                qml.Hadamard(wires=i)
            elif self.embed_type == 'S_H':
                qml.S(wires=i)
                qml.Hadamard(wires=i)
            elif self.embed_type == 'S_dagger_H':
                qml.adjoint(qml.S)(wires=i)
                qml.Hadamard(wires=i)
            elif self.embed_type == 'RX':
                # Fixed rotation to create superposition, as H has no params
                qml.RX(np.pi/2, wires=i)
            else:
                qml.Hadamard(wires=i) # Default
                
            # Input encoding (RY layer with input features)
            qml.RY(inputs[i], wires=i)
            
        # 2. Variational Layers
        for layer in range(self.q_depth):
            # Entangling layer
            for i in range(self.n_qubits):
                target = (i + 1) % self.n_qubits
                if self.entangle_type == 'CNOT':
                    qml.CNOT(wires=[i, target])
                elif self.entangle_type == 'CZ':
                    qml.CZ(wires=[i, target])
                elif self.entangle_type == 'SWAP':
                    qml.SWAP(wires=[i, target])
                elif self.entangle_type == 'CRX':
                    qml.CRX(weights[layer, i], wires=[i, target])
                else:
                    qml.CNOT(wires=[i, target])
                    
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
                
        # 3. Measurement Layer
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
    def forward(self, x):
        # Pre-process
        pre_out = self.pre_net(x)
        # Scaled to [-pi/2, pi/2]
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        
        # Quantum circuit - process each sample in the batch
        q_out = torch.zeros(x.shape[0], self.n_qubits).to(x.device)
        for i in range(x.shape[0]):
            res = self.qnode(q_in[i], self.q_weights)
            q_out[i] = torch.stack(res) if isinstance(res, (list, tuple)) else res
            
        # Post-process
        return self.post_net(q_out)

class HybridQuantumResNet18(nn.Module):
    def __init__(self, embed_type='H', entangle_type='CNOT', q_depth=6):
        super().__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all classical weights
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer (which takes 512 in ResNet18)
        self.resnet.fc = DressedQuantumNet(
            n_qubits=4, 
            q_depth=q_depth, 
            embed_type=embed_type, 
            entangle_type=entangle_type
        )
        
    def forward(self, x):
        return self.resnet(x)
