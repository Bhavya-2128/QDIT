import argparse
import torch
import os
import sys

# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import HybridQuantumResNet18
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Quantum Transfer Learning for DR Detection")
    parser.add_argument('--data_dir', type=str, default='../train&test', help='Path to dataset')
    parser.add_argument('--variation', type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=1, 
                        help='Which model variation to run (1-7 corresponding to table rows)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--q_depth', type=int, default=6, help='Depth of quantum circuit (number of variational layers)')
    
    args = parser.parse_args()
    
    # Model configs mapping as per the paper's table
    # 1: S(Phase) - Hadamard & CNOT
    # 2: S†(Dagger) - Hadamard & CNOT
    # 3: Hadamard & SWAP
    # 4: Hadamard & CZ
    # 5: RX & CNOT
    # 6: Hadamard & CRX
    # 7: RX & CRX
    configs = {
        1: ('S_H', 'CNOT'),
        2: ('S_dagger_H', 'CNOT'),
        3: ('H', 'SWAP'),
        4: ('H', 'CZ'),
        5: ('RX', 'CNOT'),
        6: ('H', 'CRX'),
        7: ('RX', 'CRX')
    }
    
    embed_type, entangle_type = configs[args.variation]
    print(f"Running Variation {args.variation}: Embedding={embed_type}, Entangling={entangle_type}, q_depth={args.q_depth}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir_abs = os.path.abspath(args.data_dir)
    print(f"Loading data from {data_dir_abs} ...")
    
    if not os.path.exists(data_dir_abs):
        print(f"Error: Data directory {data_dir_abs} does not exist.")
        return

    try:
        train_loader, test_loader, classes = get_dataloaders(data_dir_abs, batch_size=args.batch_size)
        print(f"Found classes: {classes}")
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}. Ensure '{data_dir_abs}' has the right structure (folders 0-4 for train, and 'test' folder containing 0-4 for eval).")
        return
        
    print("Initializing Hybrid Quantum Model...")
    model = HybridQuantumResNet18(embed_type=embed_type, entangle_type=entangle_type, q_depth=args.q_depth)
    
    print("Starting training...")
    train_model(model, train_loader, test_loader, num_epochs=args.epochs, device=device)

if __name__ == '__main__':
    main()
