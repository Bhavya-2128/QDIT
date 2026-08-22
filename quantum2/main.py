import argparse
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import FedQCNN
from federated_train import federated_train

def main():
    parser = argparse.ArgumentParser(description="FedQCNN: Privacy-Preserving Federated Quantum CNN")
    parser.add_argument('--data_dir', type=str, default='../train&test', help='Path to dataset')
    parser.add_argument('--clients', type=int, default=3, help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs per client per round')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir_abs = os.path.abspath(args.data_dir)
    print(f"Loading data from {data_dir_abs} ...")
    
    if not os.path.exists(data_dir_abs):
        print(f"Error: Data directory {data_dir_abs} does not exist.")
        return

    try:
        client_loaders, test_loader, classes = get_dataloaders(
            data_dir_abs, 
            batch_size=args.batch_size, 
            num_clients=args.clients
        )
        print(f"Found classes: {classes}")
        print(f"Distributed training data among {args.clients} clients.")
        for i, loader in enumerate(client_loaders):
            print(f"  Client {i+1} data size: {len(loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}. Ensure '{data_dir_abs}' has the right structure.")
        return
        
    print("\nInitializing Global FedQCNN Model...")
    global_model = FedQCNN(num_classes=len(classes))
    
    print("Starting Federated Learning Simulation...")
    federated_train(
        global_model, 
        client_loaders, 
        test_loader, 
        num_rounds=args.rounds, 
        local_epochs=args.local_epochs, 
        device=device
    )

if __name__ == '__main__':
    main()
