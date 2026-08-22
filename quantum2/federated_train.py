import torch
import torch.nn as nn
import torch.optim as optim
import copy

def federated_train(global_model, client_loaders, test_loader, num_rounds=5, local_epochs=1, learning_rate=0.001, device='cpu'):
    """
    Simulates Federated Learning over multiple communication rounds.
    """
    global_model = global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    num_clients = len(client_loaders)
    
    for round_idx in range(num_rounds):
        print(f"\n--- Global Communication Round {round_idx + 1}/{num_rounds} ---")
        
        # 1. Distribute global model weights to all clients
        global_weights = global_model.state_dict()
        local_weights = []
        
        # 2. Local Training for each client
        for client_idx in range(num_clients):
            print(f"Training Client {client_idx + 1}/{num_clients}...")
            # Initialize local model with global weights
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
            
            local_model.train()
            loader = client_loaders[client_idx]
            
            for epoch in range(local_epochs):
                running_loss = 0.0
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_loss = running_loss / len(loader.dataset)
                print(f"  Client {client_idx + 1} - Local Epoch {epoch + 1}/{local_epochs} - Loss: {epoch_loss:.4f}")
                
            # Save trained local weights
            local_weights.append(local_model.state_dict())
            
        # 3. Global Aggregation (simulating AirComp weighted average)
        print("Aggregating local models into Global Model...")
        aggregated_weights = copy.deepcopy(global_weights)
        
        for key in aggregated_weights.keys():
            # Average the weights across all clients
            aggregated_weights[key] = sum(lw[key] for lw in local_weights) / num_clients
            
        # Update global model
        global_model.load_state_dict(aggregated_weights)
        
        # 4. Global Evaluation
        global_model.eval()
        running_corrects = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = global_model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
        test_loss = running_loss / len(test_loader.dataset)
        test_acc = running_corrects.double() / len(test_loader.dataset)
        print(f"--> Global Test Loss: {test_loss:.4f} | Global Test Accuracy: {test_acc:.4f}")

    return global_model
