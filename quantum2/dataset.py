import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class GreenChannelTransform:
    """Extracts the green channel from an RGB image."""
    def __call__(self, img):
        # img is a tensor of shape (3, H, W)
        # We return the green channel, retaining the channel dimension (1, H, W)
        return img[1:2, :, :]

def get_dataloaders(data_dir, batch_size=32, num_workers=4, num_clients=3):
    """
    Creates DataLoaders for FedQCNN.
    Splits the training data among `num_clients` for Federated Learning.
    """
    # Preprocessing as per paper: extract green channel, resize to 16x16
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        GreenChannelTransform()
    ])
    
    # Load training dataset
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Filter out 'test' folder logic
    if 'test' in train_dataset.class_to_idx:
        test_idx = train_dataset.class_to_idx['test']
        train_dataset.samples = [s for s in train_dataset.samples if s[1] != test_idx]
        train_dataset.targets = [t for t in train_dataset.targets if t != test_idx]
        train_dataset.classes.remove('test')
        del train_dataset.class_to_idx['test']
        
        new_class_to_idx = {cls_name: i for i, cls_name in enumerate(train_dataset.classes)}
        idx_mapping = {old_idx: new_class_to_idx[cls_name] for cls_name, old_idx in train_dataset.class_to_idx.items() if cls_name != 'test'}
        
        train_dataset.class_to_idx = new_class_to_idx
        train_dataset.samples = [(s[0], idx_mapping.get(s[1], s[1])) for s in train_dataset.samples]
        train_dataset.targets = [idx_mapping.get(t, t) for t in train_dataset.targets]

    # Load test dataset
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    # Split train_dataset for federated learning clients
    total_len = len(train_dataset)
    split_size = total_len // num_clients
    lengths = [split_size] * num_clients
    lengths[-1] += total_len - sum(lengths) # Add remainder to last client
    
    client_datasets = random_split(train_dataset, lengths)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers) for ds in client_datasets]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return client_loaders, test_loader, train_dataset.classes
