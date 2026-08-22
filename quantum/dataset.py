import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for the DR dataset.
    Args:
        data_dir: Path to 'train&test' folder. It should contain '0', '1', '2', '3', '4' for training,
                  and a 'test' folder with '0', '1', '2', '3', '4' for testing.
    """
    
    # As per paper: resize to 224x224, center crop, normalize.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load training dataset
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Filter out the 'test' folder which ImageFolder might incorrectly treat as a class
    if 'test' in train_dataset.class_to_idx:
        test_idx = train_dataset.class_to_idx['test']
        train_dataset.samples = [s for s in train_dataset.samples if s[1] != test_idx]
        train_dataset.targets = [t for t in train_dataset.targets if t != test_idx]
        train_dataset.classes.remove('test')
        del train_dataset.class_to_idx['test']
        
        # Re-map indices to be strictly 0 to num_classes-1
        new_class_to_idx = {cls_name: i for i, cls_name in enumerate(train_dataset.classes)}
        idx_mapping = {old_idx: new_class_to_idx[cls_name] for cls_name, old_idx in train_dataset.class_to_idx.items() if cls_name != 'test'}
        
        train_dataset.class_to_idx = new_class_to_idx
        train_dataset.samples = [(s[0], idx_mapping.get(s[1], s[1])) for s in train_dataset.samples]
        train_dataset.targets = [idx_mapping.get(t, t) for t in train_dataset.targets]

    # Load test dataset
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, train_dataset.classes
