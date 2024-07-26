import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split

class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = int(self.images[idx].split('_')[0])  # Assuming label is the part before the first underscore

        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_splits(directory, transform, train_size=0.7, val_size=0.15, test_size=0.15):
    # Ensure the sizes sum to 1
    assert train_size + val_size + test_size == 1, "Sizes must sum to 1"

    # Initialize dataset
    full_dataset = CustomDataset(directory=directory, transform=transform)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size  # Ensure all data is used

    # Create data splits
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def get_dataloader(directory, batch_size, shuffle=True, num_workers=0, transform=None):
    dataset = CustomDataset(directory=directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader