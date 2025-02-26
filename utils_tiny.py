# utils_tiny.py
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from urllib.request import urlretrieve
import zipfile

def download_tiny_imagenet(root='./data'):
    print("Using root directory:", root)
    if not os.path.exists(root):
        print("Root folder does not exist. Creating:", root)
        os.makedirs(root, exist_ok=True)
    else:
        print("Root folder exists.")
        
    data_dir = os.path.join(root, 'tiny-imagenet-200')
    if not os.path.exists(data_dir):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")
        print("Downloading Tiny ImageNet from:", url)
        urlretrieve(url, zip_path)
        print("Download complete. Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet already exists at:", data_dir)
    return data_dir

def load_tiny_imagenet(batch_size=128, train=True, root='./data'):
    """
    Returns a DataLoader for Tiny ImageNet.
    For training: uses the 'train' folder with data augmentation.
    For validation/testing: uses the 'val' folder.
    """
    data_dir = download_tiny_imagenet(root)
    if train:
        folder = os.path.join(data_dir, 'train')
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821))
        ])
    else:
        folder = os.path.join(data_dir, 'val')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2770, 0.2691, 0.2821))
        ])
    print("Loading data from folder:", folder)
    dataset = datasets.ImageFolder(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
    return loader

if __name__ == "__main__":
    # Run the download and data loading functions to see debug output
    data_directory = download_tiny_imagenet()
    print("Tiny ImageNet data directory:", data_directory)
    loader = load_tiny_imagenet(train=True)
    print("DataLoader loaded. Number of batches:", len(loader))
