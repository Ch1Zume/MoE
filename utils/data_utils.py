import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import random
import os
import numpy as np

def fix_random_seed(seed=42, deterministic=True, benchmark=False):
    """
    Fixes the random seed for reproducibility in deep learning experiments.

    Args:
        seed (int): The random seed value to use (default: 42).
        deterministic (bool): Whether to enforce full determinism in CUDA operations.
        benchmark (bool): Whether to enable CuDNN benchmarking (can improve speed but reduces reproducibility).
    
    Returns:
        None
    """
    # Fix Python's built-in random seed
    random.seed(seed)

    # Fix Python hash seed (affects string hashing)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Fix NumPy seed
    np.random.seed(seed)

    # Fix PyTorch CPU randomness
    torch.manual_seed(seed)

    # Fix PyTorch CUDA randomness (for single GPU)
    torch.cuda.manual_seed(seed)

    # Fix PyTorch CUDA randomness (for multiple GPUs)
    torch.cuda.manual_seed_all(seed)

    # Set CuDNN behavior
    torch.backends.cudnn.benchmark = benchmark  # Enable/disable CuDNN benchmark mode
    torch.backends.cudnn.deterministic = deterministic  # Enforce deterministic computations

    print(f"Random seed set to {seed} | Deterministic: {deterministic} | CuDNN Benchmark: {benchmark}")

def get_dataloader(batch_size: int, dataset: str, dataset_path: str, val_split: float):
    
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_set = datasets.MNIST(root=dataset_path, train=True, transform=transform)
        test_set = datasets.MNIST(root=dataset_path, train=False, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_set = datasets.CIFAR10(root=dataset_path, train=True, transform=transform)
        test_set = datasets.CIFAR10(root=dataset_path, train=False, transform=transform)
    elif dataset == "cifar100":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_set = datasets.CIFAR100(root=dataset_path, train=True, transform=transform)
        test_set = datasets.CIFAR100(root=dataset_path, train=False, transform=transform)
    elif dataset == "imagenet224":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.ImageNet(root=dataset_path, split="train", transform=transform)
        test_set = datasets.ImageNet(root=dataset_path, split="val", transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    # Split the training set into training and validation sets
    train_size = int((1 - val_split) * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader =DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3)
    return train_loader, val_loader, test_loader
