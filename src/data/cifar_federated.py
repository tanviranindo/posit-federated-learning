#!/usr/bin/env python3
"""
CIFAR-10 Federated Dataset Management

Handles CIFAR-10 dataset preparation for federated learning scenarios
with controlled subset sizes and distribution strategies.

Authors: Tanvir Rahman, Annajiat Alim Rasel
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict, Any


class FederatedCIFAR10:
    """
    CIFAR-10 dataset manager for federated learning experiments.
    
    Provides controlled subset sizes to simulate resource-constrained
    IoT scenarios while ensuring statistical validity.
    """
    
    def __init__(self, data_dir: str = './data', download: bool = True):
        """
        Initialize CIFAR-10 federated dataset manager.
        
        Args:
            data_dir: Directory to store CIFAR-10 data
            download: Whether to download data if not present
        """
        self.data_dir = data_dir
        
        # Data transformations for training and testing
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=download, 
            transform=self.train_transform
        )
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=download,
            transform=self.test_transform
        )
        
    def create_federated_split(self, num_clients: int, 
                              samples_per_client: int = 5000,
                              test_samples: int = 10000,
                              distribution: str = "iid") -> Tuple[List[DataLoader], DataLoader]:
        """
        Create federated data splits for training.
        
        Args:
            num_clients: Number of federated clients
            samples_per_client: Training samples per client
            test_samples: Total test samples (centralized evaluation)
            distribution: "iid" or "non_iid" data distribution
            
        Returns:
            Tuple of (client_train_loaders, centralized_test_loader)
        """
        # Create training splits
        if distribution == "iid":
            client_loaders = self._create_iid_split(num_clients, samples_per_client)
        else:
            client_loaders = self._create_non_iid_split(num_clients, samples_per_client)
            
        # Create centralized test set
        test_indices = np.random.choice(len(self.test_dataset), test_samples, replace=False)
        test_subset = Subset(self.test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        
        return client_loaders, test_loader
        
    def _create_iid_split(self, num_clients: int, samples_per_client: int) -> List[DataLoader]:
        """Create IID data splits across clients."""
        client_loaders = []
        
        # Randomly sample training data for each client
        total_samples = num_clients * samples_per_client
        indices = np.random.choice(len(self.train_dataset), total_samples, replace=False)
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            client_indices = indices[start_idx:end_idx]
            
            client_subset = Subset(self.train_dataset, client_indices)
            client_loader = DataLoader(client_subset, batch_size=32, shuffle=True)
            client_loaders.append(client_loader)
            
        return client_loaders
        
    def _create_non_iid_split(self, num_clients: int, samples_per_client: int) -> List[DataLoader]:
        """Create Non-IID data splits with class imbalance."""
        client_loaders = []
        
        # Get all labels and organize by class
        all_labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        class_indices = {}
        for class_id in range(10):  # CIFAR-10 has 10 classes
            class_indices[class_id] = np.where(all_labels == class_id)[0]
            
        # Assign 2-3 dominant classes per client
        classes_per_client = 3
        client_class_assignments = []
        
        for i in range(num_clients):
            # Randomly assign dominant classes
            dominant_classes = np.random.choice(10, classes_per_client, replace=False)
            client_class_assignments.append(dominant_classes)
            
        # Create client datasets with class imbalance
        for i, dominant_classes in enumerate(client_class_assignments):
            client_indices = []
            
            # 80% samples from dominant classes
            dominant_samples = int(samples_per_client * 0.8)
            samples_per_dominant_class = dominant_samples // len(dominant_classes)
            
            for class_id in dominant_classes:
                class_samples = np.random.choice(
                    class_indices[class_id], samples_per_dominant_class, replace=False
                )
                client_indices.extend(class_samples)
                
            # 20% samples from other classes
            other_classes = [c for c in range(10) if c not in dominant_classes]
            remaining_samples = samples_per_client - len(client_indices)
            
            if remaining_samples > 0:
                other_indices = []
                for class_id in other_classes:
                    other_indices.extend(class_indices[class_id])
                    
                additional_samples = np.random.choice(
                    other_indices, remaining_samples, replace=False
                )
                client_indices.extend(additional_samples)
                
            # Create client loader
            client_subset = Subset(self.train_dataset, client_indices)
            client_loader = DataLoader(client_subset, batch_size=32, shuffle=True)
            client_loaders.append(client_loader)
            
        return client_loaders
        
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return {
            'total_train_samples': len(self.train_dataset),
            'total_test_samples': len(self.test_dataset),
            'num_classes': 10,
            'image_shape': (3, 32, 32),
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                          'dog', 'frog', 'horse', 'ship', 'truck']
        }