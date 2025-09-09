#!/usr/bin/env python3
"""
Adaptive CNN Models for Cross-Architecture Federated Learning

Architecture-specific CNN implementations that automatically adapt complexity
based on target platform capabilities (x86_64 vs ARM64).

Authors: Tanvir Rahman, Annajiat Alim Rasel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class AdaptiveCNN(nn.Module):
    """
    CNN that adapts complexity based on target architecture.
    
    x86_64: Full complexity (32→64→128 channels, ~1.28M parameters)  
    ARM64: Reduced complexity (16→32→64 channels, ~320K parameters)
    """
    
    def __init__(self, num_classes: int = 10, architecture: str = "x86_64"):
        super(AdaptiveCNN, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Architecture-specific channel configurations
        if architecture == "arm64":
            channels = [16, 32, 64]  # Energy-efficient for ARM64
            fc_sizes = [256, 128]
        else:  # x86_64 or default
            channels = [32, 64, 128]  # Full capacity for x86_64
            fc_sizes = [512, 256]
            
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.bn3 = nn.BatchNorm2d(channels[2])
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        fc_input_size = channels[2] * 4 * 4  # After adaptive pooling
        self.fc1 = nn.Linear(fc_input_size, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1]) 
        self.fc3 = nn.Linear(fc_sizes[1], num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Architecture flag for federated compatibility
        self.is_arm64_optimized = (architecture == "arm64")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2  
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture-specific configuration information."""
        return {
            'architecture': self.architecture,
            'parameter_count': self.get_parameter_count(),
            'is_arm64_optimized': self.is_arm64_optimized,
            'energy_efficient': self.architecture == "arm64"
        }


def create_model_for_architecture(architecture: str, num_classes: int = 10) -> AdaptiveCNN:
    """
    Factory function to create architecture-specific models.
    
    Args:
        architecture: Target architecture ("x86_64" or "arm64")
        num_classes: Number of output classes
        
    Returns:
        Optimized CNN model for the target architecture
    """
    return AdaptiveCNN(num_classes=num_classes, architecture=architecture)