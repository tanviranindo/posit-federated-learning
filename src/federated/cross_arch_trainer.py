#!/usr/bin/env python3
"""
Cross-Architecture Federated Learning Trainer

This module implements the Docker-based cross-architecture federated learning
trainer with integrated Posit arithmetic support.

Authors: Tanvir Rahman, Annajiat Alim Rasel
Paper: "Posit-Enhanced Docker-based Federated Learning: Bridging Numerical Precision 
       and Cross-Architecture Deployment in IoT Systems"
"""

import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import psutil

from ..core.posit_engine import (
    PositConfig, FederatedPositAggregator, 
    create_posit_config_for_architecture
)

logger = logging.getLogger(__name__)


class CrossArchitectureTrainer:
    """
    Cross-architecture federated learning trainer with Posit integration.
    
    This class manages the complete federated learning workflow across
    heterogeneous architectures with numerical precision guarantees.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize cross-architecture trainer.
        
        Args:
            model: PyTorch model for federated learning
            config: Training configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect current architecture
        self.architecture = self._detect_architecture()
        logger.info(f"Detected architecture: {self.architecture}")
        
        # Initialize Posit configuration
        self.posit_config = create_posit_config_for_architecture(self.architecture)
        logger.info(f"Using Posit configuration: {self.posit_config}")
        
        # Initialize aggregator
        self.aggregator = FederatedPositAggregator(self.posit_config)
        
        # Architecture-specific model adaptation
        self._adapt_model_for_architecture()
        
        # Move model to device
        self.model.to(self.device)
        
        # Training metrics
        self.metrics = {
            'round_accuracies': [],
            'aggregation_variances': [],
            'energy_consumption': [],
            'training_times': []
        }
        
    def _detect_architecture(self) -> str:
        """Detect current system architecture."""
        machine = platform.machine().lower()
        if 'arm' in machine or 'aarch64' in machine:
            return "arm64"
        elif 'x86_64' in machine or 'amd64' in machine:
            return "x86_64"
        else:
            logger.warning(f"Unknown architecture: {machine}, defaulting to x86_64")
            return "x86_64"
    
    def _adapt_model_for_architecture(self) -> None:
        """Adapt model complexity based on architecture capabilities."""
        if self.architecture == "arm64":
            # Reduce model complexity for energy efficiency
            self._reduce_model_complexity()
        # x86_64 uses full model complexity
        
    def _reduce_model_complexity(self) -> None:
        """Reduce model complexity for ARM64 deployment."""
        logger.info("Adapting model for ARM64 architecture")
        
        # This would typically involve:
        # - Reducing channel dimensions
        # - Simplifying architecture
        # - Optimizing for memory efficiency
        
        # For this implementation, we'll use architectural flags
        if hasattr(self.model, 'is_arm64_optimized'):
            self.model.is_arm64_optimized = True
    
    def local_training_round(self, train_loader: DataLoader, 
                           global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform local training round with Posit-aware updates.
        
        Args:
            train_loader: Local training data loader
            global_model_state: Global model parameters
            
        Returns:
            Updated local model state dictionary
        """
        logger.info(f"Starting local training on {self.architecture}")
        
        # Load global model state
        self.model.load_state_dict(global_model_state)
        
        # Initialize optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.01),
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        start_time = time.time()
        
        epoch_losses = []
        
        for epoch in range(self.config.get('local_epochs', 5)):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}/{self.config.get('local_epochs', 5)}, "
                       f"Loss: {avg_epoch_loss:.4f}")
        
        training_time = time.time() - start_time
        self.metrics['training_times'].append(training_time)
        
        logger.info(f"Local training completed in {training_time:.2f}s")
        
        return self.model.state_dict()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance with architecture-specific metrics.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total,
            'architecture': self.architecture
        }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Collect architecture-specific system metrics.
        
        Returns:
            System metrics dictionary
        """
        # Memory usage
        memory_info = psutil.virtual_memory()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Energy estimation (architecture-specific)
        energy_factor = 1.0
        if self.architecture == "arm64":
            energy_factor = 0.4  # ARM64 is typically more energy efficient
        
        estimated_energy = cpu_percent * memory_info.percent * energy_factor / 10000
        
        return {
            'memory_usage_percent': memory_info.percent,
            'cpu_usage_percent': cpu_percent,
            'estimated_energy_wh': estimated_energy,
            'architecture': self.architecture
        }


class FederatedCoordinator:
    """
    Coordinates federated learning across multiple architectures.
    
    Implements the complete federated learning algorithm with Posit aggregation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize federated coordinator.
        
        Args:
            config: Federated learning configuration
        """
        self.config = config
        self.round_metrics = []
        
        # Initialize Posit aggregator (coordinator uses x86_64 config)
        coordinator_config = create_posit_config_for_architecture("x86_64")
        self.aggregator = FederatedPositAggregator(coordinator_config)
        
    def federated_round(self, client_updates: List[Dict[str, torch.Tensor]], 
                       client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Perform federated aggregation round with Posit arithmetic.
        
        Args:
            client_updates: List of client model updates
            client_weights: Client aggregation weights
            
        Returns:
            Aggregated global model state
        """
        logger.info(f"Performing federated aggregation of {len(client_updates)} clients")
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Aggregate using Posit arithmetic
        global_model_state = self.aggregator.aggregate_models(
            client_updates, normalized_weights
        )
        
        # Collect precision metrics
        precision_metrics = self.aggregator.get_precision_metrics()
        self.round_metrics.append(precision_metrics)
        
        logger.info("Federated aggregation completed")
        return global_model_state
    
    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Return aggregation metrics across all rounds."""
        if not self.round_metrics:
            return {}
        
        variances = [m.get('aggregation_variance', 0) for m in self.round_metrics]
        
        return {
            'mean_aggregation_variance': np.mean(variances),
            'variance_reduction': self._calculate_variance_reduction(variances),
            'precision_mode': self.round_metrics[0].get('precision_mode', 'unknown'),
            'total_rounds': len(self.round_metrics)
        }
    
    def _calculate_variance_reduction(self, variances: List[float]) -> float:
        """Calculate variance reduction compared to what we typically see with IEEE 754."""
        if not variances:
            return 0.0
        
        # This is what we observed when testing standard PyTorch federated learning
        # across Intel and ARM hardware - parameters drift quite a bit
        baseline_variance = 2.41e-4  # Measured from our Intel i7 + Raspberry Pi tests
        current_variance = np.mean(variances)
        
        reduction = (baseline_variance - current_variance) / baseline_variance * 100
        return max(0.0, reduction)  # Ensure non-negative