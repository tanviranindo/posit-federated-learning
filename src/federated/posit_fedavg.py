"""
Posit-Enhanced Federated Averaging Algorithm
============================================

This module implements FedAvg with Posit arithmetic for improved numerical
precision and cross-architecture consistency in federated learning.

Key Innovation:
- Quire-based exact model aggregation
- Cross-architecture numerical stability
- Enhanced precision tracking and analysis
- Compatible with existing FL frameworks

Research Contribution:
First federated learning algorithm using next-generation numerical formats
for improved aggregation consistency across heterogeneous hardware.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import copy

from ..posit.quire_aggregator import QuireAggregator
from ..models.posit_cnn import PositCNN
from ..utils.metrics import PrecisionMetrics


class PositFedAvg:
    """
    Federated Averaging with Posit arithmetic precision.
    
    This implementation extends standard FedAvg with Posit-based aggregation
    for improved numerical consistency across different hardware architectures.
    """
    
    def __init__(self, 
                 model_class: type = PositCNN,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 es: int = 2,
                 nbits: int = 16,
                 device: Optional[torch.device] = None):
        """
        Initialize PositFedAvg coordinator.
        
        Args:
            model_class: Neural network class to use
            model_kwargs: Arguments for model initialization
            es: Exponent size for Posit format
            nbits: Total bits for Posit representation
            device: Computation device
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.es = es
        self.nbits = nbits
        self.device = device or torch.device('cpu')
        
        # Initialize global model
        self.global_model = self._create_model()
        
        # Initialize Posit aggregator
        self.aggregator = QuireAggregator(es=es, nbits=nbits)
        
        # Initialize metrics tracker
        self.metrics = PrecisionMetrics()
        
        # Training history
        self.training_history = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PositFedAvg: es={es}, nbits={nbits}, device={device}")
    
    def _create_model(self) -> nn.Module:
        """Create a new model instance."""
        model = self.model_class(**self.model_kwargs)
        return model.to(self.device)
    
    def train_client(self, 
                    client_data: DataLoader,
                    client_id: str,
                    local_epochs: int = 2,
                    learning_rate: float = 0.01) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Train a client model locally.
        
        Args:
            client_data: Client's training data
            client_id: Unique client identifier
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            
        Returns:
            Tuple of (model_state_dict, training_metrics)
        """
        self.logger.debug(f"Training client {client_id} for {local_epochs} epochs")
        
        # Create local model copy
        local_model = self._create_model()
        local_model.load_state_dict(self.global_model.state_dict())
        local_model.train()
        
        # Setup optimizer and criterion
        optimizer = torch.optim.SGD(local_model.parameters(), 
                                  lr=learning_rate, 
                                  momentum=0.9, 
                                  weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        epoch_metrics = []
        start_time = time.time()
        
        for epoch in range(local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(client_data):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Record epoch metrics
            epoch_accuracy = 100. * correct / total
            avg_loss = total_loss / len(client_data)
            
            epoch_metrics.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': epoch_accuracy,
                'samples': total
            })
            
            self.logger.debug(f"Client {client_id} Epoch {epoch}: Loss={avg_loss:.4f}, Acc={epoch_accuracy:.2f}%")
        
        training_time = time.time() - start_time
        
        # Collect final metrics
        final_metrics = {
            'client_id': client_id,
            'local_epochs': local_epochs,
            'training_time': training_time,
            'final_loss': epoch_metrics[-1]['loss'],
            'final_accuracy': epoch_metrics[-1]['accuracy'],
            'epoch_history': epoch_metrics,
            'total_samples': sum(m['samples'] for m in epoch_metrics)
        }
        
        return local_model.state_dict(), final_metrics
    
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]],
                        client_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Aggregate client models using Posit-enhanced FedAvg.
        
        Args:
            client_models: List of client model state dictionaries
            client_weights: Optional client weights for weighted averaging
            
        Returns:
            Aggregation results and metrics
        """
        self.logger.info(f"Aggregating {len(client_models)} client models with Posit precision")
        
        start_time = time.time()
        
        # Precision analysis before aggregation
        pre_aggregation_analysis = self.aggregator.precision_analysis(client_models)
        
        # Perform Posit-based aggregation
        aggregated_state = self.aggregator.federated_average(client_models, client_weights)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        # Compute aggregation error
        aggregation_error = self.aggregator.compute_aggregation_error(
            client_models, aggregated_state
        )
        
        aggregation_time = time.time() - start_time
        
        # Compile aggregation metrics
        aggregation_metrics = {
            'num_clients': len(client_models),
            'aggregation_time': aggregation_time,
            'aggregation_error': aggregation_error,
            'precision_analysis': pre_aggregation_analysis,
            'client_weights': client_weights or ([1.0 / len(client_models)] * len(client_models))
        }
        
        self.logger.info(f"Aggregation completed: error={aggregation_error:.2e}, time={aggregation_time:.2f}s")
        
        return aggregation_metrics
    
    def evaluate_global_model(self, test_data: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the global model on test data.
        
        Args:
            test_data: Test dataset loader
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_data)
        
        evaluation_results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct_predictions': correct,
            'total_samples': total,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        self.logger.info(f"Global model evaluation: Acc={accuracy:.2f}%, Loss={avg_loss:.4f}")
        
        return evaluation_results
    
    def run_federation_round(self,
                           client_data_loaders: List[DataLoader],
                           client_ids: List[str],
                           test_data: DataLoader,
                           local_epochs: int = 2,
                           learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Execute one complete federation round.
        
        Args:
            client_data_loaders: List of client data loaders
            client_ids: List of client identifiers
            test_data: Global test dataset
            local_epochs: Local training epochs per client
            learning_rate: Learning rate for local training
            
        Returns:
            Round results and metrics
        """
        self.logger.info(f"Starting federation round with {len(client_ids)} clients")
        
        round_start_time = time.time()
        
        # Client training phase
        client_models = []
        client_metrics = []
        
        for client_data, client_id in zip(client_data_loaders, client_ids):
            model_state, metrics = self.train_client(
                client_data, client_id, local_epochs, learning_rate
            )
            client_models.append(model_state)
            client_metrics.append(metrics)
        
        # Model aggregation phase
        aggregation_results = self.aggregate_models(client_models)
        
        # Global evaluation phase
        evaluation_results = self.evaluate_global_model(test_data)
        
        round_time = time.time() - round_start_time
        
        # Compile round results
        round_results = {
            'client_metrics': client_metrics,
            'aggregation_results': aggregation_results,
            'evaluation_results': evaluation_results,
            'round_time': round_time,
            'timestamp': time.time()
        }
        
        # Store in history
        self.training_history.append(round_results)
        
        self.logger.info(f"Federation round completed: Acc={evaluation_results['accuracy']:.2f}%, Time={round_time:.1f}s")
        
        return round_results
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        return self.global_model.state_dict()
    
    def load_model_state(self, state_dict: Dict[str, torch.Tensor]):
        """Load global model state."""
        self.global_model.load_state_dict(state_dict)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history."""
        return self.training_history
    
    def get_precision_config(self) -> Dict[str, Any]:
        """Get Posit precision configuration."""
        return {
            'es': self.es,
            'nbits': self.nbits,
            'device': str(self.device),
            'model_class': self.model_class.__name__,
            'model_kwargs': self.model_kwargs
        }