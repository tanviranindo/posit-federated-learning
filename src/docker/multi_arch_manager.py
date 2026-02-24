#!/usr/bin/env python3
"""
Multi-Architecture Docker Manager

This module manages Docker-based deployment across heterogeneous architectures
with automatic optimization and Posit arithmetic integration.

Authors: Tanvir Rahman, Annajiat Alim Rasel
Paper: "Posit-Enhanced Docker-based Federated Learning: Bridging Numerical Precision 
       and Cross-Architecture Deployment in IoT Systems"
"""

import os
import json
import subprocess
import platform
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import docker
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiArchitectureManager:
    """
    Manages multi-architecture Docker deployments for federated learning.
    
    Provides automatic architecture detection, optimization, and deployment
    consistency across x86_64 and ARM64 platforms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-architecture manager.
        
        Args:
            config: Deployment configuration dictionary
        """
        self.config = config
        self.client = docker.from_env()
        self.deployment_metrics = []
        
        # Detect current architecture
        self.current_arch = self._detect_architecture()
        logger.info(f"Current architecture: {self.current_arch}")
        
        # Initialize architecture-specific configurations
        self.arch_configs = {
            'x86_64': {
                'base_image': 'python:3.10-slim',
                'optimizations': ['mkl', 'avx2'],
                'resource_limits': {'memory': '4g', 'cpus': '2.0'}
            },
            'arm64': {
                'base_image': 'python:3.10-slim',
                'optimizations': ['openblas', 'neon'],
                'resource_limits': {'memory': '2g', 'cpus': '1.0'}
            }
        }
    
    def _detect_architecture(self) -> str:
        """Detect current system architecture."""
        machine = platform.machine().lower()
        if 'arm' in machine or 'aarch64' in machine:
            return 'arm64'
        elif 'x86_64' in machine or 'amd64' in machine:
            return 'x86_64'
        else:
            logger.warning(f"Unknown architecture: {machine}")
            return 'x86_64'
    
    def generate_dockerfile(self, target_arch: str) -> str:
        """
        Generate optimized Dockerfile for target architecture.
        
        Args:
            target_arch: Target architecture ('x86_64' or 'arm64')
            
        Returns:
            Dockerfile content as string
        """
        arch_config = self.arch_configs.get(target_arch, self.arch_configs['x86_64'])
        
        dockerfile_content = f"""# Multi-stage Dockerfile for {target_arch}
FROM --platform=linux/{target_arch} {arch_config['base_image']} as builder

ARG TARGETARCH
ARG TARGETVARIANT

# Architecture-specific optimizations
RUN apt-get update && apt-get install -y \\
    gcc g++ cmake build-essential \\
    git wget curl

# Install architecture-specific dependencies
"""
        
        if target_arch == 'arm64':
            dockerfile_content += """RUN apt-get install -y libblas-dev liblapack-dev libopenblas-dev
RUN pip install --no-cache-dir numpy==1.24.3 --index-url https://pypi.org/simple/"""
        else:
            dockerfile_content += """RUN pip install --no-cache-dir numpy[mkl]==1.24.3"""
        
        dockerfile_content += """

# Install PyTorch with architecture optimizations
RUN if [ "$TARGETARCH" = "arm64" ]; then \\
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \\
    else \\
      pip install torch torchvision; \\
    fi

# Install SoftPosit for genuine Posit arithmetic
RUN git clone https://github.com/cjdelisle/SoftPosit.git /tmp/SoftPosit && \\
    cd /tmp/SoftPosit && \\
    mkdir build && cd build && \\
    cmake .. && make && make install

# Install Python bindings for SoftPosit
RUN cd /tmp/SoftPosit/Python && python setup.py install

# Production stage
FROM --platform=linux/{target_arch} {arch_config['base_image']} as runtime

# Copy built dependencies
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt /opt

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY requirements.txt /app/requirements.txt
COPY src/ /app/src/
COPY scripts/ /app/scripts/

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set architecture-specific environment variables
ENV POSIT_ARCH={target_arch}
ENV PYTORCH_ARCH={target_arch}

# Expose ports for federated communication
EXPOSE 8080 8081

# Default command
CMD ["python", "-m", "src.federated.client"]
"""
        
        return dockerfile_content
    
    def build_multi_arch_image(self, image_name: str, 
                              architectures: List[str] = None) -> Dict[str, bool]:
        """
        Build multi-architecture Docker images.
        
        Args:
            image_name: Docker image name
            architectures: List of target architectures
            
        Returns:
            Build status for each architecture
        """
        if architectures is None:
            architectures = ['x86_64', 'arm64']
        
        build_results = {}
        
        for arch in architectures:
            logger.info(f"Building image for {arch}")
            
            try:
                # Generate architecture-specific Dockerfile
                dockerfile_content = self.generate_dockerfile(arch)
                
                # Write Dockerfile to temporary location
                dockerfile_path = f"Dockerfile.{arch}"
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                
                # Build image
                start_time = time.time()
                
                image, build_logs = self.client.images.build(
                    path=".",
                    dockerfile=dockerfile_path,
                    tag=f"{image_name}:{arch}",
                    platform=f"linux/{arch}",
                    pull=True,
                    rm=True
                )
                
                build_time = time.time() - start_time
                
                # Record metrics
                self.deployment_metrics.append({
                    'architecture': arch,
                    'build_time': build_time,
                    'image_size': image.attrs['Size'],
                    'success': True
                })
                
                build_results[arch] = True
                logger.info(f"Successfully built {image_name}:{arch} in {build_time:.2f}s")
                
                # Cleanup temporary Dockerfile
                os.remove(dockerfile_path)
                
            except Exception as e:
                logger.error(f"Failed to build image for {arch}: {str(e)}")
                build_results[arch] = False
                
                # Record failure metrics
                self.deployment_metrics.append({
                    'architecture': arch,
                    'build_time': 0,
                    'image_size': 0,
                    'success': False,
                    'error': str(e)
                })
        
        return build_results
    
    def deploy_federated_client(self, image_name: str, client_id: str, 
                               server_address: str) -> Optional[str]:
        """
        Deploy federated learning client container.
        
        Args:
            image_name: Docker image name
            client_id: Unique client identifier
            server_address: Federated server address
            
        Returns:
            Container ID if successful, None otherwise
        """
        arch_config = self.arch_configs[self.current_arch]
        
        container_name = f"federated-client-{client_id}-{self.current_arch}"
        
        try:
            # Run container with architecture-specific optimizations
            container = self.client.containers.run(
                f"{image_name}:{self.current_arch}",
                name=container_name,
                detach=True,
                environment={
                    'CLIENT_ID': client_id,
                    'SERVER_ADDRESS': server_address,
                    'ARCHITECTURE': self.current_arch,
                    'POSIT_ENABLED': 'true'
                },
                mem_limit=arch_config['resource_limits']['memory'],
                cpu_count=float(arch_config['resource_limits']['cpus']),
                network_mode='bridge',
                ports={'8080/tcp': None}  # Dynamic host port assignment
            )
            
            logger.info(f"Deployed federated client {client_id} on {self.current_arch}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to deploy client {client_id}: {str(e)}")
            return None
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """
        Get deployment metrics and statistics.
        
        Returns:
            Deployment metrics dictionary
        """
        if not self.deployment_metrics:
            return {}
        
        # Calculate success rates by architecture
        arch_metrics = {}
        for arch in ['x86_64', 'arm64']:
            arch_deployments = [m for m in self.deployment_metrics if m['architecture'] == arch]
            if arch_deployments:
                successful = sum(1 for m in arch_deployments if m['success'])
                arch_metrics[arch] = {
                    'success_rate': successful / len(arch_deployments) * 100,
                    'avg_build_time': sum(m['build_time'] for m in arch_deployments if m['success']) / max(successful, 1),
                    'total_deployments': len(arch_deployments)
                }
        
        return {
            'architecture_metrics': arch_metrics,
            'overall_success_rate': sum(1 for m in self.deployment_metrics if m['success']) / len(self.deployment_metrics) * 100,
            'current_architecture': self.current_arch
        }
    
    def cleanup_containers(self, prefix: str = "federated-client") -> int:
        """
        Cleanup federated learning containers.
        
        Args:
            prefix: Container name prefix to match
            
        Returns:
            Number of containers cleaned up
        """
        cleaned_count = 0
        
        try:
            containers = self.client.containers.list(all=True)
            
            for container in containers:
                if container.name.startswith(prefix):
                    try:
                        container.stop(timeout=10)
                        container.remove()
                        cleaned_count += 1
                        logger.info(f"Cleaned up container: {container.name}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {container.name}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        logger.info(f"Cleaned up {cleaned_count} containers")
        return cleaned_count


class DockerComposeManager:
    """
    Manages Docker Compose orchestration for federated learning experiments.
    """
    
    def __init__(self, compose_file: str = "docker-compose.yml"):
        """
        Initialize Docker Compose manager.
        
        Args:
            compose_file: Path to Docker Compose file
        """
        self.compose_file = compose_file
        
    def generate_compose_file(self, config: Dict[str, Any]) -> str:
        """
        Generate Docker Compose file for federated learning experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Docker Compose file content
        """
        num_clients = config.get('num_clients', 3)
        architectures = config.get('architectures', ['x86_64', 'arm64', 'x86_64'])
        
        compose_content = {
            'version': '3.8',
            'services': {
                'federated-server': {
                    'image': 'posit-federated-learning:x86_64',
                    'ports': ['8080:8080'],
                    'environment': {
                        'ROLE': 'server',
                        'NUM_CLIENTS': str(num_clients)
                    },
                    'networks': ['federated-net']
                }
            },
            'networks': {
                'federated-net': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Add client services
        for i in range(num_clients):
            arch = architectures[i % len(architectures)]
            client_name = f'federated-client-{i}'
            
            compose_content['services'][client_name] = {
                'image': f'posit-federated-learning:{arch}',
                'depends_on': ['federated-server'],
                'environment': {
                    'ROLE': 'client',
                    'CLIENT_ID': str(i),
                    'SERVER_ADDRESS': 'federated-server:8080',
                    'ARCHITECTURE': arch
                },
                'networks': ['federated-net']
            }
        
        return json.dumps(compose_content, indent=2)
    
    def run_experiment(self, config: Dict[str, Any]) -> bool:
        """
        Run federated learning experiment using Docker Compose.
        
        Args:
            config: Experiment configuration
            
        Returns:
            True if experiment completed successfully
        """
        try:
            # Generate compose file
            compose_content = self.generate_compose_file(config)
            
            with open(self.compose_file, 'w') as f:
                f.write(compose_content)
            
            # Start experiment
            logger.info("Starting federated learning experiment")
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            # Wait for experiment completion
            experiment_time = config.get('experiment_time', 300)  # 5 minutes default
            time.sleep(experiment_time)
            
            # Stop experiment
            subprocess.run(['docker-compose', 'down'], check=True)
            
            logger.info("Experiment completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker Compose command failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            return False
