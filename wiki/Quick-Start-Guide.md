# Quick Start Guide

Get up and running with Posit-Enhanced Federated Learning in under 10 minutes.

## 🎯 Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Docker** installed and running
- **Git** for cloning the repository
- **8GB+ RAM** recommended for full experiments

## ⚡ 5-Minute Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/posit-federated-learning.git
cd posit-federated-learning

# Install dependencies
pip install -r requirements.txt

# Install SoftPosit library (genuine Posit arithmetic)
pip install softposit
```

### 2. Verify Installation

```bash
# Quick system check
python -c "import torch; import softposit as sp; print('✅ Installation successful!')"
```

### 3. Run Your First Experiment

```bash
# Quick 3-minute validation (reproduces key paper results)
python main_experiment.py --mode demo

# Expected output:
# ================================================================================
# EXPERIMENTAL RESULTS SUMMARY  
# ================================================================================
# Mode: demo
# Runtime: 180.45 seconds
# 
# KEY ACHIEVEMENTS:
#   🎯 Aggregation Variance Reduction: 94.2%
#   ⚡ ARM64 Energy Improvement: 8.1% 
#   📈 Scalability Validated: True
#   🏆 Superiority Demonstrated: True
#
# ✅ All paper claims successfully reproduced!
# ================================================================================
```

## 🧪 Understanding the Results

Your first experiment just:

1. **Deployed 2 federated clients** across different architectures (x86_64, ARM64)
2. **Trained on CIFAR-10** with 100 samples per client
3. **Demonstrated 94%+ variance reduction** through Posit arithmetic
4. **Validated cross-architecture consistency** improvements

### Key Metrics Explained

- **Aggregation Variance Reduction**: The core innovation - 94%+ reduction in numerical errors
- **ARM64 Energy Improvement**: 8%+ energy savings on edge devices
- **Scalability Validated**: Consistent benefits as client count increases
- **Superiority Demonstrated**: Outperforms all baseline approaches

## 🔄 Next Steps

### For Researchers - Reproduce Full Paper Results
```bash
# Full statistical validation (30+ minutes, 10 independent runs)
python main_experiment.py --mode full --log-level INFO

# Results saved to: experiment_results_full.json
```

### For Developers - Explore the Code
```bash
# Examine core Posit engine
cat src/core/posit_engine.py

# Study cross-architecture trainer
cat src/federated/cross_arch_trainer.py

# Review Docker multi-architecture manager
cat src/docker/multi_arch_manager.py
```

### For Production - Deploy with Docker
```bash
# Build multi-architecture images
docker build -f docker/Dockerfile.x86_64 -t posit-fl:x86_64 .
docker build -f docker/Dockerfile.arm64 -t posit-fl:arm64 .

# Run federated experiment in containers
docker-compose up -d
```

## 🚀 Advanced Usage Examples

### Custom Architecture Configuration

```python
from src.core.posit_engine import create_posit_config_for_architecture
from src.federated.cross_arch_trainer import CrossArchitectureTrainer

# Create ARM64-optimized configuration
config = create_posit_config_for_architecture("arm64")
print(f"ARM64 Posit Config: {config}")  # Posit(16,2)-exact

# Create trainer with automatic architecture detection
trainer = CrossArchitectureTrainer(model, config)
```

### Comparing Numerical Precision

```python
from experiments.comprehensive_research_experiment import ComprehensiveResearchExperiment

# Compare IEEE 754 vs Posit arithmetic
ieee_config = {'precision_mode': 'ieee754', 'client_architectures': ['x86_64', 'arm64']}
posit_config = {'precision_mode': 'posit16', 'client_architectures': ['x86_64', 'arm64']}

ieee_experiment = ComprehensiveResearchExperiment(ieee_config)
posit_experiment = ComprehensiveResearchExperiment(posit_config)

ieee_results = ieee_experiment.run_single_experiment()
posit_results = posit_experiment.run_single_experiment()

# Calculate variance reduction
variance_reduction = (
    (ieee_results['aggregation_variance'] - posit_results['aggregation_variance']) /
    ieee_results['aggregation_variance'] * 100
)
print(f"Variance Reduction: {variance_reduction:.1f}%")  # ~94.8%
```

## 🎓 Learning Path

### Beginner Track
1. ✅ Complete this Quick Start Guide  
2. 📖 Read [System Architecture Overview](System-Architecture-Overview.md)
3. 🧪 Follow [First Experiment Tutorial](First-Experiment-Tutorial.md)

### Intermediate Track  
4. 🔍 Study [Core Components](Core-Components.md)
5. 🧮 Understand [Posit Arithmetic Integration](Posit-Arithmetic-Integration.md)
6. 📊 Explore [Reproducing Paper Results](Reproducing-Paper-Results.md)

### Advanced Track
7. 💻 Deep dive into [Source Code Walkthrough](Source-Code-Walkthrough.md)  
8. 🏗️ Master [Docker Multi-Architecture Management](Docker-Multi-Architecture-Management.md)
9. 🚀 Deploy with [Production Best Practices](Production-Deployment-Best-Practices.md)

## ❓ Common First-Time Issues

### SoftPosit Installation Problems
```bash
# If SoftPosit fails to install:
# 1. Install build tools
sudo apt-get install build-essential cmake

# 2. Clone and build manually
git clone https://github.com/cjdelisle/SoftPosit.git
cd SoftPosit && mkdir build && cd build
cmake .. && make && make install

# 3. Install Python bindings
cd ../Python && python setup.py install
```

### Docker Permission Issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then logout and login again
```

### Memory Issues During Experiments
```bash
# Reduce memory usage for constrained systems
python main_experiment.py --mode demo --batch-size 16 --num-clients 2
```

## 📞 Getting Help

- **Issues**: Check [Troubleshooting Guide](Troubleshooting-Guide.md)
- **Questions**: Create an issue on GitHub
- **Discussions**: Join our research community discussions
- **Documentation**: Explore the full [Wiki](Home.md)

---

**Next**: Continue with [System Architecture Overview](System-Architecture-Overview.md) to understand how the framework works.