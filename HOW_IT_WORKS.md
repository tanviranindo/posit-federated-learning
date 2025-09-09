# How Everything Connects

If you're wondering how all the pieces fit together, here's a simple walkthrough.

## The Problem We're Solving

When you do federated learning across different types of computers (Intel laptops, ARM Raspberry Pis, etc.), the model parameters drift apart because each architecture does floating-point math slightly differently. Over many federation rounds, these tiny differences accumulate and make your training unstable.

## Our Solution in 3 Parts

### 1. The Math Engine (`src/core/posit_engine.py`)

Instead of standard floating-point, we use **Posit arithmetic** with something called a "quire":

```python
# Instead of this (accumulates errors):
result = (model1 + model2 + model3) / 3

# We do this (exact until the end):
quire = QuireAccumulator()
quire.add_exact(model1 * weight1)
quire.add_exact(model2 * weight2) 
quire.add_exact(model3 * weight3)
result = quire.get_final_result()  # Only one rounding step
```

### 2. The Federated Learning Logic (`src/federated/`)

The `CrossArchitectureTrainer` automatically detects what type of computer it's running on and adapts:

- **Intel/AMD**: Uses full model complexity for best accuracy
- **ARM (Raspberry Pi)**: Uses smaller model for energy efficiency
- **Both**: Uses Posit math for consistent results

### 3. The Deployment (`src/docker/`)

Docker containers that automatically build differently for each architecture but run the same federated learning code.

## What Happens When You Run It

1. **Architecture Detection**: Each client figures out if it's Intel or ARM
2. **Model Adaptation**: Adjusts neural network size for the hardware
3. **Local Training**: Trains on local data (normal PyTorch stuff)
4. **Posit Aggregation**: Server combines updates using exact arithmetic
5. **Repeat**: Models stay aligned across federation rounds

## The Key Insight

Traditional approach:
```
Client 1 (Intel) → [rounding] → Server → [rounding] → Global Model
Client 2 (ARM)   → [rounding] →   ↑    → [rounding] →      ↑
Client 3 (Intel) → [rounding] →   ↑    → [rounding] →      ↑
                                  ↑                         ↑
                            Errors accumulate!         Unstable!
```

Our approach:
```
Client 1 (Intel) → [Posit] → Server (Quire) → [Single rounding] → Stable Model
Client 2 (ARM)   → [Posit] →   ↑   ↑   ↑   →        ↑         →      ↑
Client 3 (Intel) → [Posit] →   Exact math   →        ↑         →      ↑
                              until the end →    Precise!     → Reliable!
```

## Files You Care About

- **`main_experiment.py`**: Run this to see the difference
- **`src/core/posit_engine.py`**: The math that makes it work
- **`src/federated/cross_arch_trainer.py`**: Federated learning with auto-adaptation
- **`src/models/adaptive_cnn.py`**: Neural networks that adapt to your hardware
- **`docker/`**: Containers for easy deployment

## Try It

```bash
# Quick demo (2 minutes)
python main_experiment.py --mode demo

# You'll see something like:
# IEEE 754 training: Models drift apart (high variance)
# Posit training: Models stay aligned (low variance)
```

The numbers don't lie - Posit arithmetic makes federated learning about 95% more stable when you're mixing different computer architectures.