#!/usr/bin/env python3
"""B4: Bitwise Reproducibility Experiment

Runs inside Docker containers (arm64 and x86_64) to test whether IEEE 754
float32 rounding differs across ISAs for the exact aggregation used in the
paper: sequential weighted sum  result += w_k * tensor_k.

Outputs a JSON object to stdout with per-scenario hashes and raw bytes for
later comparison by b4_compare.py.
"""

import hashlib
import json
import platform
import struct
import sys
import time

import torch


def get_environment_info():
    """Collect environment metadata for reproducibility."""
    return {
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_config": {
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        },
    }


def sha256_tensor(t: torch.Tensor) -> str:
    """SHA-256 of the raw float bytes of a contiguous tensor."""
    data = t.contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def tensor_to_hex(t: torch.Tensor) -> list[str]:
    """Convert first 20 elements to hex representations for debugging."""
    flat = t.contiguous().flatten()
    n = min(20, flat.numel())
    fmt = "<d" if t.dtype == torch.float64 else "<f"
    hexvals = []
    for i in range(n):
        raw = struct.pack(fmt, flat[i].item())
        hexvals.append(raw.hex())
    return hexvals


def sequential_weighted_sum(tensors: list[torch.Tensor],
                            weights: list[float],
                            dtype: torch.dtype) -> torch.Tensor:
    """Exact aggregation from the paper: result += w_k * tensor_k.

    This is the critical operation — sequential accumulation in the target
    dtype without any reordering or tree reduction.
    """
    result = torch.zeros_like(tensors[0], dtype=dtype)
    for w, t in zip(weights, tensors):
        result = result + (w * t.to(dtype))
    return result


def generate_client_tensors(seed: int, num_clients: int, shape: tuple,
                            dtype: torch.dtype) -> tuple[list[torch.Tensor], list[float]]:
    """Generate deterministic client tensors and weights from a fixed seed.

    Always generates in float32 first (proven ISA-identical), then casts to
    the target dtype. This ensures identical inputs across ARM64 and x86_64
    so that any output differences can only come from the aggregation itself.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    tensors = []
    for _ in range(num_clients):
        # Generate as float32 (ISA-identical), then cast
        t = torch.randn(shape, generator=gen, dtype=torch.float32)
        tensors.append(t.to(dtype))

    # Weights: generate as float32, normalize in float64 for precision
    raw_weights = torch.rand(num_clients, generator=gen, dtype=torch.float32)
    normalized = raw_weights.to(torch.float64) / raw_weights.to(torch.float64).sum()
    weights = normalized.tolist()

    return tensors, weights


def run_scenario(scenario_name: str, num_clients: int, shape: tuple,
                 dtype: torch.dtype, seeds: list[int]) -> dict:
    """Run one scenario across all seeds and collect results."""
    dtype_name = "float32" if dtype == torch.float32 else "float64"
    results = []

    for seed in seeds:
        tensors, weights = generate_client_tensors(seed, num_clients, shape, dtype)

        # Hash input tensors to detect RNG differences vs aggregation differences
        input_hasher = hashlib.sha256()
        for t in tensors:
            input_hasher.update(t.contiguous().numpy().tobytes())
        for w in weights:
            input_hasher.update(struct.pack("<d", w))
        input_hash = input_hasher.hexdigest()

        agg = sequential_weighted_sum(tensors, weights, dtype)

        result_hash = sha256_tensor(agg)
        hex_head = tensor_to_hex(agg)

        # Element-wise statistics
        flat = agg.flatten()
        results.append({
            "seed": seed,
            "input_hash": input_hash,
            "hash": result_hash,
            "hex_head_20": hex_head,
            "num_elements": flat.numel(),
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "min": flat.min().item(),
            "max": flat.max().item(),
            "sum": flat.to(torch.float64).sum().item(),
        })

    return {
        "scenario": scenario_name,
        "num_clients": num_clients,
        "shape": list(shape),
        "dtype": dtype_name,
        "num_seeds": len(seeds),
        "results": results,
    }


def main():
    torch.set_num_threads(1)  # Deterministic single-threaded execution

    seeds = list(range(42, 52))  # 10 seeds: 42..51

    # Model parameter shapes from AdaptiveCNN (ARM64 config: smaller model)
    model_shapes = {
        "conv1_weight": (32, 3, 3, 3),      # First conv layer
        "conv2_weight": (64, 32, 3, 3),      # Second conv layer
        "fc1_weight": (128, 256),            # First FC layer
        "fc1_bias": (128,),                  # Bias vector
    }

    scenarios = []

    # Scenario 1: 5 clients, float32 (main test)
    scenarios.append(("5c_f32", 5, (1000,), torch.float32))

    # Scenario 2: 10 clients, float32
    scenarios.append(("10c_f32", 10, (1000,), torch.float32))

    # Scenario 3: 5 clients, float64 (control — should always match)
    scenarios.append(("5c_f64", 5, (1000,), torch.float64))

    # Scenario 4: 10 clients, float64 (control)
    scenarios.append(("10c_f64", 10, (1000,), torch.float64))

    # Scenario 5: Actual model parameter shapes, float32
    for param_name, shape in model_shapes.items():
        scenarios.append((f"model_{param_name}_f32", 5, shape, torch.float32))

    env = get_environment_info()
    all_results = []

    t0 = time.time()
    for name, nc, shape, dtype in scenarios:
        r = run_scenario(name, nc, shape, dtype, seeds)
        all_results.append(r)
    elapsed = time.time() - t0

    output = {
        "experiment": "B4_bitwise_reproducibility",
        "environment": env,
        "elapsed_seconds": round(elapsed, 3),
        "scenarios": all_results,
    }

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
