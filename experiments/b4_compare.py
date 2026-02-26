#!/usr/bin/env python3
"""B4: Comparison Script

Reads the JSON outputs from arm64 and x86_64 Docker runs and reports:
- Hash match/mismatch per scenario per seed
- Element-wise max absolute difference (via hex → float reconstruction)
- Number of differing elements (ULP analysis)
- Statistical summary across seeds

Usage:
    python b4_compare.py b4_arm64.json b4_x86.json [--output b4_report.json]
"""

import argparse
import json
import struct
import sys


def hex_to_float(h: str) -> float:
    """Convert a little-endian hex string to a Python float."""
    raw = bytes.fromhex(h)
    if len(raw) == 8:
        return struct.unpack("<d", raw)[0]
    return struct.unpack("<f", raw)[0]


def ulp_distance(a_hex: str, b_hex: str) -> int:
    """Compute ULP distance between two floats from their hex representations."""
    raw_a = bytes.fromhex(a_hex)
    raw_b = bytes.fromhex(b_hex)
    if len(raw_a) == 8:  # float64
        ua = struct.unpack("<Q", raw_a)[0]
        ub = struct.unpack("<Q", raw_b)[0]
        sign_bit = 1 << 63
        mask = sign_bit - 1
    else:  # float32
        ua = struct.unpack("<I", raw_a)[0]
        ub = struct.unpack("<I", raw_b)[0]
        sign_bit = 1 << 31
        mask = sign_bit - 1
    if ua & sign_bit:
        ua = sign_bit - (ua & mask)
    if ub & sign_bit:
        ub = sign_bit - (ub & mask)
    return abs(ua - ub)


def compare_scenario(arm64_scenario: dict, x86_scenario: dict) -> dict:
    """Compare a single scenario across both platforms."""
    name = arm64_scenario["scenario"]
    dtype = arm64_scenario["dtype"]
    num_seeds = arm64_scenario["num_seeds"]

    seed_comparisons = []
    total_matching_hashes = 0
    total_matching_inputs = 0

    for arm_result, x86_result in zip(arm64_scenario["results"],
                                       x86_scenario["results"]):
        seed = arm_result["seed"]
        hash_match = arm_result["hash"] == x86_result["hash"]
        if hash_match:
            total_matching_hashes += 1

        # Compare input hashes to distinguish RNG vs aggregation differences
        input_match = (arm_result.get("input_hash") == x86_result.get("input_hash"))
        if input_match:
            total_matching_inputs += 1

        # Compare hex head values for element-wise analysis
        arm_hex = arm_result["hex_head_20"]
        x86_hex = x86_result["hex_head_20"]
        n = min(len(arm_hex), len(x86_hex))

        max_abs_diff = 0.0
        max_ulp_diff = 0
        num_differing = 0
        diffs = []

        for i in range(n):
            if arm_hex[i] != x86_hex[i]:
                num_differing += 1
                a = hex_to_float(arm_hex[i])
                b = hex_to_float(x86_hex[i])
                abs_diff = abs(a - b)
                ulp_diff = ulp_distance(arm_hex[i], x86_hex[i])
                max_abs_diff = max(max_abs_diff, abs_diff)
                max_ulp_diff = max(max_ulp_diff, ulp_diff)
                if len(diffs) < 5:  # Keep first 5 examples
                    diffs.append({
                        "index": i,
                        "arm64_hex": arm_hex[i],
                        "x86_hex": x86_hex[i],
                        "arm64_val": a,
                        "x86_val": b,
                        "abs_diff": abs_diff,
                        "ulp_diff": ulp_diff,
                    })

        # Aggregate statistics comparison
        mean_diff = abs(arm_result["mean"] - x86_result["mean"])
        sum_diff = abs(arm_result["sum"] - x86_result["sum"])

        seed_comparisons.append({
            "seed": seed,
            "input_match": input_match,
            "hash_match": hash_match,
            "arm64_hash": arm_result["hash"][:16] + "...",
            "x86_hash": x86_result["hash"][:16] + "...",
            "head_elements_compared": n,
            "head_elements_differing": num_differing,
            "max_abs_diff": max_abs_diff,
            "max_ulp_diff": max_ulp_diff,
            "mean_diff": mean_diff,
            "sum_diff": sum_diff,
            "example_diffs": diffs,
        })

    all_match = total_matching_hashes == num_seeds
    any_match = total_matching_hashes > 0

    return {
        "scenario": name,
        "dtype": dtype,
        "num_clients": arm64_scenario["num_clients"],
        "shape": arm64_scenario["shape"],
        "num_seeds": num_seeds,
        "inputs_matching": total_matching_inputs,
        "hashes_matching": total_matching_hashes,
        "all_hashes_match": all_match,
        "any_hash_matches": any_match,
        "seed_comparisons": seed_comparisons,
    }


def print_summary(comparisons: list[dict], arm64_env: dict, x86_env: dict):
    """Print a human-readable summary table."""
    print("=" * 78)
    print("B4: BITWISE REPRODUCIBILITY — CROSS-ISA COMPARISON")
    print("=" * 78)
    print()

    # Environment
    print("ENVIRONMENTS:")
    print(f"  ARM64:  {arm64_env['platform_machine']}, "
          f"Python {arm64_env['python_version']}, "
          f"PyTorch {arm64_env['torch_version']}")
    print(f"  x86_64: {x86_env['platform_machine']}, "
          f"Python {x86_env['python_version']}, "
          f"PyTorch {x86_env['torch_version']}")
    print()

    # Summary table
    hdr = f"{'Scenario':<28} {'dtype':<8} {'Clients':>7} {'Shape':<16} {'Input':>7} {'Output':>7} {'MaxAbsDiff':>12} {'MaxULP':>8}"
    print(hdr)
    print("-" * len(hdr))

    f32_scenarios = []
    f64_scenarios = []

    for c in comparisons:
        name = c["scenario"]
        dtype = c["dtype"]
        nc = c["num_clients"]
        shape = "x".join(str(s) for s in c["shape"])

        input_str = f"{c['inputs_matching']}/{c['num_seeds']}"
        output_str = f"{c['hashes_matching']}/{c['num_seeds']}"

        # Aggregate max diffs across all seeds
        max_abs = max(s["max_abs_diff"] for s in c["seed_comparisons"])
        max_ulp = max(s["max_ulp_diff"] for s in c["seed_comparisons"])

        if max_abs == 0:
            abs_str = "0 (exact)"
            ulp_str = "0"
        else:
            abs_str = f"{max_abs:.2e}"
            ulp_str = str(max_ulp)

        print(f"{name:<28} {dtype:<8} {nc:>7} {shape:<16} {input_str:>7} {output_str:>7} {abs_str:>12} {ulp_str:>8}")

        if dtype == "float32":
            f32_scenarios.append(c)
        else:
            f64_scenarios.append(c)

    print()

    # Verdict
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    # Check float64 control — both inputs and outputs
    f64_all_inputs_match = all(c["inputs_matching"] == c["num_seeds"]
                               for c in f64_scenarios) if f64_scenarios else True
    f64_all_outputs_match = all(c["all_hashes_match"] for c in f64_scenarios)

    f32_all_inputs_match = all(c["inputs_matching"] == c["num_seeds"]
                               for c in f32_scenarios)
    f32_any_output_differ = any(not c["all_hashes_match"] for c in f32_scenarios)

    # Float64 control
    if f64_all_outputs_match:
        print("[PASS] Float64 control: ALL output hashes match across ISAs.")
        print("       Test infrastructure is correct.")
    elif not f64_all_inputs_match:
        print("[INFO] Float64 control: Output hashes differ, but INPUT hashes also")
        print("       differ — torch.randn(dtype=float64) produces different values")
        print("       across ISAs. This is an RNG difference, not an aggregation")
        print("       difference. The control is still informative.")
    else:
        print("[WARN] Float64 control: Output hashes differ despite identical inputs.")
        print("       Float64 aggregation itself differs across ISAs.")

    print()

    # Float32 results
    if not f32_all_inputs_match:
        differing_inputs = sum(c["num_seeds"] - c["inputs_matching"]
                               for c in f32_scenarios)
        total = sum(c["num_seeds"] for c in f32_scenarios)
        print(f"[INFO] Float32 inputs: {differing_inputs}/{total} scenario-seeds have")
        print("       different RNG outputs. This means torch.randn differs across ISAs.")
        print()

    if f32_any_output_differ:
        total_f32_seeds = sum(c["num_seeds"] for c in f32_scenarios)
        differing_f32 = sum(c["num_seeds"] - c["hashes_matching"]
                           for c in f32_scenarios)
        print(f"[RESULT] Float32: {differing_f32}/{total_f32_seeds} scenario-seeds "
              f"produced DIFFERENT output hashes across ISAs.")

        if f32_all_inputs_match:
            print("         Inputs are IDENTICAL — the aggregation itself differs.")
            print("         IEEE 754 float32 rounding DIFFERS between ARM64 and x86_64")
            print("         for sequential weighted-sum aggregation in PyTorch.")
            print("         This validates the paper's numerical-drift premise.")
        else:
            print("         Note: Input hashes also differ, so the output differences")
            print("         may be caused by RNG differences, not aggregation rounding.")

        # Aggregate ULP stats
        all_ulps = []
        for c in f32_scenarios:
            for s in c["seed_comparisons"]:
                if s["max_ulp_diff"] > 0:
                    all_ulps.append(s["max_ulp_diff"])
        if all_ulps:
            print(f"         Max ULP difference: {max(all_ulps)}")
            print(f"         Median ULP difference: {sorted(all_ulps)[len(all_ulps)//2]}")
    else:
        if f32_all_inputs_match:
            print("[RESULT] Float32: ALL output hashes match across ISAs, with")
            print("         identical inputs. PyTorch CPU kernels produce bitwise-")
            print("         identical float32 results on both ARM64 and x86_64 for")
            print("         sequential weighted sums.")
            print()
            print("         IMPLICATION: The cross-arch numerical drift in the paper")
            print("         does NOT come from aggregation rounding differences.")
            print("         Likely sources: BLAS library differences, BatchNorm")
            print("         running statistics, gradient computation, or training-")
            print("         time non-determinism.")
        else:
            print("[RESULT] Float32: ALL output hashes match despite some input")
            print("         hash differences. This needs further investigation —")
            print("         the matching outputs may be coincidental.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare B4 bitwise reproducibility results from two ISAs")
    parser.add_argument("arm64_json", help="Path to ARM64 JSON output")
    parser.add_argument("x86_json", help="Path to x86_64 JSON output")
    parser.add_argument("--output", "-o", help="Save structured report as JSON")
    args = parser.parse_args()

    with open(args.arm64_json) as f:
        arm64_data = json.load(f)
    with open(args.x86_json) as f:
        x86_data = json.load(f)

    arm64_env = arm64_data["environment"]
    x86_env = x86_data["environment"]

    # Match scenarios by name
    arm64_by_name = {s["scenario"]: s for s in arm64_data["scenarios"]}
    x86_by_name = {s["scenario"]: s for s in x86_data["scenarios"]}

    common = sorted(set(arm64_by_name.keys()) & set(x86_by_name.keys()))
    if not common:
        print("ERROR: No matching scenarios found between the two files.",
              file=sys.stderr)
        sys.exit(1)

    comparisons = []
    for name in common:
        c = compare_scenario(arm64_by_name[name], x86_by_name[name])
        comparisons.append(c)

    print_summary(comparisons, arm64_env, x86_env)

    if args.output:
        report = {
            "experiment": "B4_bitwise_comparison",
            "arm64_environment": arm64_env,
            "x86_environment": x86_env,
            "comparisons": comparisons,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Structured report saved to: {args.output}")


if __name__ == "__main__":
    main()
