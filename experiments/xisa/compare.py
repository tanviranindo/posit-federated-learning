#!/usr/bin/env python3
"""Compare pilot outputs from two platforms (or two runs on one platform).

  python experiments/xisa/compare.py ops  DIR_A DIR_B [--threads 1]
  python experiments/xisa/compare.py traj RUN_A RUN_B

DIR_* are platform result folders (containing ops_t*.npz); RUN_* are trajectory
run folders (containing summary.json, preds.npz, states.npz). A JSON report is
written next to RUN_A / DIR_A unless --out is given.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def ordered_bits(a: np.ndarray) -> np.ndarray:
    """Map floats to integers whose difference is the ULP distance."""
    if a.dtype == np.float32:
        u = a.view(np.int32).astype(np.int64)
        return np.where(u < 0, -(u & 0x7FFFFFFF), u)
    if a.dtype == np.float64:
        u = a.view(np.int64)
        return np.where(u < 0, -(u & 0x7FFFFFFFFFFFFFFF), u)
    raise TypeError(a.dtype)


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"error": f"shape/dtype mismatch {a.shape}{a.dtype} vs {b.shape}{b.dtype}"}
    if not np.issubdtype(a.dtype, np.floating):
        neq = a != b
        return {"identical": not neq.any(), "frac_diff": float(neq.mean())}
    ulp = np.abs(ordered_bits(a) - ordered_bits(b))
    differs = ulp > 0
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    denom = np.linalg.norm(a64)
    return {
        "identical": not differs.any(),
        "frac_diff": float(differs.mean()),
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        "median_ulp_of_diff": float(np.median(ulp[differs])) if differs.any() else 0.0,
        "rel_l2": float(np.linalg.norm(a64 - b64) / denom) if denom > 0 else 0.0,
        "max_abs": float(np.abs(a64 - b64).max()) if a.size else 0.0,
    }


def compare_ops(dir_a: Path, dir_b: Path, threads: int) -> dict:
    meta_a = json.loads((dir_a / f"ops_t{threads}.json").read_text())
    meta_b = json.loads((dir_b / f"ops_t{threads}.json").read_text())
    if meta_a["input_hash"] != meta_b["input_hash"]:
        raise SystemExit("input hashes differ: the platforms did not see identical data")
    with np.load(dir_a / f"ops_t{threads}.npz") as fa, np.load(dir_b / f"ops_t{threads}.npz") as fb:
        ops = {k: diff_stats(fa[k], fb[k]) for k in fa.files}
    print(f"{'op':34s} {'same':>5s} {'%diff':>8s} {'maxULP':>9s} {'rel_l2':>10s}")
    for k, s in ops.items():
        if "error" in s:
            print(f"{k:34s} {s['error']}")
            continue
        print(f"{k:34s} {str(s['identical']):>5s} {100 * s['frac_diff']:8.3f} "
              f"{s.get('max_ulp', 0):9d} {s.get('rel_l2', 0):10.2e}")
    return {"a": meta_a["env"], "b": meta_b["env"], "threads": threads, "ops": ops}


def compare_traj(run_a: Path, run_b: Path) -> dict:
    sa = json.loads((run_a / "summary.json").read_text())
    sb = json.loads((run_b / "summary.json").read_text())
    if sa["input_hash"] != sb["input_hash"]:
        raise SystemExit("input hashes differ: the runs did not see identical data")
    rounds = []
    with np.load(run_a / "preds.npz") as pa, np.load(run_b / "preds.npz") as pb, \
            np.load(run_a / "states.npz") as xa, np.load(run_b / "states.npz") as xb:
        for ha, hb in zip(sa["history"], sb["history"]):
            r = ha["round"]
            row = {
                "round": r,
                "same_state": ha["state_hash"] == hb["state_hash"],
                "pred_disagreement": float((pa[f"r{r}"] != pb[f"r{r}"]).mean()),
                "acc_a": ha["accuracy"], "acc_b": hb["accuracy"],
            }
            prefix = f"global_r{r}/"
            keys = [k for k in xa.files if k.startswith(prefix)
                    and np.issubdtype(xa[k].dtype, np.floating)]
            if keys:
                va = np.concatenate([xa[k].ravel() for k in keys]).astype(np.float64)
                vb = np.concatenate([xb[k].ravel() for k in keys]).astype(np.float64)
                row["param_rel_l2"] = float(np.linalg.norm(va - vb) / np.linalg.norm(va))
            rounds.append(row)
    print(f"{'round':>5s} {'same':>5s} {'pred_dis%':>9s} {'acc_a':>7s} {'acc_b':>7s} {'param_rel_l2':>12s}")
    for row in rounds:
        rel = f"{row['param_rel_l2']:.2e}" if "param_rel_l2" in row else ""
        print(f"{row['round']:5d} {str(row['same_state']):>5s} {100 * row['pred_disagreement']:9.2f} "
              f"{row['acc_a']:7.4f} {row['acc_b']:7.4f} {rel:>12s}")
    return {"a": sa["env"], "b": sb["env"], "args_a": sa["args"], "args_b": sb["args"],
            "rounds": rounds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ops", "traj"])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "ops":
        report = compare_ops(args.a, args.b, args.threads)
        default = args.a.parent / f"cmp_ops_t{args.threads}_{args.a.name}__{args.b.name}.json"
    else:
        report = compare_traj(args.a, args.b)
        default = args.a.parent.parent / (
            f"cmp_traj_{args.a.parent.name}.{args.a.name}__{args.b.parent.name}.{args.b.name}.json")
    out = args.out or default
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
