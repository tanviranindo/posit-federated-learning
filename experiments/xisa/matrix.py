#!/usr/bin/env python3
"""Pairwise comparison of every platform in a results folder.

Finds each platform folder (one containing ops_t1.json), checks that all saw
identical inputs and initial weights, then writes MATRIX.md and matrix.json
with: per-platform self-checks, op categories that differ for each pair, and
FedAvg drift (test-prediction disagreement, parameter distance) for each pair.

Usage: python experiments/xisa/matrix.py [RESULTS_DIR] [--threads 1] [--seed 7]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from common import describe_cpu
from compare import diff_stats

CATEGORIES = [("ew_", "element-wise"), ("red_", "reductions"), ("tr_", "transcendental"),
              ("gemm_", "GEMM"), ("conv_", "conv"), ("bn_", "BatchNorm"),
              ("avgpool", "pooling"), ("model_", "model fwd/bwd")]


def category(op: str) -> str:
    return next(name for prefix, name in CATEGORIES if op.startswith(prefix))


def load_summary(run: Path) -> dict | None:
    p = run / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def traj_pair(run_a: Path, run_b: Path) -> dict:
    sa, sb = load_summary(run_a), load_summary(run_b)
    last = sa["history"][-1]["round"]
    with np.load(run_a / "preds.npz") as pa, np.load(run_b / "preds.npz") as pb:
        dis = {r: float((pa[f"r{r}"] != pb[f"r{r}"]).mean())
               for r in (1, 5, 10, last) if f"r{r}" in pa.files}
    rel = None
    prefix = f"global_r{last}/"
    with np.load(run_a / "states.npz") as xa, np.load(run_b / "states.npz") as xb:
        keys = [k for k in xa.files if k.startswith(prefix) and np.issubdtype(xa[k].dtype, np.floating)]
        if keys:
            va = np.concatenate([xa[k].ravel() for k in keys]).astype(np.float64)
            vb = np.concatenate([xb[k].ravel() for k in keys]).astype(np.float64)
            rel = float(np.linalg.norm(va - vb) / np.linalg.norm(va))
    return {"identical": sa["history"][-1]["state_hash"] == sb["history"][-1]["state_hash"],
            "pred_disagreement": dis, "param_rel_l2": rel,
            "acc": [sa["history"][-1]["accuracy"], sb["history"][-1]["accuracy"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[1] / "results" / "xisa")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    plats = sorted(p for p in args.results.iterdir() if (p / f"ops_t{args.threads}.json").exists())
    meta = {p.name: json.loads((p / f"ops_t{args.threads}.json").read_text()) for p in plats}
    run = f"traj_t{args.threads}_o{args.seed}"
    hashes = {p.name: meta[p.name]["input_hash"] for p in plats}
    inits = {p.name: load_summary(p / run)["history"][0]["state_hash"]
             for p in plats if load_summary(p / run)}
    if len(set(hashes.values())) != 1 or len(set(inits.values())) != 1:
        raise SystemExit(f"inputs differ across platforms: {hashes} {inits}")

    report = {"platforms": {p.name: meta[p.name]["env"] for p in plats},
              "self": {}, "ops": {}, "traj": {}}

    for p in plats:
        row = {}
        if (p / f"{run}_rep").exists():
            row["rerun_identical"] = traj_pair(p / run, p / f"{run}_rep")["identical"]
        multi = sorted(p.glob(f"traj_t*_o{args.seed}"))
        multi = [m for m in multi if m.name != run]
        if multi:
            row["threads_vs_1"] = {m.name: traj_pair(p / run, m) for m in multi}
        other = sorted(q for q in p.glob(f"traj_t{args.threads}_o*") if q.name not in (run, f"{run}_rep"))
        if other:
            row["seed_baseline"] = traj_pair(p / run, other[0])
        report["self"][p.name] = row

    for a, b in itertools.combinations(plats, 2):
        key = f"{a.name} | {b.name}"
        with np.load(a / f"ops_t{args.threads}.npz") as fa, np.load(b / f"ops_t{args.threads}.npz") as fb:
            stats = {k: diff_stats(fa[k], fb[k]) for k in fa.files}
        cats = sorted({category(k) for k, s in stats.items() if not s.get("identical", False)})
        report["ops"][key] = {"n_diff": sum(not s.get("identical", False) for s in stats.values()),
                              "n_ops": len(stats), "categories": cats, "per_op": stats}
        if (a / run).exists() and (b / run).exists():
            report["traj"][key] = traj_pair(a / run, b / run)

    (args.results / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# Cross-platform matrix", "",
             f"Threads {args.threads}, order seed {args.seed}; all platforms saw identical "
             "inputs and initial weights.", "", "## Platforms", "",
             "| Platform | CPU | OS | BLAS | SIMD |", "|---|---|---|---|---|"]
    for name, e in report["platforms"].items():
        lines.append(f"| {name} | {describe_cpu(e['cpu'])} | {e['system']} {e['machine']} | {e['blas']} | {e['cpu_capability']} |")
    lines += ["", "## Same-platform checks", "",
              "| Platform | Rerun identical | 1 vs N threads: disagreement at last round | Seed baseline |",
              "|---|---|---|---|"]
    for name, row in report["self"].items():
        thr = ", ".join(f"{k}: {100 * max(v['pred_disagreement'].items())[1]:.2f}%"
                        for k, v in row.get("threads_vs_1", {}).items()) or "-"
        seed = row.get("seed_baseline")
        seed_s = f"{100 * max(seed['pred_disagreement'].items())[1]:.2f}%" if seed else "-"
        lines.append(f"| {name} | {row.get('rerun_identical', '-')} | {thr} | {seed_s} |")
    lines += ["", "## Pairs", "",
              "| Pair | Ops differing | Categories | Pred. disagreement r1 / r10 / last | Param rel. L2 |",
              "|---|---|---|---|---|"]
    for key, o in report["ops"].items():
        t = report["traj"].get(key)
        if t:
            d = t["pred_disagreement"]
            r_last = max(d)
            dis = " / ".join(f"{100 * d[r]:.2f}%" for r in (1, 10, r_last) if r in d)
            rel = f"{t['param_rel_l2']:.1e}" if t["param_rel_l2"] is not None else "-"
        else:
            dis, rel = "-", "-"
        lines.append(f"| {key} | {o['n_diff']}/{o['n_ops']} | {', '.join(o['categories']) or 'none'} "
                     f"| {dis} | {rel} |")
    (args.results / "MATRIX.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
