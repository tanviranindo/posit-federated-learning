#!/usr/bin/env python3
"""H3: can a client's update be reproduced on another platform?

A replay verifier re-runs a client's local training from the same global state
and compares the result with the update the client reported. This script
replays the client updates recorded by fl_trajectory.py (possibly on another
platform) on the current platform and measures the mismatch, i.e. the
tolerance a cross-platform verifier would have to allow.

Usage: python experiments/xisa/replay.py SOURCE_RUN_DIR [--threads 1]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from common import (env_info, load_cifar, platform_tag, REPO_ROOT, results_dir,
                    setup_torch, write_json)
from compare import diff_stats
from fl_trajectory import client_indices, local_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="trajectory run folder to replay")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    src = json.loads((args.source / "summary.json").read_text())
    if src["input_hash"] != data["input_hash"]:
        raise SystemExit("input hash differs from the source run")
    sargs = src["args"]
    parts = client_indices(len(data["y_train"]), sargs["clients"])

    with np.load(args.source / "states.npz") as z:
        files = z.files
        jobs = sorted({(int(m.group(1)), int(m.group(2)))
                       for f in files if (m := re.match(r"client(\d+)_r(\d+)/", f))},
                      key=lambda kr: (kr[1], kr[0]))
        results = []
        for k, r in jobs:
            g_prefix, c_prefix = f"global_r{r - 1}/", f"client{k}_r{r}/"
            glob = {f[len(g_prefix):]: z[f] for f in files if f.startswith(g_prefix)}
            recorded = {f[len(c_prefix):]: z[f] for f in files if f.startswith(c_prefix)}
            replayed = local_train(glob, data, parts[k], order_seed=sargs["order_seed"],
                                   round_idx=r, client=k, epochs=sargs["epochs"],
                                   batch=sargs["batch"], lr=sargs["lr"])
            keys = [key for key in recorded if np.issubdtype(recorded[key].dtype, np.floating)]
            rec = np.concatenate([recorded[key].ravel() for key in keys]).astype(np.float64)
            rep = np.concatenate([replayed[key].ravel() for key in keys]).astype(np.float64)
            g = np.concatenate([glob[key].ravel() for key in keys]).astype(np.float64)
            update_norm = float(np.linalg.norm(rec - g))
            row = {
                "client": k, "round": r,
                "identical": all(np.array_equal(recorded[key], replayed[key]) for key in recorded),
                "update_norm": update_norm,
                "mismatch_l2": float(np.linalg.norm(rep - rec)),
                "mismatch_rel_to_update": float(np.linalg.norm(rep - rec) / update_norm),
                "mismatch_linf": float(np.abs(rep - rec).max()),
                "per_tensor": {key: diff_stats(recorded[key], replayed[key]) for key in keys},
            }
            results.append(row)
            print(f"client {k} round {r:3d}: identical={row['identical']} "
                  f"rel_to_update={row['mismatch_rel_to_update']:.3e} "
                  f"linf={row['mismatch_linf']:.3e}", flush=True)

    out_dir = results_dir(args.out, args.tag)
    name = f"replay_t{args.threads}_from_{args.source.parent.name}.{args.source.name}.json"
    write_json(out_dir / name, {
        "verifier_env": env_info(),
        "verifier_platform": args.tag or platform_tag(),
        "source_env": src["env"],
        "source_run": f"{args.source.parent.name}/{args.source.name}",
        "results": results,
    })
    print(f"report: {out_dir / name}")


if __name__ == "__main__":
    main()
