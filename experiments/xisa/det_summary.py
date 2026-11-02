#!/usr/bin/env python3
"""Score the mitigation runs (M1 reproducibility, M2 overhead, M3 accuracy parity).

Usage: python experiments/xisa/det_summary.py [RESULTS_DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tau_summary import cpu


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[1] / "results" / "xisa")
    args = ap.parse_args()
    root = args.results

    runs = sorted(root.glob("*/det_t1_o7_e5/summary.json"))
    sigs, preds = [], []
    lines = ["# Mitigation: exact-accumulation stack", "",
             "## M1 reproducibility and M2 overhead (5 local epochs, 3 rounds, seed 7)", "",
             "| Platform | CPU | BLAS | Same hashes as first | std s/epoch | det s/epoch | ratio |",
             "|---|---|---|---|---|---|---|"]
    for p in runs:
        d = load(p)
        bench_p = p.parent.parent / "det_t1_o7_e5_bench" / "summary.json"
        b = load(bench_p)["bench"] if bench_p.exists() else d["bench"]
        sig = [(h["state_hash"], h["client_hashes"], h["logits_hash"]) for h in d["history"]]
        sigs.append(sig)
        with np.load(p.parent / "preds.npz") as z:
            preds.append({k: z[k] for k in z.files})
        lines.append(f"| {p.parent.parent.name} | {cpu(d['env'])} | {d['env']['blas']} | "
                     f"{'yes' if sig == sigs[0] else '**no**'} | {b['standard']:.2f} | "
                     f"{b['deterministic']:.2f} | {b['ratio']:.1f}x |")
    m1 = all(s == sigs[0] for s in sigs) and all(
        all(np.array_equal(preds[0][k], q[k]) for k in preds[0]) for q in preds)
    lines += ["", f"M1 (every client, global and logits hash identical on {len(runs)} platforms, "
              f"all rounds): **{'pass' if m1 else 'fail'}**. Replay tolerance is therefore 0.",
              "Local ratios use an idle-machine re-run (`det_t1_o7_e5_bench`) where present.", "",
              "## M3 accuracy parity (1 local epoch, 20 rounds, M2 Pro)", "",
              "| Seed | standard | deterministic | difference (pp) |", "|---|---|---|---|"]
    plat = root / "darwin-arm64-accelerate"
    std = {s: load(plat / f"traj_t1_o{s}" / "summary.json")["history"][-1]["accuracy"] for s in (7, 8)}
    det = {s: load(plat / f"det_t1_o{s}_e1" / "summary.json")["history"][-1]["accuracy"] for s in (7, 8)}
    bound = max(0.02, abs(std[7] - std[8]))
    for s in (7, 8):
        lines.append(f"| {s} | {100 * std[s]:.2f} | {100 * det[s]:.2f} | {100 * (det[s] - std[s]):+.2f} |")
    m3 = all(abs(det[s] - std[s]) <= bound for s in (7, 8))
    lines += ["", f"M3 (|difference| <= {100 * bound:.2f} pp): **{'pass' if m3 else 'fail'}**"]
    (root / "MITIGATION.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
