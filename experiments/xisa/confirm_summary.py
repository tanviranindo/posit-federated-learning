#!/usr/bin/env python3
"""Confirmation C2: score the full-test-set attack runs.

For each attack budget (one --tau-schedule), reports per seed and as a mean
with a 95% t-interval: the round-20 flip gap (targeted - random cat->dog
rate), the accuracy gap (random - targeted) and the accuracy drop of the
targeted run vs honest. Pass rule (pre-registered): flip gap >= 1 pp in every
seed and the interval of the mean flip gap above 0. Budgets are named from
tau.json when present.

Usage: python experiments/xisa/confirm_summary.py [RESULTS_DIR] [--tau TAU_JSON]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THRESHOLD = 0.01
STEALTH = 0.02
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def mean_ci(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, m - h, m + h


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results" / "xisa"
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=root / "confirm")
    ap.add_argument("--tau", type=Path, default=root / "tau.json")
    args = ap.parse_args()

    names = {}
    if args.tau.exists():
        names = {v: k for k, v in json.loads(args.tau.read_text())["schedules"].items()}

    runs = defaultdict(list)
    for p in sorted(args.results.glob("attack_*.json")):
        d = json.loads(p.read_text())
        runs[d["args"].get("tau_schedule") or str(d["args"]["tau"])].append(d)

    lines = ["# Confirmation C2: hidden attack on the full test set", "",
             "Round-20 values in pp. flip gap = targeted - random cat->dog rate; acc gap = "
             "random - targeted accuracy; drop = honest - targeted accuracy.", "",
             "| Budget | Seeds | flip gap per seed | flip gap mean [95% CI] | acc gap mean [95% CI] "
             "| drop mean | cat->dog honest / random / targeted | Pass | Stealth |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for sched, ds in sorted(runs.items()):
        ds.sort(key=lambda d: d["args"]["order_seed"])
        last = [d["rounds"][-1] for d in ds]
        flip = [r["src_to_tgt_targeted"] - r["src_to_tgt_random"] for r in last]
        acc = [r["acc_random"] - r["acc_targeted"] for r in last]
        drop = [r["acc_honest"] - r["acc_targeted"] for r in last]
        fm, fl, fh = mean_ci(flip)
        am, al, ah = mean_ci(acc)
        rate = {m: sum(r[f"src_to_tgt_{m}"] for r in last) / len(last)
                for m in ("honest", "random", "targeted")}
        ok = len(ds) >= 5 and all(f >= THRESHOLD for f in flip) and fl > 0
        stealth = sum(drop) / len(drop) <= STEALTH
        name = names.get(sched, sched)
        summary[name] = {"schedule": sched, "seeds": [d["args"]["order_seed"] for d in ds],
                         "flip_gap": flip, "flip_gap_ci": [fm, fl, fh], "acc_gap": acc,
                         "acc_gap_ci": [am, al, ah], "acc_drop_vs_honest": drop,
                         "src_to_tgt_mean": rate, "pass": ok, "stealth": stealth}
        lines.append(
            f"| {name} (`{sched}`) | {len(ds)} | " + ", ".join(f"{100 * f:+.1f}" for f in flip)
            + f" | {100 * fm:+.1f} [{100 * fl:+.1f}, {100 * fh:+.1f}] | "
            f"{100 * am:+.2f} [{100 * al:+.2f}, {100 * ah:+.2f}] | {100 * sum(drop) / len(drop):+.2f} | "
            f"{100 * rate['honest']:.1f} / {100 * rate['random']:.1f} / {100 * rate['targeted']:.1f} | "
            + ("**yes**" if ok else "no") + " | " + ("yes" if stealth else "no") + " |")
    (args.results / "CONFIRM.md").write_text("\n".join(lines) + "\n")
    (args.results / "confirm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
