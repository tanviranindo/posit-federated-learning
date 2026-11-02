#!/usr/bin/env python3
"""Confirmation C1: replay tolerance for every (verifier, source) pair.

Reads the replay reports of one trajectory label (default: the 5-epoch run)
from every platform folder, tabulates the honest mismatch per round, checks
the pre-registered C1 rule, and derives the C2 attack budgets:

  B_min(r)  smallest honest mismatch at round r over all cross-ISA pairs and
            replayed clients, rounded down to 0.1 pp
  B_half(r) B_min(r) / 2

Writes TAU.md and tau.json to the results folder and prints the budgets as
--tau-schedule strings for hidden_attack.py.

Usage: python experiments/xisa/tau_summary.py [RESULTS_DIR] [--run traj_t1_o7_e5]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PASS_MISMATCH = 0.10   # C1: cross-ISA mismatch at rounds >= 3
PASS_FROM_ROUND = 3


def isa(env: dict) -> str:
    return {"aarch64": "arm64", "AMD64": "x86_64"}.get(env["machine"], env["machine"])


def floor_01pp(x: float) -> float:
    return math.floor(x * 1000) / 1000


def schedule(budget: dict[int, float]) -> str:
    return ",".join(f"{r}:{budget[r]:g}" for r in sorted(budget))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[1] / "results" / "xisa")
    ap.add_argument("--run", default="traj_t1_o7_e5")
    args = ap.parse_args()

    pairs = []
    for p in sorted(args.results.glob(f"*/replay_t1_from_*.{args.run}.json")):
        d = json.loads(p.read_text())
        rows = defaultdict(list)
        for row in d["results"]:
            rows[row["round"]].append(row)
        pairs.append({
            "verifier": p.parent.name, "source": d["source_run"].split("/")[0],
            "isa": (isa(d["verifier_env"]), isa(d["source_env"])),
            "identical": all(row["identical"] for row in d["results"]),
            "rounds": {r: [row["mismatch_rel_to_update"] for row in rs]
                       for r, rs in sorted(rows.items())},
        })
    if not pairs:
        raise SystemExit(f"no replay reports for {args.run} under {args.results}")

    self_pairs = [q for q in pairs if q["verifier"] == q["source"]]
    cross = [q for q in pairs if q["isa"][0] != q["isa"][1]]
    rounds = sorted({r for q in pairs for r in q["rounds"]})

    b_min = {}
    for r in rounds:
        vals = [v for q in cross for v in q["rounds"].get(r, [])]
        if vals:
            b_min[r] = floor_01pp(min(vals))
    b_half = {r: floor_01pp(v / 2) for r, v in b_min.items()}
    c1_fail = [(q["verifier"], q["source"], r, v) for q in cross
               for r, vs in q["rounds"].items() if r >= PASS_FROM_ROUND
               for v in vs if v < PASS_MISMATCH]
    c1_pass = bool(cross) and not c1_fail

    lines = [f"# Replay tolerance ({args.run})", "",
             "Honest mismatch ||replayed - reported|| / ||update|| per round, min-max over "
             "replayed clients (%).", "",
             "| Verifier | Source | ISA | Identical | " + " | ".join(f"r{r}" for r in rounds) + " |",
             "|---|---|---|---|" + "---|" * len(rounds)]
    for q in sorted(pairs, key=lambda q: (q["isa"][0] == q["isa"][1], q["verifier"], q["source"])):
        cells = [f"{100 * min(q['rounds'][r]):.1f}-{100 * max(q['rounds'][r]):.1f}"
                 if r in q["rounds"] else "" for r in rounds]
        lines.append(f"| {q['verifier']} | {q['source']} | {q['isa'][0]} <- {q['isa'][1]} | "
                     f"{'yes' if q['identical'] else 'no'} | " + " | ".join(cells) + " |")
    lines += ["",
              f"Self-replays bitwise identical: {all(q['identical'] for q in self_pairs)} "
              f"({len(self_pairs)} platforms).",
              f"C1 (every cross-ISA mismatch >= {100 * PASS_MISMATCH:g}% at rounds >= "
              f"{PASS_FROM_ROUND}, {len(cross)} pairs): **{'pass' if c1_pass else 'fail'}**"
              + ("" if c1_pass else f" ({len(c1_fail)} values below)"), "",
              "| Budget | " + " | ".join(f"r{r}" for r in b_min) + " | --tau-schedule |",
              "|---|" + "---|" * (len(b_min) + 1),
              "| B_min | " + " | ".join(f"{100 * v:.1f}%" for v in b_min.values())
              + f" | `{schedule(b_min)}` |",
              "| B_half | " + " | ".join(f"{100 * v:.1f}%" for v in b_half.values())
              + f" | `{schedule(b_half)}` |"]
    (args.results / "TAU.md").write_text("\n".join(lines) + "\n")
    (args.results / "tau.json").write_text(json.dumps({
        "run": args.run, "pairs": pairs, "c1_pass": c1_pass, "c1_fail": c1_fail,
        "b_min": b_min, "b_half": b_half,
        "schedules": {"b_min": schedule(b_min), "b_half": schedule(b_half)}}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
