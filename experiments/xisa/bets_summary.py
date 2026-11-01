#!/usr/bin/env python3
"""Score the harm bets against the pre-registered rule.

A bet succeeds if, at the judged round, the targeted attack beats the random
control by >= 1 pp of accuracy loss or >= 1 pp more source->target predictions,
in every seed. Reads attack_*.json from a results folder and writes
BETS.md next to them.

Usage: python experiments/xisa/bets_summary.py [RESULTS_DIR]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

THRESHOLD = 0.01


def bet_name(a: dict) -> str:
    parts = []
    if a["epochs"] != 1:
        parts.append(f"E{a['epochs']}")
    if a["malicious"] != 1:
        parts.append(f"C{a['malicious']}")
    if a["attack_rounds"] != "all":
        parts.append(f"W{a['attack_rounds']}")
    if a.get("tau_schedule"):
        parts.append("sched")
    if a["split"] != "iid":
        parts.append(f"N{a['alpha']:g}")
    return "+".join(parts) or "baseline"


def judge(row: dict) -> dict:
    acc_gap = row["acc_random"] - row["acc_targeted"]          # extra accuracy lost
    flip_gap = row["src_to_tgt_targeted"] - row["src_to_tgt_random"]
    return {"acc_gap": acc_gap, "flip_gap": flip_gap,
            "acc_vs_honest": row["acc_honest"] - row["acc_targeted"],
            "flip_vs_honest": row["src_to_tgt_targeted"] - row["src_to_tgt_honest"],
            "hit": acc_gap >= THRESHOLD or flip_gap >= THRESHOLD}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?",
                    default=Path(__file__).resolve().parents[1] / "results" / "xisa" / "bets")
    args = ap.parse_args()

    runs = defaultdict(list)
    for p in sorted(args.results.glob("attack_*.json")):
        d = json.loads(p.read_text())
        runs[(bet_name(d["args"]), d["args"].get("tau_schedule") or d["args"]["tau"])].append(d)

    lines = ["# Harm bets", "",
             "Targeted attack vs random control of the same size (rule: gap >= 1 pp in every seed).",
             "acc gap = extra accuracy lost by targeted vs random; flip gap = extra cat->dog rate.", "",
             "| Bet | tau | Round | Seeds | acc gap per seed (pp) | flip gap per seed (pp) | "
             "targeted vs honest: acc / flips (pp, mean) | Success |",
             "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for (bet, tau), ds in sorted(runs.items()):
        last = ds[0]["rounds"][-1]["round"]
        judged = [3, last] if "W" in bet else [last]
        for r in judged:
            js = [judge(d["rounds"][r]) for d in ds]
            ok = len(js) >= 3 and all(j["hit"] for j in js)
            summary[f"{bet}@r{r}"] = {"tau": tau, "seeds": [d["args"]["order_seed"] for d in ds],
                                      "per_seed": js, "success": ok}
            mean = lambda k: sum(j[k] for j in js) / len(js)  # noqa: E731
            lines.append(
                f"| {bet} | {tau if isinstance(tau, str) else format(tau, "g")} | {r} | {len(js)} | "
                + ", ".join(f"{100 * j['acc_gap']:+.2f}" for j in js) + " | "
                + ", ".join(f"{100 * j['flip_gap']:+.2f}" for j in js) + " | "
                + f"{100 * mean('acc_vs_honest'):+.2f} / {100 * mean('flip_vs_honest'):+.2f} | "
                + ("**yes**" if ok else "no") + " |")
    (args.results / "BETS.md").write_text("\n".join(lines) + "\n")
    (args.results / "bets_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
