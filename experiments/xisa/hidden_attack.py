#!/usr/bin/env python3
"""H3 follow-up: what can an attacker hide inside the replay tolerance?

A cross-platform replay verifier must accept a reported update u' whenever
||u' - u_replayed|| <= tau * ||u_replayed||, where tau is the mismatch measured
by replay.py. This script runs FedAvg in which --malicious clients add, in the
rounds given by --attack-rounds, the largest perturbation that still passes
that check, and compares each attack direction against one honest run.

Directions (--modes, comma-separated):
  targeted  push test images of --source-class towards --target-class
  reverse   oppose the client's own honest update (slows or undoes learning)
  random    random direction of the same size (control: effect of any
            perturbation, since training amplifies tiny differences)

Usage: python experiments/xisa/hidden_attack.py --tau 0.02 [--modes targeted,random]
           [--malicious 3] [--epochs 5] [--split dirichlet] [--attack-rounds 1-3]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (env_info, exact_init, load_cifar, make_model, portable_args, REPO_ROOT,
                    results_dir, setup_torch, state_to_numpy, write_json)
from fl_trajectory import client_indices, evaluate, fedavg, local_train, split_hash


def flat(state: dict, keys: list[str]) -> np.ndarray:
    return np.concatenate([state[k].ravel() for k in keys]).astype(np.float64)


def unflat(vec: np.ndarray, like: dict, keys: list[str]) -> dict:
    out, i = dict(like), 0
    for k in keys:
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape).astype(like[k].dtype)
        i += n
    return out


def targeted_direction(state: dict, data: dict, keys: list[str], src: int, dst: int) -> np.ndarray:
    """Descent direction of the loss for relabelling class src as dst."""
    model = make_model()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()})
    model.train()
    mask = data["y_train"] == src
    x = torch.from_numpy(data["x_train"][mask][:256])
    y = torch.full((len(x),), dst, dtype=torch.int64)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    by_name = {n: g.numpy() for (n, _), g in zip(model.named_parameters(), grads)}
    return -np.concatenate([by_name[k].ravel() if k in by_name else np.zeros(state[k].size)
                            for k in keys]).astype(np.float64)


def parse_rounds(spec: str, rounds: int) -> set[int]:
    if spec == "all":
        return set(range(1, rounds + 1))
    lo, _, hi = spec.partition("-")
    return set(range(int(lo), int(hi or lo) + 1))


def run(args, data, parts, mode: str | None) -> list[dict]:
    """One FedAvg run; mode None is the honest baseline."""
    weights = [len(p) / sum(len(q) for q in parts) for p in parts]
    attack_rounds = parse_rounds(args.attack_rounds, args.rounds)
    model = make_model()
    exact_init(model)
    glob = state_to_numpy(model)
    keys = [k for k in glob if np.issubdtype(glob[k].dtype, np.floating)]
    src_mask = data["y_test"] == args.source_class
    history = []
    for r in range(args.rounds + 1):
        if r > 0:
            states = [local_train(glob, data, parts[k], order_seed=args.order_seed, round_idx=r,
                                  client=k, epochs=args.epochs, batch=32, lr=args.lr)
                      for k in range(args.clients)]
            if mode is not None and r in attack_rounds:
                g = flat(glob, keys)
                for k in range(args.malicious):
                    u = flat(states[k], keys) - g
                    if mode == "targeted":
                        d = targeted_direction(states[k], data, keys,
                                               args.source_class, args.target_class)
                    elif mode == "reverse":
                        d = -u
                    else:
                        d = np.random.Generator(np.random.PCG64([4242, r, k])).standard_normal(u.size)
                    d *= args.tau * np.linalg.norm(u) / max(np.linalg.norm(d), 1e-30)
                    states[k] = unflat(g + u + d, states[k], keys)
            glob = fedavg(states, weights)
        _, pred = evaluate(glob, data)
        history.append({
            "round": r,
            "accuracy": float((pred == data["y_test"]).mean()),
            "source_to_target": float((pred[src_mask] == args.target_class).mean()),
            "pred": pred,
        })
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True,
                    help="relative tolerance from replay.py (mismatch_rel_to_update)")
    ap.add_argument("--modes", default="targeted,random")
    ap.add_argument("--malicious", type=int, default=1, help="clients 0..m-1 attack")
    ap.add_argument("--attack-rounds", default="all", help='"all" or a range such as 1-3')
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--split", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--source-class", type=int, default=3)  # cat
    ap.add_argument("--target-class", type=int, default=5)  # dog
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--name", default=None, help="report file stem (default: built from args)")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    parts = client_indices(data["y_train"], args.clients, args.split, args.alpha)
    t0 = time.time()
    honest = run(args, data, parts, None)
    modes = {m: run(args, data, parts, m) for m in args.modes.split(",")}

    rows = []
    for i, h in enumerate(honest):
        row = {"round": h["round"], "acc_honest": h["accuracy"],
               "src_to_tgt_honest": h["source_to_target"]}
        for m, hist in modes.items():
            a = hist[i]
            row[f"acc_{m}"] = a["accuracy"]
            row[f"src_to_tgt_{m}"] = a["source_to_target"]
            row[f"pred_changed_{m}"] = float((h["pred"] != a["pred"]).mean())
        rows.append(row)
        print(f"round {h['round']:3d} acc {h['accuracy']:.4f} " + " ".join(
            f"{m}:{row[f'acc_{m}']:.4f}/{row[f'src_to_tgt_{m}']:.3f}" for m in modes)
            + f"  (honest src->tgt {h['source_to_target']:.3f})", flush=True)

    name = args.name or (f"attack_{args.split}_e{args.epochs}_m{args.malicious}_"
                         f"r{args.attack_rounds}_tau{args.tau:g}_o{args.order_seed}")
    out = results_dir(args.out, args.tag) / f"{name}.json"
    write_json(out, {"env": env_info(), "args": portable_args(vars(args)),
                     "input_hash": data["input_hash"], "split_hash": split_hash(parts),
                     "client_sizes": [len(p) for p in parts],
                     "elapsed_s": round(time.time() - t0, 1), "rounds": rows})
    print(f"report: {out}")


if __name__ == "__main__":
    main()
