#!/usr/bin/env python3
"""H3 follow-up: what can an attacker hide inside the replay tolerance?

A cross-platform replay verifier must accept a reported update u' whenever
||u' - u_replayed|| <= tau * ||u_replayed||, where tau is the mismatch measured
by replay.py. This script runs FedAvg with one malicious client that, every
round, adds the largest perturbation that still passes that check, pointed in
a harmful direction, and reports the effect against an honest run.

Directions:
  targeted  push test images of --source-class towards --target-class
  reverse   oppose the client's own honest update (slows or undoes learning)
  random    random direction of the same size (control: effect of any
            perturbation, since training amplifies tiny differences)

Usage: python experiments/xisa/hidden_attack.py --tau 1e-3 [--mode targeted] [--rounds 20]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (env_info, exact_init, load_cifar, make_model, portable_args, REPO_ROOT,
                    results_dir, setup_torch, state_to_numpy, write_json)
from fl_trajectory import client_indices, evaluate, fedavg, local_train


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


def run(args, data, tau: float) -> list[dict]:
    parts = client_indices(len(data["y_train"]), args.clients)
    weights = [len(p) / sum(len(q) for q in parts) for p in parts]
    model = make_model()
    exact_init(model)
    glob = state_to_numpy(model)
    keys = [k for k in glob if np.issubdtype(glob[k].dtype, np.floating)]
    history = []
    for r in range(args.rounds + 1):
        if r > 0:
            states = [local_train(glob, data, parts[k], order_seed=args.order_seed, round_idx=r,
                                  client=k, epochs=1, batch=32, lr=args.lr)
                      for k in range(args.clients)]
            if tau > 0:
                g, u = flat(glob, keys), flat(states[0], keys) - flat(glob, keys)
                if args.mode == "targeted":
                    d = targeted_direction(states[0], data, keys, args.source_class, args.target_class)
                elif args.mode == "reverse":
                    d = -u
                else:
                    d = np.random.Generator(np.random.PCG64([4242, r])).standard_normal(u.size)
                d *= tau * np.linalg.norm(u) / max(np.linalg.norm(d), 1e-30)
                states[0] = unflat(g + u + d, states[0], keys)
            glob = fedavg(states, weights)
        _, pred = evaluate(glob, data)
        src_mask = data["y_test"] == args.source_class
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
    ap.add_argument("--mode", choices=["targeted", "reverse", "random"], default="targeted")
    ap.add_argument("--source-class", type=int, default=3)  # cat
    ap.add_argument("--target-class", type=int, default=5)  # dog
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    t0 = time.time()
    honest = run(args, data, 0.0)
    attacked = run(args, data, args.tau)
    rows = []
    for h, a in zip(honest, attacked):
        rows.append({
            "round": h["round"],
            "acc_honest": h["accuracy"], "acc_attacked": a["accuracy"],
            "src_to_tgt_honest": h["source_to_target"],
            "src_to_tgt_attacked": a["source_to_target"],
            "pred_changed": float((h["pred"] != a["pred"]).mean()),
        })
        print(f"round {h['round']:3d} acc {h['accuracy']:.4f} -> {a['accuracy']:.4f}  "
              f"src->tgt {h['source_to_target']:.3f} -> {a['source_to_target']:.3f}  "
              f"changed {rows[-1]['pred_changed']:.3f}", flush=True)
    out = results_dir(args.out, args.tag) / f"attack_{args.mode}_tau{args.tau:g}.json"
    write_json(out, {"env": env_info(), "args": portable_args(vars(args)),
                     "input_hash": data["input_hash"], "elapsed_s": round(time.time() - t0, 1),
                     "rounds": rows})
    print(f"report: {out}")


if __name__ == "__main__":
    main()
