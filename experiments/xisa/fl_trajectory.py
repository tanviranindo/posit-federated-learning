#!/usr/bin/env python3
"""H2: do identical FedAvg runs drift apart on different platforms?

Simulates K clients in one process with fully fixed inputs: same model, same
initial weights, same data split and the same minibatch order on every
platform. Records the global model hash, accuracy and every test prediction
after each round, and saves full states at selected rounds for compare.py and
replay.py.

Usage: python experiments/xisa/fl_trajectory.py --threads 1 --rounds 20 [--order-seed 7]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (env_info, exact_init, load_cifar, make_model, portable_args, REPO_ROOT,
                    results_dir, setup_torch, sha256_bytes, state_hash,
                    state_to_numpy, write_json)

SPLIT_SEED = 99


def client_indices(labels: np.ndarray, clients: int, split: str = "iid",
                   alpha: float = 0.3, min_size: int = 32) -> list[np.ndarray]:
    """Client partition, fixed independently of the order seed.

    iid: equal random shares. dirichlet: per-class shares drawn from
    Dirichlet(alpha), redrawn until every client has at least min_size
    samples. Dirichlet sampling goes through libm, so callers record
    split_hash() and check it across platforms.
    """
    rng = np.random.Generator(np.random.PCG64(SPLIT_SEED))
    n_train = len(labels)
    if split == "iid":
        perm = rng.permutation(n_train)
        size = n_train // clients
        return [np.sort(perm[k * size:(k + 1) * size]) for k in range(clients)]
    if split != "dirichlet":
        raise ValueError(split)
    while True:
        parts = [[] for _ in range(clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            cuts = (np.cumsum(rng.dirichlet(np.full(clients, alpha))) * len(idx)).astype(np.int64)
            for k, chunk in enumerate(np.split(idx, cuts[:-1])):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=np.int64)) for p in parts]


def split_hash(parts: list[np.ndarray]) -> str:
    return sha256_bytes(*parts)


def local_train(global_state: dict, data: dict, idx: np.ndarray, *, order_seed: int,
                round_idx: int, client: int, epochs: int, batch: int, lr: float) -> dict:
    """One client's local update from a given global state; returns its new state."""
    model = make_model()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in global_state.items()})
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    rng = np.random.Generator(np.random.PCG64([order_seed, round_idx, client]))
    x_all = torch.from_numpy(data["x_train"])
    y_all = torch.from_numpy(data["y_train"])
    for _ in range(epochs):
        order = idx[rng.permutation(len(idx))]
        for start in range(0, len(order), batch):
            b = torch.from_numpy(order[start:start + batch])
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_all[b]), y_all[b])
            loss.backward()
            opt.step()
    return state_to_numpy(model)


def fedavg(states: list[dict], weights: list[float]) -> dict:
    """Sequential float32 weighted sum, as in the paper; integer buffers copied."""
    out = {}
    for key in states[0]:
        if not np.issubdtype(states[0][key].dtype, np.floating):
            out[key] = states[0][key].copy()
            continue
        acc = torch.zeros_like(torch.from_numpy(states[0][key]))
        for w, s in zip(weights, states):
            acc = acc + w * torch.from_numpy(s[key])
        out[key] = acc.numpy()
    return out


@torch.no_grad()
def evaluate(state: dict, data: dict) -> tuple[np.ndarray, np.ndarray]:
    model = make_model()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()})
    model.eval()
    x = torch.from_numpy(data["x_test"])
    logits = torch.cat([model(x[i:i + 500]) for i in range(0, len(x), 500)])
    return logits.numpy(), logits.argmax(1).numpy().astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--split", choices=["iid", "dirichlet"], default="iid")
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--order-seed", type=int, default=7)
    ap.add_argument("--save-rounds", default="0,1,2,5,10,20")
    ap.add_argument("--replay-clients", default="0,1")
    ap.add_argument("--label", default="", help="suffix for repeated runs, e.g. _rep")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "xisa_cifar10.npz"))
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "results" / "xisa"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    setup_torch(args.threads)
    data = load_cifar(args.data)
    save_rounds = {int(r) for r in args.save_rounds.split(",")}
    replay_clients = [int(c) for c in args.replay_clients.split(",") if c]
    run_dir = results_dir(args.out, args.tag) / (
        f"traj_t{args.threads}_o{args.order_seed}{args.label}")
    run_dir.mkdir(parents=True, exist_ok=True)

    parts = client_indices(data["y_train"], args.clients, args.split, args.alpha)
    weights = [len(p) / sum(len(q) for q in parts) for p in parts]
    train_kw = dict(order_seed=args.order_seed, epochs=args.epochs,
                    batch=args.batch, lr=args.lr)

    model = make_model()
    exact_init(model)
    glob = state_to_numpy(model)

    saved, preds, history = {}, {}, []
    t_start = time.time()
    for r in range(args.rounds + 1):
        if r > 0:
            prev = glob
            client_states = [local_train(prev, data, parts[k], round_idx=r, client=k, **train_kw)
                             for k in range(args.clients)]
            if r - 1 in save_rounds:
                for k in replay_clients:
                    for key, v in client_states[k].items():
                        saved[f"client{k}_r{r}/{key}"] = v
            glob = fedavg(client_states, weights)
        logits, pred = evaluate(glob, data)
        preds[f"r{r}"] = pred
        if r in save_rounds:
            for key, v in glob.items():
                saved[f"global_r{r}/{key}"] = v
            saved[f"logits_r{r}"] = logits
        history.append({
            "round": r,
            "state_hash": state_hash(glob),
            "logits_hash": sha256_bytes(logits),
            "accuracy": float((pred == data["y_test"]).mean()),
            "elapsed_s": round(time.time() - t_start, 2),
        })
        print(f"round {r:3d} acc {history[-1]['accuracy']:.4f} "
              f"hash {history[-1]['state_hash'][:12]} ({history[-1]['elapsed_s']}s)", flush=True)

    np.savez(run_dir / "states.npz", **saved)
    np.savez(run_dir / "preds.npz", **preds)
    write_json(run_dir / "summary.json", {
        "env": env_info(),
        "args": portable_args(vars(args)),
        "input_hash": data["input_hash"],
        "split_seed": SPLIT_SEED,
        "split_hash": split_hash(parts),
        "client_sizes": [len(p) for p in parts],
        "history": history,
    })


if __name__ == "__main__":
    main()
