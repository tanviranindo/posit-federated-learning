#!/usr/bin/env python3
"""Build the fixed CIFAR-10 subset used by every pilot platform.

Reads the official cifar-10-python.tar.gz (downloading it if missing) and
writes uint8 arrays to an .npz file, so every platform starts from the same
bytes. Normalisation happens later, in common.normalize.

Usage: python experiments/xisa/prepare_data.py [--train 5000] [--test 2000]
"""

import argparse
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_batch(tar: tarfile.TarFile, name: str):
    # The official CIFAR-10 batches are pickles; main() checks the archive's
    # published MD5 before anything is unpickled.
    member = tar.getmember(f"cifar-10-batches-py/{name}")
    batch = pickle.load(tar.extractfile(member), encoding="bytes")
    x = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch[b"labels"], dtype=np.uint8)
    return np.ascontiguousarray(x, dtype=np.uint8), y


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=Path, default=root / "data" / "cifar-10-python.tar.gz")
    ap.add_argument("--out", type=Path, default=root / "data" / "xisa_cifar10.npz")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()

    if not args.tar.exists():
        args.tar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, args.tar)
    digest = md5(args.tar)
    if digest != TAR_MD5:
        raise SystemExit(f"unexpected md5 {digest} for {args.tar}")

    with tarfile.open(args.tar, "r:gz") as tar:
        x_tr, y_tr = read_batch(tar, "data_batch_1")
        x_te, y_te = read_batch(tar, "test_batch")

    np.savez(args.out, x_train=x_tr[: args.train], y_train=y_tr[: args.train],
             x_test=x_te[: args.test], y_test=y_te[: args.test])
    print(f"wrote {args.out}: train {args.train}, test {args.test}")


if __name__ == "__main__":
    main()
