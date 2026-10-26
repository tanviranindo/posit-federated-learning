"""Shared helpers for the cross-ISA divergence pilot.

Everything that feeds a comparison across platforms (data, initial weights,
minibatch order) is produced with operations whose results IEEE 754 or NumPy's
PCG64 generator fix exactly, so any output difference comes from the
operations under test rather than from the inputs.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.models.adaptive_cnn import AdaptiveCNN  # noqa: E402

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def sha256_bytes(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# Arm "CPU part" codes from /proc/cpuinfo for cores likely to appear in runs.
ARM_CPU_PARTS = {"0xd0c": "Neoverse-N1", "0xd40": "Neoverse-V1",
                 "0xd49": "Neoverse-N2", "0xd4f": "Neoverse-V2"}


def describe_cpu(cpu: str) -> str:
    """Readable name for a raw Arm part code; other strings are returned unchanged."""
    if cpu in ARM_CPU_PARTS:
        return ARM_CPU_PARTS[cpu]
    return f"Arm part {cpu}" if cpu.startswith("0x") else cpu


def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if system == "Linux":
            text = Path("/proc/cpuinfo").read_text()
            for key in ("model name", "Model", "CPU part"):
                m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
                if m:
                    return describe_cpu(m.group(1).strip())
        if system == "Windows":
            return platform.processor()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _blas_info() -> str:
    m = re.search(r"BLAS_INFO=([^,\s]+)", torch.__config__.show())
    return m.group(1) if m else "unknown"


def env_info() -> dict:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "blas": _blas_info(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn": torch.backends.mkldnn.is_available(),
        "threads": torch.get_num_threads(),
        "in_container": Path("/.dockerenv").exists(),
    }


def platform_tag() -> str:
    """Short identifier such as darwin-arm64-accelerate."""
    e = env_info()
    machine = {"aarch64": "arm64", "AMD64": "x86_64"}.get(e["machine"], e["machine"])
    return f"{e['system'].lower()}-{machine}-{e['blas'].lower()}"


def setup_torch(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)


def exact_uniform(rng: np.random.Generator, shape, scale_exp: int = 0) -> np.ndarray:
    """float32 values k * 2**-23 * 2**scale_exp with |k| < 2**23: exact on every ISA."""
    k = rng.integers(-(2**23), 2**23, size=shape, dtype=np.int64)
    return k.astype(np.float32) * np.float32(2.0 ** (scale_exp - 23))


def load_cifar(path: Path) -> dict:
    with np.load(path) as d:
        data = {k: d[k] for k in d.files}
    x_tr = normalize(data["x_train"])
    x_te = normalize(data["x_test"])
    return {
        "x_train": x_tr, "y_train": data["y_train"].astype(np.int64),
        "x_test": x_te, "y_test": data["y_test"].astype(np.int64),
        "input_hash": sha256_bytes(x_tr, x_te, data["y_train"], data["y_test"]),
    }


def normalize(x_uint8: np.ndarray) -> np.ndarray:
    """NHWC uint8 -> NCHW float32 with element-wise ops only (correctly rounded)."""
    x = x_uint8.astype(np.float32) / np.float32(255.0)
    x = (x - CIFAR_MEAN) / CIFAR_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def make_model() -> AdaptiveCNN:
    """The paper's 1.28M-parameter CNN on every platform, dropout disabled."""
    model = AdaptiveCNN(num_classes=10, architecture="x86_64")
    model.dropout.p = 0.0
    return model


def exact_init(model: torch.nn.Module, seed: int = 1234) -> None:
    """Initialise weights from PCG64 integers with power-of-two scales.

    Each weight tensor gets uniform values in about [-b, b] where b is the
    power of two nearest to 1/sqrt(fan_in); biases start at zero and BatchNorm
    at weight 1, bias 0. The values are bit-identical on every platform.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                fan_in = int(np.prod(p.shape[1:]))
                exp = int(np.round(np.log2(1.0 / np.sqrt(fan_in))))
                p.copy_(torch.from_numpy(exact_uniform(rng, tuple(p.shape), exp)))
            elif "bn" in name and name.endswith("weight"):
                p.fill_(1.0)
            else:
                p.zero_()


def state_to_numpy(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def state_hash(state: dict) -> str:
    return sha256_bytes(*(state[k] for k in sorted(state)))


def portable_args(args: dict) -> dict:
    """Arguments with repository paths made relative, so reports carry no host paths."""
    out = {}
    for k, v in args.items():
        if isinstance(v, (str, Path)) and Path(v).is_absolute():
            try:
                v = str(Path(v).resolve().relative_to(REPO_ROOT))
            except ValueError:
                v = Path(v).name
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def results_dir(base: str, tag: str | None = None) -> Path:
    out = Path(base) / (tag or platform_tag())
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_threads() -> int:
    return int(os.environ.get("XISA_THREADS", "1"))
