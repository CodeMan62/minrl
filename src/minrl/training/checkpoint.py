"""Atomic, verified, resumable checkpoints.

One checkpoint is a directory::

    step_00000100/
        manifest.json     written last: a directory without one is incomplete
        model.pt          full unsharded weights                       (rank 0)
        optimizer.pt      full optimizer state                         (rank 0)
        trainer.pt        step, token count, config, topology          (rank 0)
        rank0/rng.pt      python / torch / cuda / numpy RNG streams    (per rank)
        rank0/source.pt   the batch source's own cursor                (per rank)

A save writes into ``<dir>.tmp`` and renames it into place, so a crash mid-write
never touches the previous good checkpoint. ``latest`` names the newest one.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from minrl.training import dist

FORMAT = 1
LATEST = "latest"
_STEP_DIR = re.compile(r"step_(\d+)")


@dataclass
class Loaded:
    manifest: Dict[str, Any]
    trainer: Dict[str, Any]
    rank: Dict[str, Any]   # this rank's rng / source state, empty if it has none
    model: Dict[str, Any]  # rank 0 only; other ranks get it by broadcast
    optimizer: Dict[str, Any]


# ---- RNG ---------------------------------------------------------------

def rng_state() -> Dict[str, Any]:
    """Every RNG stream this process draws from."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


# ---- save ----------------------------------------------------------------

def save(
    path: str,
    *,
    model_state: Dict[str, Any],
    optim_state: Dict[str, Any],
    trainer_state: Dict[str, Any],
    rank_state: Dict[str, Any],
    keep_last: int = 0,
) -> None:
    """Every rank calls this together. Rank 0 owns the big files and the manifest."""
    path = os.path.abspath(path)
    tmp = path + ".tmp"
    if dist.is_main():
        shutil.rmtree(tmp, ignore_errors=True)  # a previous attempt that died
        os.makedirs(tmp)
    dist.barrier()

    rank_dir = os.path.join(tmp, f"rank{dist.rank()}")
    os.makedirs(rank_dir, exist_ok=True)
    for name, obj in rank_state.items():
        _dump(obj, os.path.join(rank_dir, f"{name}.pt"))
    if dist.is_main():
        _dump(model_state, os.path.join(tmp, "model.pt"))
        _dump(optim_state, os.path.join(tmp, "optimizer.pt"))
        _dump(trainer_state, os.path.join(tmp, "trainer.pt"))
    dist.barrier()  # every rank's files exist before anything is hashed

    if dist.is_main():
        files = sorted(
            os.path.relpath(os.path.join(d, f), tmp)
            for d, _, fs in os.walk(tmp) for f in fs
        )
        manifest = {
            "format": FORMAT,
            "step": trainer_state["step"],
            "world_size": dist.world_size(),
            "files": {f: _sha256(os.path.join(tmp, f)) for f in files},
        }
        _dump_json(manifest, os.path.join(tmp, "manifest.json"))
        _swap_in(tmp, path)
        _write_latest(path)
        if keep_last > 0:
            _prune(os.path.dirname(path), keep_last, keep=os.path.basename(path))
    dist.barrier()  # no rank moves on before the checkpoint is durable


# ---- load ----------------------------------------------------------------

def latest(ckpt_dir: str) -> Optional[str]:
    """Path of the newest checkpoint under ``ckpt_dir``, or None on a fresh run."""
    marker = os.path.join(ckpt_dir, LATEST)
    if not os.path.isfile(marker):
        return None
    with open(marker) as f:
        return os.path.join(ckpt_dir, f.read().strip())


def resolve(path: str) -> str:
    """``path`` itself, or its newest checkpoint if it is a ckpt_dir with a ``latest``."""
    return latest(path) or path


def load(path: str, *, strict: bool = True) -> Loaded:
    """Read and verify a checkpoint. ``strict`` refuses a different world size."""
    path = resolve(path)
    manifest_file = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_file):
        raise FileNotFoundError(f"{path} has no manifest.json: not a complete checkpoint")
    manifest = _load_json(manifest_file)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"checkpoint format {manifest.get('format')} != {FORMAT}")

    corrupt = 0
    if dist.is_main():
        corrupt = sum(
            _sha256(os.path.join(path, f)) != h for f, h in manifest["files"].items()
        )
    if dist.sum_all(corrupt)[0]:
        raise RuntimeError(f"{path}: {int(corrupt)} file(s) fail their checksum; refusing to resume")

    if manifest["world_size"] != dist.world_size():
        msg = (f"checkpoint was saved by {manifest['world_size']} rank(s), "
               f"resuming on {dist.world_size()}")
        if strict:
            raise ValueError(msg + "; pass strict=False to resume anyway (RNG and data "
                             "cursors will not line up)")
        warnings.warn(msg, stacklevel=2)

    trainer = _load(os.path.join(path, "trainer.pt"))
    rank_dir = os.path.join(path, f"rank{dist.rank()}")
    rank = {}
    if os.path.isdir(rank_dir):
        rank = {f[:-3]: _load(os.path.join(rank_dir, f)) for f in os.listdir(rank_dir)}
    model = optimizer = {}
    if dist.is_main():
        model = _load(os.path.join(path, "model.pt"))
        optimizer = _load(os.path.join(path, "optimizer.pt"))
    return Loaded(manifest, trainer, rank, model, optimizer)


# ---- files ---------------------------------------------------------------

def _dump(obj: Any, file: str) -> None:
    with open(file, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())


def _load(file: str) -> Any:
    # weights_only=False: RNG states and numpy arrays are not plain tensors.
    return torch.load(file, map_location="cpu", weights_only=False)


def _dump_json(obj: Any, file: str) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _load_json(file: str) -> Any:
    with open(file) as f:
        return json.load(f)


def _sha256(file: str) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(d: str) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _swap_in(tmp: str, final: str) -> None:
    """Rename ``tmp`` into ``final``; an older ``final`` stays until the new one is in place."""
    old = final + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)
    _fsync_dir(os.path.dirname(final))


def _write_latest(path: str) -> None:
    ckpt_dir, name = os.path.split(path)
    tmp = os.path.join(ckpt_dir, LATEST + ".tmp")
    with open(tmp, "w") as f:
        f.write(name)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(ckpt_dir, LATEST))
    _fsync_dir(ckpt_dir)


def _prune(ckpt_dir: str, keep_last: int, *, keep: str) -> None:
    """Delete all but the newest ``keep_last`` ``step_*`` dirs; never ``keep``."""
    steps = sorted(
        (d for d in os.listdir(ckpt_dir) if _STEP_DIR.fullmatch(d)),
        key=lambda d: int(_STEP_DIR.fullmatch(d).group(1)),
    )
    for d in steps[:-keep_last]:
        if d != keep:
            shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)
