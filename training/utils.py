"""Small shared helpers: logging, seeding, dtype, parameter counting."""

from __future__ import annotations

import logging
import os
import sys

import jax


def get_logger(name: str = "kimi") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def human(n: int) -> str:
    """Compact human-readable count, e.g. 12_300_000 -> '12.3M'."""
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def device_summary() -> str:
    devs = jax.devices()
    return f"{len(devs)}x {devs[0].platform.upper()} ({devs[0].device_kind})"


def set_xla_flags_for_cpu() -> None:
    """On CPU, expose all cores to XLA so the tiny model trains a bit faster.
    No-op if the user already set the flag or is on GPU/TPU."""
    if jax.default_backend() == "cpu" and "XLA_FLAGS" not in os.environ:
        try:
            n = os.cpu_count() or 1
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={1}"
            os.environ.setdefault("OMP_NUM_THREADS", str(n))
        except Exception:
            pass
