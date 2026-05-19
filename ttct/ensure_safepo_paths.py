"""Make forked gym_minigrid (HazardWorld) importable and registered with gym."""
from __future__ import annotations

import os
import sys


def _env_registered(env_id: str) -> bool:
    import gym

    try:
        gym.spec(env_id)
        return True
    except Exception:
        return False


def _purge_gym_minigrid_modules() -> None:
    for name in list(sys.modules):
        if name == "gym_minigrid" or name.startswith("gym_minigrid."):
            del sys.modules[name]


def _load_fork_from_safepo_dir() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    safepo_dir = os.path.join(root, "safepo")
    if not os.path.isdir(os.path.join(safepo_dir, "gym_minigrid")):
        raise RuntimeError(f"Forked gym_minigrid not found under {safepo_dir}")
    if safepo_dir not in sys.path:
        sys.path.insert(0, safepo_dir)
    _purge_gym_minigrid_modules()
    import gym_minigrid.envs  # noqa: F401 — registers HazardWorld-*


def ensure_gym_minigrid_path() -> None:
    """Backward-compatible: ensure HazardWorld envs exist."""
    ensure_hazardworld_env()


def ensure_hazardworld_env(env_id: str = "MiniGrid-HazardWorld-B-v0") -> None:
    """
  Load safepo fork of gym_minigrid so MiniGrid-HazardWorld-* is registered.
  PyPI gym_minigrid does NOT include HazardWorld — do not stop at `import gym_minigrid`.
    """
    if _env_registered(env_id):
        return

    _load_fork_from_safepo_dir()
    if _env_registered(env_id):
        return

    ensure_safepo_editable()
    _load_fork_from_safepo_dir()
    if _env_registered(env_id):
        return

    import gym

    known = [k for k in gym.envs.registry.env_specs.keys() if "HazardWorld" in k]
    raise RuntimeError(
        f"Environment {env_id!r} is not registered.\n"
        f"HazardWorld ids in registry: {known or '(none)'}\n"
        "Run: cd /usr/home/workspace/ttct && pip install --no-build-isolation --no-deps -e ."
    )


def ensure_safepo_editable() -> None:
    """pip install -e this package (registers gym_minigrid via setup.py)."""
    import subprocess

    root = os.path.dirname(os.path.abspath(__file__))
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            root,
        ]
    )
