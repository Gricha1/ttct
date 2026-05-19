from setuptools import setup, find_packages

# HazardWorld lives under safepo/gym_minigrid but must be importable as gym_minigrid.*
_packages = [p for p in find_packages() if not p.startswith("safepo.gym_minigrid")]

setup(
    name="safepo",
    version="0.1",
    packages=_packages + ["gym_minigrid", "gym_minigrid.envs"],
    package_dir={"gym_minigrid": "safepo/gym_minigrid"},
)