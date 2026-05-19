#!/usr/bin/env bash
# Sync editable packages and caged_craftext deps from the mounted workspace (host code).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/usr/home/workspace}"
export WORKSPACE_ROOT="$WORKSPACE"

# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate dynalang

quick_editable_install() {
  # Inner ttct/: safepo + top-level gym_minigrid (HazardWorld). Required for MiniGrid scripts.
  if [[ -f "$WORKSPACE/ttct/setup.py" ]]; then
    echo "entrypoint: pip install -e $WORKSPACE/ttct (safepo / gym_minigrid)"
    pip install --no-build-isolation --no-deps -e "$WORKSPACE/ttct"
  else
    echo "entrypoint: WARNING: $WORKSPACE/ttct/setup.py not found — mount workspace?"
  fi
  local craftax_dir="$WORKSPACE/caged_craftext/Craftax"
  if [[ -d "$craftax_dir" ]] && [[ -f "$craftax_dir/setup.py" || -f "$craftax_dir/pyproject.toml" ]]; then
    pip install -q --no-build-isolation --no-deps -e "$craftax_dir" || true
  fi
}

deps_stamp_file() {
  echo "$WORKSPACE/.docker/caged_deps.stamp"
}

requirements_hash() {
  local req="$WORKSPACE/caged_craftext/requirements.txt"
  local fallback="$WORKSPACE/caged_craftext/docker/requirements.txt"
  if [[ -f "$req" ]]; then
    md5sum "$req" | awk '{print $1}'
  elif [[ -f "$fallback" ]]; then
    md5sum "$fallback" | awk '{print $1}'
  else
    echo "missing"
  fi
}

maybe_install_caged_deps() {
  local install_script="$WORKSPACE/caged_craftext/docker/install_deps.sh"
  if [[ ! -f "$install_script" ]]; then
    echo "entrypoint: no $install_script (mount workspace or caged_craftext?)"
    return 0
  fi

  if [[ "${FORCE_INSTALL_DEPS:-0}" == "1" ]]; then
    echo "entrypoint: FORCE_INSTALL_DEPS=1 — running install_deps.sh"
    bash "$install_script"
    mkdir -p "$WORKSPACE/.docker"
    requirements_hash > "$(deps_stamp_file)"
    return 0
  fi

  if [[ "${SKIP_INSTALL_DEPS:-0}" == "1" ]]; then
    return 0
  fi

  local stamp
  stamp="$(deps_stamp_file)"
  local current prev=""
  current="$(requirements_hash)"
  [[ -f "$stamp" ]] && prev="$(cat "$stamp")"

  if [[ "$current" != "$prev" ]]; then
    echo "entrypoint: requirements changed — running install_deps.sh"
    bash "$install_script"
    mkdir -p "$WORKSPACE/.docker"
    echo "$current" > "$stamp"
  fi
}

if [[ ! -d "$WORKSPACE" ]]; then
  echo "entrypoint: workspace $WORKSPACE not found"
else
  quick_editable_install
  maybe_install_caged_deps
fi

exec "$@"
