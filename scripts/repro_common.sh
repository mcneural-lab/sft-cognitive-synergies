#!/usr/bin/env bash
# Shared helpers for public-release reproduction scripts.

set -euo pipefail

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." >/dev/null 2>&1
  pwd
}

REPO_ROOT="${REPO_ROOT:-$(repo_root)}"

default_release_data_root() {
  if [[ -d "data" || -d "results" ]]; then
    printf '%s\n' "."
  else
    printf '%s\n' "."
  fi
}

RELEASE_DATA_ROOT="${RELEASE_DATA_ROOT:-$(default_release_data_root)}"

cd_repo_root() {
  cd "$REPO_ROOT"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found in PATH: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Required directory not found: $1"
}

ensure_dir() {
  mkdir -p "$1"
}

release_path() {
  local rel_path="$1"
  printf '%s/%s\n' "${RELEASE_DATA_ROOT%/}" "${rel_path#/}"
}

run_cmd() {
  info "$*"
  "$@"
}

run_uv() {
  require_cmd uv
  run_cmd uv run "$@"
}

list_figures_from_config() {
  require_cmd python3
  python3 - <<'PY'
from pathlib import Path

config = Path("configs/figures.toml")
figures = []
current = None
for raw_line in config.read_text().splitlines():
    line = raw_line.strip()
    if line == "[[figures]]":
        if current:
            figures.append(current)
        current = {}
        continue
    if current is None or "=" not in line or line.startswith("#"):
        continue
    key, value = line.split("=", 1)
    current[key.strip()] = value.strip().strip('"')
if current:
    figures.append(current)

print(f"{'id':28} {'number':22} description")
print("-" * 90)
for fig in figures:
    print(f"{fig.get('id', ''):28} {fig.get('number', ''):22} {fig.get('description', '')}")
PY
}
