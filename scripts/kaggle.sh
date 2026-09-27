#!/usr/bin/env bash
# Kaggle helpers for the p3a project.
#
#   bundle        build kaggle/code_bundle, then create or version the dataset
#   push JOB      push kaggle/jobs/JOB to Kaggle on a T4
#   status JOB    show the kernel status
#   output JOB    download the kernel output to kaggle/out/JOB/
#   dry-run       build and validate the bundle locally, no Kaggle calls
#
# The code bundle is never uploaded by `dry-run`; only `bundle` talks to Kaggle.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
DATASET_REF="mustaqimnishat/p300-proto-attack-code"
MSG="${MSG:-update $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}"
  exit 2
}

require_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found on PATH" >&2
    exit 1
  fi
}

job_ref() {
  local job="$1"
  local metadata="kaggle/jobs/$job/kernel-metadata.json"
  [[ -f "$metadata" ]] || { echo "missing $metadata" >&2; exit 1; }
  "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$metadata"
}

cmd_bundle() {
  require_kaggle
  "$PYTHON" scripts/make_code_bundle.py
  if kaggle datasets status "$DATASET_REF" >/dev/null 2>&1; then
    echo "dataset $DATASET_REF exists, uploading a new version"
    kaggle datasets version -p kaggle/code_bundle -m "$MSG" --dir-mode zip
  else
    echo "dataset $DATASET_REF not found, creating it"
    kaggle datasets create -p kaggle/code_bundle --dir-mode zip
  fi
}

cmd_dry_run() {
  "$PYTHON" scripts/make_code_bundle.py --dry-run
}

cmd_push() {
  local job="${1:-}"
  [[ -n "$job" ]] || usage
  require_kaggle
  # If the accelerator name is rejected, kaggle prints the error and we stop.
  # Do not substitute a different accelerator name automatically.
  kaggle kernels push -p "kaggle/jobs/$job" --accelerator NvidiaTeslaT4
}

cmd_status() {
  local job="${1:-}"
  [[ -n "$job" ]] || usage
  require_kaggle
  kaggle kernels status "$(job_ref "$job")"
}

cmd_output() {
  local job="${1:-}"
  [[ -n "$job" ]] || usage
  require_kaggle
  mkdir -p "kaggle/out/$job"
  kaggle kernels output "$(job_ref "$job")" -p "kaggle/out/$job"
}

case "${1:-}" in
  bundle) shift; cmd_bundle "$@" ;;
  dry-run) shift; cmd_dry_run "$@" ;;
  push) shift; cmd_push "$@" ;;
  status) shift; cmd_status "$@" ;;
  output) shift; cmd_output "$@" ;;
  *) usage ;;
esac
