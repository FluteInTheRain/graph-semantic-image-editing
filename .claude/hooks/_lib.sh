#!/usr/bin/env bash
# Shared helpers for the project hooks. Sourced, never executed directly.
# Design rule: hooks must be fast and must never break normal work. If a
# dependency (jq/python3/git) is missing we "fail open" instead of blocking.

# Read the whole hook JSON payload from stdin once.
HOOK_INPUT="$(cat)"

# json_get <jq-filter> <python-expression-on-d>
# Extract a value from HOOK_INPUT with jq, falling back to python3.
json_get() {
  local jq_filter="$1" py_expr="$2"
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$HOOK_INPUT" | jq -r "$jq_filter" 2>/dev/null
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$HOOK_INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); v=$py_expr; print('' if v is None else v)" 2>/dev/null
  else
    return 1
  fi
}

# json_escape <string> -> JSON string literal (with quotes).
json_escape() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$1" | jq -Rs .
  else
    printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
  fi
}

# repo_root_for <path>: git top-level of the nearest existing parent dir.
repo_root_for() {
  local dir
  dir="$(dirname "$1")"
  while [ ! -d "$dir" ] && [ "$dir" != "/" ]; do dir="$(dirname "$dir")"; done
  git -C "$dir" rev-parse --show-toplevel 2>/dev/null
}
