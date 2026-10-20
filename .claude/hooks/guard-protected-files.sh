#!/usr/bin/env bash
# PreToolUse hook (Edit|Write). Two guards, both about reproducibility (PDF §6):
#  1. Committed (git-tracked) files under splits/ and queries/ are frozen: all
#     models must share the same split files, and the >=20 evaluation queries are
#     eval-only. Creating NEW untracked files there is allowed.
#     Override for a deliberate change: start Claude with MMXAI_ALLOW_PROTECTED_EDIT=1
#     (and write an ADR explaining why).
#  2. Do not write large or binary artifacts into git-tracked paths. Anything that
#     .gitignore already ignores (data/, results/runs/, *.npy, ...) is allowed.
# Denials use the documented JSON output (permissionDecision=deny, exit 0).

source "$(dirname "$0")/_lib.sh" || exit 0

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}\n' "$(json_escape "$1")"
  exit 0
}

if ! command -v jq >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "guard-protected-files: jq/python3 not found, guard skipped" >&2
  exit 1  # non-blocking error: visible notice, action proceeds
fi

file="$(json_get '.tool_input.file_path // empty' 'd.get("tool_input",{}).get("file_path")')"
[ -n "$file" ] || exit 0
root="$(repo_root_for "$file")"
[ -n "$root" ] || exit 0          # not inside a git repo: nothing to protect
rel="${file#"$root"/}"

# --- Guard 1: frozen split / query files -----------------------------------
case "$rel" in
  splits/*|queries/*)
    if [ "${MMXAI_ALLOW_PROTECTED_EDIT:-0}" != "1" ] \
       && git -C "$root" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
      deny "BLOCKED: $rel is a committed split/evaluation-query file (PDF §6: every model must use the same split files; the evaluation queries are eval-only). Do not edit it. If the change is truly intended, a human must restart Claude with MMXAI_ALLOW_PROTECTED_EDIT=1 and record the reason in a new ADR under docs/decisions/."
    fi
    ;;
esac

# --- Guard 2: large / binary artifacts in tracked paths ---------------------
if git -C "$root" check-ignore -q -- "$rel" 2>/dev/null; then
  exit 0  # gitignored path: it will never be committed, allow
fi

lower="$(printf '%s' "$rel" | tr '[:upper:]' '[:lower:]')"
case "$lower" in
  *.pt|*.pth|*.ckpt|*.safetensors|*.bin|*.onnx|*.npy|*.npz|*.pkl|*.pickle|\
  *.faiss|*.index|*.h5|*.hdf5|*.parquet|*.feather|*.arrow|*.zip|*.tar|*.gz|*.tgz|*.7z)
    deny "BLOCKED: $rel looks like a model/feature/data artifact and is not gitignored. Write it under data/ or results/runs/ (gitignored), not into tracked repo paths."
    ;;
esac

max_bytes=$((1024 * 1024))  # 1 MiB: larger text blobs belong in data/, not git
size="$(json_get '(.tool_input.content // "") | utf8bytelength' 'len(d.get("tool_input",{}).get("content","").encode())')"
if [ -n "$size" ] && [ "$size" -gt "$max_bytes" ] 2>/dev/null; then
  deny "BLOCKED: writing $size bytes to tracked path $rel (limit $max_bytes). Large generated data belongs under data/ or results/runs/ (gitignored)."
fi
exit 0
