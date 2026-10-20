#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): auto-format + auto-fix a single edited .py file
# with the project's ruff. Always exits 0: formatting help must never block work.
# Remaining (unfixable) lint issues are sent back to Claude as additionalContext.

source "$(dirname "$0")/_lib.sh" || exit 0

file="$(json_get '.tool_input.file_path // empty' 'd.get("tool_input",{}).get("file_path")')" || exit 0
case "$file" in *.py) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(repo_root_for "$file")}"

# Prefer the ruff pinned in the project venv (same version as `uv run ruff`),
# then a ruff on PATH. We do NOT call `uv run` here: it may trigger a slow sync.
if [ -x "$root/.venv/bin/ruff" ]; then
  ruff_bin="$root/.venv/bin/ruff"
elif [ -x "$root/.venv/Scripts/ruff.exe" ]; then
  ruff_bin="$root/.venv/Scripts/ruff.exe"
elif command -v ruff >/dev/null 2>&1; then
  ruff_bin="ruff"
else
  exit 0  # no ruff available (env not synced yet): silently skip
fi

cd "$root" 2>/dev/null || exit 0
# Order matters: fix first (e.g. drop unused imports), then format the result,
# then a read-only check to report whatever could not be auto-fixed.
"$ruff_bin" check --fix --quiet "$file" >/dev/null 2>&1
"$ruff_bin" format --quiet "$file" >/dev/null 2>&1
remaining="$("$ruff_bin" check --quiet --output-format concise "$file" 2>&1 | head -n 20)"

if [ -n "$remaining" ]; then
  msg="ruff check --fix still reports issues in $file (fix them, keep code simple):
$remaining"
  printf '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":%s}}\n' "$(json_escape "$msg")"
fi
exit 0
