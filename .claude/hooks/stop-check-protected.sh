#!/usr/bin/env bash
# Stop hook: a cheap, non-blocking reminder. If committed files under splits/ or
# queries/ were modified (e.g. by a Bash command, which the PreToolUse guard does
# not see), show the user ONE warning per session per distinct change.
# Never blocks Claude from stopping.

source "$(dirname "$0")/_lib.sh" || exit 0
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$root" ] || exit 0

changed="$(git -C "$root" diff --name-only HEAD -- splits queries 2>/dev/null)"
[ -n "$changed" ] || exit 0

# Remember what we already warned about, so the message is not repeated every turn.
session="$(json_get '.session_id // "nosession"' 'd.get("session_id","nosession")')"
digest="$(printf '%s' "$changed" | cksum | cut -d' ' -f1)"
marker="${TMPDIR:-/tmp}/mmxai-stop-${session}-${digest}"
[ -e "$marker" ] && exit 0
: > "$marker" 2>/dev/null

files="$(printf '%s' "$changed" | tr '\n' ' ')"
msg="Cảnh báo: file split/query đã commit vừa bị thay đổi: ${files}. Kiểm tra lại (PDF §6) và ghi ADR nếu thay đổi là có chủ đích; nếu không, chạy: git restore -- splits queries"
printf '{"systemMessage":%s}\n' "$(json_escape "$msg")"
exit 0
