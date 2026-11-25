#!/usr/bin/env bash
# Authenticated remote setup only. No credentials are embedded or printed.
set -euo pipefail
expected_owner='KuanWuChu'
fork_name='PSP-UAP-CrossArch'
parent='ChanhuiLee1111/PSP-UAP'
target="${expected_owner}/${fork_name}"
plan_branch='plan/crossarch-handoff-20261125'
source_tag='research/upstream-20261125'
source_sha='dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c'
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
cd -- "$repo_dir"
for cmd in git gh python3; do command -v "$cmd" >/dev/null || { echo "Missing dependency: $cmd" >&2; exit 2; }; done
gh auth status >/dev/null 2>&1 || { echo 'GitHub CLI is not signed in. Run gh auth login through the normal GitHub flow, then rerun.' >&2; exit 3; }
actual_owner="$(gh api user --jq .login)"
[[ "$actual_owner" == "$expected_owner" ]] || { echo "Account mismatch: expected $expected_owner, found $actual_owner" >&2; exit 4; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or stash local changes before publishing this handoff.' >&2; exit 5; }
git cat-file -e "${source_sha}^{commit}"
[[ "$(git rev-parse "${source_tag}^{commit}")" == "$source_sha" ]] || { echo "Source tag does not match the pinned upstream commit." >&2; exit 6; }
git show-ref --verify --quiet "refs/heads/${plan_branch}" || { echo 'Missing handoff branch; restore the supplied planning bundle first.' >&2; exit 6; }
python3 scripts/check_upstream_integrity.py

# Differentiate an absent repository from a generic API/network/permission failure.
probe="$(gh api --silent -i "repos/${target}" 2>&1)" && probe_status=0 || probe_status=$?
if [[ "$probe_status" -ne 0 ]]; then
  if [[ "$probe" == *'HTTP 404'* || "$probe" == *'HTTP/2.0 404'* || "$probe" == *'HTTP/1.1 404'* || "$probe" == *'(HTTP 404)'* ]]; then
    gh repo fork "$parent" --fork-name "$fork_name" --clone=false
  else
    echo 'Repository lookup failed for a reason other than 404; stopping without creating a replacement.' >&2
    printf '%s\n' "$probe" >&2
    exit 7
  fi
fi

# Native forks can take a few seconds to become readable. Bounded readiness retry.
info=''
for attempt in 1 2 3 4 5 6; do
  if info="$(gh api "repos/${target}" 2>/dev/null)"; then break; fi
  if [[ "$attempt" -lt 6 ]]; then sleep 2; fi
done
[[ -n "$info" ]] || { echo 'Fork creation/readiness not verified; inspect GitHub before retrying.' >&2; exit 8; }
printf '%s' "$info" | python3 -c '
import json,sys
r=json.load(sys.stdin)
assert r.get("fork") is True, "Destination exists but is not a native fork"
assert r.get("private") is False, "Destination is not public"
assert r.get("full_name")=="KuanWuChu/PSP-UAP-CrossArch", "Wrong destination"
assert r.get("parent",{}).get("full_name")=="ChanhuiLee1111/PSP-UAP", "Wrong fork parent"
print("Verified public native fork:",r["html_url"])
'

if git remote get-url upstream >/dev/null 2>&1; then
  up="$(git remote get-url upstream)"
  [[ "$up" == "https://github.com/${parent}.git" || "$up" == "https://github.com/${parent}" ]] || { echo 'Unexpected upstream remote; inspect before continuing.' >&2; exit 9; }
else
  git remote add upstream "https://github.com/${parent}.git"
fi

# A bundle clone has origin=<bundle path>, so replace only after verifying destination.
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/${target}.git"
else
  git remote add origin "https://github.com/${target}.git"
fi
gh auth setup-git
git fetch origin
git push -u origin "$plan_branch"
git push origin "refs/tags/${source_tag}"
printf '\nPublished planning branch. Next: open a docs-only PR against %s main.\n' "$target"
printf 'Repository: https://github.com/%s\nBranch: %s\n' "$target" "$plan_branch"
