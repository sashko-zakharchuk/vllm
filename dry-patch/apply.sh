#!/usr/bin/env bash
# apply.sh - put the DRY patch onto a vLLM checkout, refusing anything it cannot do safely.
#
#   ./apply.sh /path/to/vllm            # checkout the pinned base, verify, apply
#   ./apply.sh /path/to/vllm --here     # apply to whatever is checked out, at your own risk
#
# Why a script rather than two lines of README: the patch is pinned to one upstream commit, and the
# failure people actually hit is applying it to a different one and getting a half-patched tree.
# `git apply` is all-or-nothing per invocation, so this checks first and tells you which files would
# have conflicted instead of leaving you to find out.
set -uo pipefail

BASE=410f6da5c4bb62010728502035bee1b5f0eab2ac
EXPECT_TREE=f90760a36f0ffc082ba38ea085e7ff357d6fe342
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH="$HERE/dry-sampling-vllm-410f6da5c4.patch"

REPO="${1:-}"
[ -n "$REPO" ] || { sed -n '2,8p' "$0"; exit 2; }
shift
MODE=pinned
for a in "$@"; do case "$a" in --here) MODE=here ;; *) echo "  !! unknown argument: $a"; exit 2 ;; esac; done

[ -f "$PATCH" ] || { echo "  !! patch not found beside this script: $PATCH"; exit 2; }
# Check the patch's own checksum FIRST. A tampered or truncated patch should fail at the door, not
# be caught later as a warning about a tree hash.
if [ -f "$HERE/SHA256SUMS" ]; then
  # Only the PATCH line. SHA256SUMS also covers this script, the README and the verify/ scripts so a
  # reader can check them, but an edited README is no reason to refuse to apply a patch whose bytes
  # are correct.
  if ! ( cd "$HERE" && grep -F "$(basename "$PATCH")" SHA256SUMS | sha256sum --quiet -c - ) 2>/dev/null; then
    echo "  !! $PATCH does not match SHA256SUMS. Refusing to apply a patch that is not the one"
    echo "     that was tested. Re-download it."
    exit 5
  fi
fi
cd "$REPO" 2>/dev/null || { echo "  !! no such directory: $REPO"; exit 2; }
# -e, not -d: in a git worktree .git is a file pointing at the real gitdir.
git rev-parse --git-dir >/dev/null 2>&1 || { echo "  !! not a git checkout: $REPO"; exit 2; }

# A dirty tree plus a failed apply is the one state that is genuinely annoying to unpick.
if [ -n "$(git status --porcelain)" ]; then
  echo "  !! $REPO has uncommitted changes. Commit or stash them first; a partial apply on top of"
  echo "     local edits is hard to undo."
  exit 3
fi

if [ "$MODE" = pinned ]; then
  if ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
    echo "  fetching the pinned base $BASE"
    git fetch --quiet origin "$BASE" 2>/dev/null || git fetch --quiet origin || true
  fi
  git cat-file -e "$BASE^{commit}" 2>/dev/null \
    || { echo "  !! cannot find base $BASE in $REPO. Is this a vllm-project/vllm clone?"; exit 3; }
  echo "  checking out the pinned base $BASE"
  git checkout --quiet --detach "$BASE" || exit 3
else
  echo "  --here: applying to $(git rev-parse --short HEAD), which is NOT the pinned base."
  echo "          Conflicts, if any, will be in sampling_params.py and the two protocol.py files."
  echo "          NOTE: the tree-hash check is SKIPPED in this mode, so nothing will confirm that"
  echo "          what you end up with matches the revision that was tested."
fi

echo "  checking the patch applies"
ERR=$(mktemp) || exit 4
if ! git apply --check "$PATCH" 2>"$ERR"; then
  echo "  !! it does not. Nothing has been changed. Details:"
  sed 's/^/     /' "$ERR"
  rm -f "$ERR"
  [ "$MODE" = here ] && echo "     Try without --here to use the pinned base $BASE."
  exit 4
fi
rm -f "$ERR"

git apply "$PATCH" || { echo "  !! apply failed after the check passed; tree may be partial."; exit 4; }
echo "  applied: $(git status --porcelain | wc -l | tr -d ' ') files touched"
git status --short | sed 's/^/     /'

if [ "$MODE" = pinned ]; then
  # Hash the applied tree and compare with the revision that was tested. `git add -A` honours
  # .gitignore, so this covers every TRACKED and non-ignored file; an ignored artifact (a stale .so,
  # a sitecustomize.py) is outside the guarantee. A dropped `git stash create` used to sit in front
  # of this: it changed the result not at all and only leaked a dangling commit.
  git add -A >/dev/null 2>&1 || { echo "  !! could not stage the tree to hash it"; exit 5; }
  TREE=$(git write-tree) || { echo "  !! could not hash the applied tree"; exit 5; }
  git reset --quiet   # unstage; the working tree keeps the patch
  if [ "$TREE" = "$EXPECT_TREE" ]; then
    echo "  tree matches the tested revision exactly (${EXPECT_TREE:0:14}); all tracked,"
    echo "  non-ignored files are the bytes that were tested"
  else
    echo "  !! tree is $TREE"
    echo "     expected $EXPECT_TREE"
    echo "     The patch applied but the result is NOT the tested tree. Do not build this."
    exit 5
  fi
fi

cat <<'EOF'

  Next: build vLLM as you normally would, for example
      VLLM_USE_PRECOMPILED=1 pip install -e .
  Then run the included suite before relying on it:
      pytest tests/v1/sample/test_dry.py
  Usage and the parameter table are in README.md beside this script.
EOF
