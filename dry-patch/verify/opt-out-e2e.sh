#!/usr/bin/env bash
# opt-out-e2e.sh - six engine runs that together prove the patch is inert unless asked for.
#
#   ./opt-out-e2e.sh [/path/to/vllm/clone]
#
# ARM 1  patched tree, DRY at its defaults   -> token ids
# ARM 2  pristine base, same request          -> token ids    MUST EQUAL arm 1
# ARM 3  patched tree, DRY enabled            -> token ids    MUST DIFFER from arm 1
#
# Arm 3 is not decoration. "Identical with DRY off" is also what a patch whose code never runs would
# produce, so without a positive control this script would bless a dead code path.
#
# The editable install points at the clone's source tree, so checking out a different commit swaps
# the code under the same venv. That is what makes arms 1 and 2 comparable: one build, one card, one
# model, one seed, and only the source differing.
set -uo pipefail

# Defaults suit the machine this was written on; every one is overridable, and the script says what
# it could not find rather than failing obscurely somewhere later.
REPO="${1:-${DRY_VLLM_REPO:-/home/karapuzy/vllm-rebase-20260906}}"
# Canonicalised, because the lock name is derived from it: /clone and /clone/ and a symlink and a
# relative path would otherwise take FOUR different locks on one directory, which is exactly the
# concurrent-checkout corruption the lock exists to stop.
# RAW is kept because the failure message must not itself expand an unset variable: with no
# argument and no DRY_VLLM_REPO, `${1:-$DRY_VLLM_REPO}` died with "unbound variable" under set -u and
# exited 1 instead of printing this. That is the first thing anyone not on the authoring machine saw.
RAW="$REPO"
REPO="$(cd "$REPO" 2>/dev/null && pwd -P)" || { echo "  !! no such directory: $RAW"; exit 2; }
VENV="${VLLM_DRY_VENV:-/tmp/vllm-dry-venv-20260913}"
if [ ! -x "$VENV/bin/python" ] && command -v python3 >/dev/null 2>&1; then
  # A plain `pip install -e .` in the caller's own environment is the normal case for anyone who is
  # not us; use it when the hardcoded venv is absent.
  python3 -c 'import vllm' >/dev/null 2>&1 && VENV=""
fi
PATCHED_REF="${PATCHED_REF:-dry-v2only-20260913}"
BASE_REF="${BASE_REF:-410f6da5c4bb62010728502035bee1b5f0eab2ac}"
HERE="$(cd "$(dirname "$0")" && pwd)"

PY="${VENV:+$VENV/bin/python}"; PY="${PY:-python3}"
"$PY" -c 'import vllm' >/dev/null 2>&1 \
  || { echo "  !! $PY cannot import vllm. Build it first, or set VLLM_DRY_VENV."; exit 2; }
( cd "$REPO" && git rev-parse --git-dir >/dev/null 2>&1 ) \
  || { echo "  !! not a git checkout: $REPO"; exit 2; }
# F3: a local edit to a tracked file rides silently into EVERY arm and makes the comparison
# meaningless. apply.sh already refuses a dirty tree; so must this.
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "  !! $REPO has uncommitted changes to tracked files. They would be present in every arm,"
  echo "     so nothing measured here could be attributed to the patch. Commit or stash first."
  exit 3
fi

# GPU SELECTION. Never an index: CUDA orders devices fastest-first, so index 0 is not necessarily
# the card you meant. Three ways in, in order of precedence:
#   CUDA_VISIBLE_DEVICES already set   -> respected as-is, you have chosen
#   DRY_GPU_NAME_MATCH=<substring>     -> the UUID of the first card whose name matches
#   neither, and exactly one GPU       -> that one
#   neither, and several GPUs          -> REFUSE, because guessing could land on a card in use
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _gpus=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader 2>/dev/null)
  [ -n "$_gpus" ] || { echo "  !! no NVIDIA GPU visible"; exit 3; }
  if [ -n "${DRY_GPU_NAME_MATCH:-}" ]; then
    CUDA_VISIBLE_DEVICES=$(printf '%s\n' "$_gpus" | grep -F "$DRY_GPU_NAME_MATCH" | head -1 | cut -d, -f1)
    [ -n "$CUDA_VISIBLE_DEVICES" ] \
      || { echo "  !! no GPU whose name contains '$DRY_GPU_NAME_MATCH'"; exit 3; }
  elif [ "$(printf '%s\n' "$_gpus" | wc -l)" = 1 ]; then
    CUDA_VISIBLE_DEVICES=$(printf '%s\n' "$_gpus" | cut -d, -f1)
  else
    echo "  !! this machine has several GPUs and none was chosen. Refusing to guess, because the"
    echo "     wrong one may be in use. Set CUDA_VISIBLE_DEVICES to a UUID, or"
    echo "     DRY_GPU_NAME_MATCH to a substring of the card's name. Available:"
    printf '%s\n' "$_gpus" | sed 's/^/       /'
    exit 3
  fi
  export CUDA_VISIBLE_DEVICES
fi
echo "   gpu: $CUDA_VISIBLE_DEVICES"
[ -n "${HF_HOME:-}" ] && export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_LOGGING_LEVEL=WARNING
# FlashInfer's JIT aborts here with "requires GPUs with sm75 or higher", which is a misleading
# message: this card is sm120 and flashinfer's own check passes anything with major >= 8. It fails
# because current_compilation_context.TARGET_CUDA_ARCHS comes back EMPTY, so its `eligible` flag is
# never set. TORCH_CUDA_ARCH_LIST fills that in. Independently we pin vLLM to its own sampler, which
# is both the path DRY actually hooks and one fewer JIT between the test and the answer.
export VLLM_USE_FLASHINFER_SAMPLER=0
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

# Logs land on the NAS here so the other machine can read them; anywhere writable will do for
# anyone else, and the default falls back rather than refusing.
if [ -n "${DRY_LOG_DIR:-}" ]; then
  LOGDIR="$DRY_LOG_DIR"
elif [ -d "${ZABA_NAS:-/mnt/nas}/zaba/logs" ]; then
  LOGDIR="${ZABA_NAS:-/mnt/nas}/zaba/logs"
else
  LOGDIR="${TMPDIR:-/tmp}/dry-verify-logs"
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
# Resolve the refs and put the ACTUAL shas in the filename. A hardcoded base string used to sit here
# and a run with BASE_REF overridden wrote a log asserting a base it had not measured.
BASE_SHA=$(git -C "$REPO" rev-parse --short=10 "$BASE_REF" 2>/dev/null || echo UNKNOWN)
PATCHED_SHA=$(git -C "$REPO" rev-parse --short=10 "$PATCHED_REF" 2>/dev/null || echo UNKNOWN)
if [ "$BASE_SHA" = UNKNOWN ] || [ "$PATCHED_SHA" = UNKNOWN ]; then
  echo "  !! cannot resolve BASE_REF=$BASE_REF or PATCHED_REF=$PATCHED_REF in $REPO"; exit 2
fi
if [ "$BASE_SHA" = "$PATCHED_SHA" ]; then
  echo "  !! BASE_REF and PATCHED_REF resolve to the SAME commit ($BASE_SHA). Every arm would run"
  echo "     identical code and the comparison would pass for no reason. Refusing."
  exit 2
fi
TAG="dry-optout-e2e-qwen2.5-0.5b-greedy-seed0-max64-5prompts-base$BASE_SHA-patched$PATCHED_SHA-$STAMP-$(hostname -s)"
LOG="$LOGDIR/$TAG.log"
mkdir -p "$LOGDIR" || exit 3

say() { echo "$@" | tee -a "$LOG"; }
# A LOCK, because two runs sharing one clone destroy each other. Each arm checks out a different
# ref in $REPO, so a second invocation swaps the tree under the first one's engine: it imports a
# mixture and dies with something like "'SamplingParams' object has no attribute 'dry_multiplier'"
# on the PRISTINE arm, which reads as a patch defect and is not one. Observed 2026-09-13, from
# relaunching without checking the first run had exited.
LOCK="${TMPDIR:-/tmp}/.dry-optout-e2e.$(echo "$REPO" | tr -c 'A-Za-z0-9' '_').lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "  !! another opt-out run holds $LOCK for $REPO."
  echo "     Two runs would check out different refs in the same clone and corrupt each other."
  echo "     Wait for it, or remove the directory if you are sure it is stale."
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# The log is opened only AFTER the lock is held: a blocked second invocation used to leave a
# header-only log on the NAS for a run that never happened.
{ echo "# command : $0 $*"; echo "# when    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "# repo    : $REPO"
  echo "# venv    : $VENV"; echo "# gpu     : $CUDA_VISIBLE_DEVICES"
  echo "# patched : $PATCHED_REF   base: $BASE_REF"; echo "#"; } > "$LOG"

ORIG=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)
[ "$ORIG" = HEAD ] && ORIG=$(git -C "$REPO" rev-parse HEAD)
restore() {
  git -C "$REPO" checkout -q "$ORIG" 2>/dev/null \
    || echo "  !! COULD NOT RESTORE $REPO to $ORIG; it is left at $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null). Fix before the next run." >&2
}
trap 'restore; rmdir "$LOCK" 2>/dev/null' EXIT

run_arm() {  # run_arm <ref> <arm> <outfile>
  # --detach and the trailing -- matter: `git checkout -q <path>` RETURNS 0 WITHOUT MOVING HEAD,
  # so a typo naming a tracked file instead of a ref would leave the previous arm's code in place
  # and the comparison would silently be code-against-itself.
  git -C "$REPO" checkout -q --detach "$1" -- 2>/dev/null \
    || git -C "$REPO" checkout -q --detach "$1" \
    || { say "  !! cannot check out $1"; return 4; }
  local head; head=$(git -C "$REPO" rev-parse HEAD)
  ( cd "$REPO" && PYTHONUNBUFFERED=1 "$PY" "$HERE/opt_out_e2e.py" \
      --arm "$2" --out "$3" --head "$head" --repo "$REPO" ) 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  return $rc
}

# The per-arm token files go BESIDE THE LOG ON THE NAS and are not deleted. A first version put
# them in /tmp and removed them in the trap, which threw away the evidence at exactly the moment the
# comparison failed and there was something to look at.
A="$LOGDIR/$TAG.arm1-patched-off.json"
B="$LOGDIR/$TAG.arm2-pristine-off.json"
C="$LOGDIR/$TAG.arm3-patched-on.json"

say "== DRY opt-out, end to end =="
say ""
# WARM UP FIRST, and throw the result away. This is not hygiene, it is the difference between a
# correct verdict and a wrong one. A first version of this script reported the patch as NOT inert:
# arm 1 (patched) ran first on a cold machine and one prompt in five diverged at token 19, then
# re-converged. Re-running with warm caches made the arms byte-identical. THE CAUSE WAS NEVER
# ISOLATED - an early explanation blaming flashinfer autotune configs is refuted by the log, which
# shows `Loaded 0 configs` in every arm - so the only defensible statement is that something about a
# cold first run mattered and the arm ORDER was the variable. Anything comparing sampled token ids
# across separate engine processes has to control for it, which is what this run does.
say "---- warm-up: one discarded run, so no measured arm pays first-run autotuning"
run_arm "$BASE_REF" off "$LOGDIR/$TAG.warmup-discarded.json" || exit 4

say ""
say "---- arm 0: PRISTINE base twice, to establish the engine repeats itself"
say "     (without this, a difference between arms 1 and 2 cannot be attributed to the patch)"
D="$LOGDIR/$TAG.arm0a-pristine-off.json"
E="$LOGDIR/$TAG.arm0b-pristine-off.json"
run_arm "$BASE_REF" off "$D" || exit 4
run_arm "$BASE_REF" off "$E" || exit 4
# Paths via argv, not interpolated into the program text: a path containing a quote used to turn
# this into a SyntaxError, which the old code then read as "the engine is nondeterministic".
"$PY" -c '
import json,sys
a=json.load(open(sys.argv[1]))["token_ids"]; b=json.load(open(sys.argv[2]))["token_ids"]
sys.exit(0 if a==b else 1)' "$D" "$E"
DET_RC=$?
if [ "$DET_RC" = 0 ]; then
  say "     engine is reproducible across processes: arm 0a == arm 0b"
  DETERMINISTIC=1
else
  # Python exits 1 both when the arms differ AND on an uncaught exception, so a corrupt or missing
  # arm-0 record is indistinguishable here from real nondeterminism. Both must stop the run, so the
  # conflation is safe, but the message must not assert a cause it cannot know.

  say "     !! the two pristine arms did not match: either the engine does not repeat itself across"
  say "        processes, or an arm-0 record is missing or corrupt. Either way token equality has no"
  say "        power here. The remaining arms are still run and reported, but no difference between"
  say "        them would be evidence about the patch, and the verdict will be INCONCLUSIVE."
  DETERMINISTIC=0
fi

say ""
say "---- arm 1: PATCHED tree ($PATCHED_REF), DRY at defaults"
run_arm "$PATCHED_REF" off "$A" || exit 4
say ""
say "---- arm 2: PRISTINE base ($BASE_REF), same request"
run_arm "$BASE_REF" off "$B" || exit 4
say ""
say "---- arm 3: PATCHED tree, DRY ENABLED (positive control)"
run_arm "$PATCHED_REF" on "$C" || exit 4

say ""
say "---- verdict"
"$PY" - "$A" "$B" "$C" "$DETERMINISTIC" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
a, b, c = (json.load(open(p)) for p in sys.argv[1:4])
deterministic = sys.argv[4] == "1"
rc = 0
fatal = []

# F4: a comparison of two empty lists is "identical" and proves nothing. Everything the records
# already carry is now asserted, rather than carried and ignored.
def tok(r):
    return r["token_ids"]

for name, r, want_arm in (("arm1", a, "off"), ("arm2", b, "off"), ("arm3", c, "on")):
    if r.get("arm") != want_arm:
        fatal.append(f"{name} is arm={r.get('arm')!r}, expected {want_arm!r}")
    t = tok(r)
    if not t or min((len(x) for x in t), default=0) == 0:
        fatal.append(f"{name} has an empty token list: {[len(x) for x in t]}")
if not (len(tok(a)) == len(tok(b)) == len(tok(c))):
    fatal.append(
        f"arms have different prompt counts: {[len(tok(r)) for r in (a, b, c)]}; zip would "
        "silently truncate the comparison")
for field in ("model", "max_tokens", "device"):
    vals = {r.get(field) for r in (a, b, c)}
    if len(vals) != 1:
        fatal.append(f"arms disagree on {field}: {vals}")
# The two off-arms must come from DIFFERENT code, and arm3 from the same code as arm1.
# A missing head is itself fatal: guarding these on `a.get("head") and ...` meant a record written
# without --head (which running opt_out_e2e.py standalone produces) silently switched off the very
# check added to catch arm mixing.
if not all(r.get("head") for r in (a, b, c)):
    fatal.append("an arm did not record its git sha, so arm mixing cannot be ruled out")
elif a["head"] == b["head"]:
    fatal.append("arm1 and arm2 ran the SAME commit; the pristine arm never ran")
elif a["head"] != c["head"]:
    fatal.append("arm3 ran a different commit from arm1; the positive control is not comparable")

if fatal:
    print("  !! the run is not interpretable:")
    for f in fatal:
        print(f"     - {f}")
    print("  INVALID")
    sys.exit(3)

ta, tb, tc = tok(a), tok(b), tok(c)
same_off = ta == tb
n = sum(len(t) for t in ta)
print(f"  arm1 patched/off vs arm2 pristine/off: {'IDENTICAL' if same_off else 'DIFFER'}"
      f"  ({len(ta)} prompts, {n} tokens)")
if not same_off:
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            j = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print(f"    prompt {i}: first difference at token {j}: {x[j:j+3]} vs {y[j:j+3]}")

changed = sum(1 for x, y in zip(ta, tc) if x != y)
print(f"  arm3 patched/on vs arm1 patched/off:  {changed}/{len(ta)} prompts changed"
      f"   ({'DRY is live' if changed else 'DRY CHANGED NOTHING'})")
if changed == 0:
    print("    !! the positive control failed: with dry_multiplier=0.8 the output is unchanged, so")
    print("       arm1 == arm2 would prove only that the code did not run.")
    rc = 1

# THE VERDICT. Three outcomes, not two. A harness that cannot say "I failed to establish this"
# launders a non-result into a pass, and an earlier version of this script did exactly that: it
# printed INCONCLUSIVE and then PASS, exit 0, on a run where patched/off genuinely differed from
# pristine/off.
# DETERMINISM GATES EVERYTHING. A previous version tested it only on the DIFFER branch, so an
# engine that could not repeat itself still got "PASS: opt-out proved" when the arms happened to
# agree - a green light for a guarantee the run was incapable of testing, printed moments after the
# script itself said token equality has no power here. That asymmetry was in the block rewritten to
# close the FIRST false pass, which is why this check now comes before any verdict at all.
if not deterministic:
    print("  INCONCLUSIVE: the engine did not repeat itself across processes, so token equality")
    print("     cannot separate the patch from run-to-run variation - in either direction. This is")
    print("     NOT a pass. A different observable is needed (same-process comparison, or logits")
    print("     rather than sampled ids).")
    sys.exit(2)
if not same_off:
    print("  FAIL: the patch changes output for a request that does not enable DRY.")
    sys.exit(1)
if rc:
    print("  FAIL")
    sys.exit(1)
print("  PASS: opt-out proved, and the positive control shows the code runs.")
sys.exit(0)
PYEOF
RC=${PIPESTATUS[0]}
say ""
say "   log: $LOG"
exit "$RC"
