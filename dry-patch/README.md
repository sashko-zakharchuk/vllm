# DRY (Don't Repeat Yourself) sampling for vLLM: a source patch

DRY penalises tokens that would extend a token sequence already present in the context, with a
penalty growing exponentially in the repetition length. It is the usual fix for a model falling into
a loop. Parameter names and matching semantics follow llama.cpp; one default does not, and that is
called out below.

- **Patch:** `dry-sampling-vllm-410f6da5c4.patch`
- **Applies to:** vLLM commit `410f6da5c4bb62010728502035bee1b5f0eab2ac`, which was the tip of `main`
  on 2026-09-13 (committer date `05:56:14Z`). `main` has moved since; this patch is pinned to that
  commit and does not track it.
- **Size:** 9 files, +1586 lines, no deletions. Four files are new
  (`vllm/v1/sample/dry_core.py`, `vllm/v1/sample/dry_utils.py`,
  `vllm/v1/worker/gpu/sample/dry.py`, `tests/v1/sample/test_dry.py`) and five are existing upstream
  files modified (`vllm/sampling_params.py`, both OpenAI `protocol.py`,
  `vllm/v1/engine/input_processor.py`, `vllm/v1/worker/gpu/sample/sampler.py`).

## Apply

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 410f6da5c4bb62010728502035bee1b5f0eab2ac
git apply /path/to/dry-sampling-vllm-410f6da5c4.patch
VLLM_USE_PRECOMPILED=1 pip install -e .
```

`apply.sh /path/to/vllm` does the same with guard rails: it verifies the patch against `SHA256SUMS`,
refuses a dirty tree, refuses a base the patch does not fit (naming the files, changing nothing), and
afterwards confirms the resulting git tree hash equals the revision that was tested. It exits
non-zero on each of those.

On a base other than the pinned one, conflicts can appear in any of the five modified upstream files.
Counting upstream commits per file over the 2000 before the pinned base, the most churned are
`chat_completion/protocol.py` (14), `sampling_params.py` (12) and `completion/protocol.py` (11);
`input_processor.py` (9) and `sampler.py` (8) move least.

## Use

Offline:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="...")
out = llm.generate("...", SamplingParams(dry_multiplier=0.8))
```

Server, same names on `/v1/completions` and `/v1/chat/completions`:

```bash
curl localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "...", "prompt": "...", "max_tokens": 256, "dry_multiplier": 0.8
}'
```

| parameter | default here | meaning |
|---|---|---|
| `dry_multiplier` | `0.0` | Penalty multiplier. `0.0` disables DRY entirely. `0.8` is a typical enabled value. |
| `dry_base` | `1.75` | Exponential base. Below `1.0` disables, as in llama.cpp. |
| `dry_allowed_length` | `2` | Repetitions up to this length are not penalised. Bounded to `[0, 2147483647]`. |
| `dry_penalty_last_n` | `-1` | Context scanned for repetitions. `-1` scans everything, `0` disables, otherwise `[1, 2147483647]`. **This default differs from llama.cpp**, see below. |
| `dry_sequence_breakers` | `["\n", ":", "\"", "*"]` | Strings that interrupt sequence matching, so repetition is not tracked across lines or chat turns. Any vocabulary token whose text *contains* one of these acts as a breaker, following llama.cpp. At most 64 entries (`MAX_DRY_SEQUENCE_BREAKERS`, a rejection; llama.cpp has no such cap), each truncated to 40 code points (`dry_utils.py:_MAX_BREAKER_CHAR_LEN`) where llama.cpp truncates to 40 bytes - identical for ASCII breakers, not for others. |

**Where this diverges from llama.cpp.** `dry_multiplier`, `dry_base`, `dry_allowed_length` and
`dry_sequence_breakers` match `common/common.h` exactly. `dry_penalty_last_n` does not: llama.cpp
defaults it to **64** and rejects negative values outright (`common/arg.cpp`, "error: invalid
dry-penalty-last-n"), whereas the default here is `-1`, meaning scan the whole context. So a config
carried over from llama.cpp without setting this field gets a whole-context scan, which costs more and
penalises differently. Set it to `64` to match llama.cpp's behaviour.

## Removing it

The patch only adds; nothing upstream is deleted.

```bash
git checkout -- vllm/ tests/
rm -f vllm/v1/sample/dry_core.py vllm/v1/sample/dry_utils.py \
      vllm/v1/worker/gpu/sample/dry.py tests/v1/sample/test_dry.py
```

Rebuild afterwards. **Be aware that clients still sending `dry_*` fields will then be silently
ignored, not rejected**: vLLM's request models are declared `extra="allow"`
(`vllm/entrypoints/serve/engine/protocol.py:39`) and unknown fields are only logged at DEBUG. So a
caller that expects DRY then gets unpenalised output with no error. Check your callers before reverting.

## Requirements and limits

- **Model Runner V2 only**, which is vLLM's default runner. If something in your configuration has
  made the engine fall back to the V1 runner, a request that *enables* DRY is rejected with an error
  naming the problem. Note the gate is on truthiness: `dry_multiplier: 0.0` is not rejected. The
  startup log names the feature that caused the fallback.
- **Speculative decoding is refused, not silently skipped.** A request enabling DRY on an engine with
  a speculative config is rejected, alongside `min_p` and `logit_bias`. An earlier revision skipped it
  in the sampler instead, but that skip keys on the logits being draft-expanded, which is false on any
  step where no request happens to carry draft tokens, so DRY would have applied on some steps and not
  others, flickering with the schedule.
- **`dry_base` below 1.0 disables DRY**, as in llama.cpp. It is accepted, and logs a warning *when a
  non-zero `dry_multiplier` is also set*. Note the gap: sending only `{"dry_base": 0.8}`, where
  `dry_multiplier: 0.8` was meant, leaves the multiplier at its 0.0 default, so DRY is off and nothing
  warns - the multiplier being unset is indistinguishable from not wanting DRY.
- **CUDA only in practice.** Developed and measured on NVIDIA hardware; no other accelerator has been
  exercised.

## What has been verified, and under exactly what conditions

**The included suite is 32 passed in about 8 s** on an RTX 5050 at this patch's own base, and
`ruff check` and `ruff format --diff` are clean on all nine files. That run matters because upstream
moved 319 commits between the 2026-09-06 revision this code was last verified against and the
pinned base. (Against the revision it was *first* written on, 2026-07-31, the drift is 1866.)

**Applying it does not change output for a request that does not enable DRY**, measured end to end
rather than argued. Read the scope before relying on it:

```
model            Qwen2.5-0.5B-Instruct        prompts   5
decoding         greedy, seed 0               tokens    64 per prompt (320 total)
engine           enforce_eager=True: torch.compile and CUDA graphs DISABLED
sampler          VLLM_USE_FLASHINFER_SAMPLER=0, i.e. vLLM's native sampler only
hardware         one RTX 5050, one build, one process at a time

warm-up                        one discarded run (see the caveat below)
pristine vs pristine           IDENTICAL   establishes the engine repeats itself at all
patched/off vs pristine/off    IDENTICAL   the opt-out result
patched/on  vs patched/off     2 of 5 prompts differ   DRY is actually running
```

The last line is load-bearing: "identical with DRY off" is also what a patch whose code never
executed would produce.

**The caveat that matters most, because it nearly produced the opposite answer.** The observable here
is a sampled token id, and greedy argmax can turn on a near-tie. An earlier ordering of this same
experiment, with the patched arm running first on a cold machine, reported the patch as **not** inert:
one prompt in five diverged at token 19 and re-converged two tokens later. **The cause of that
divergence was never isolated.** A first write-up of this section blamed differently autotuned
flashinfer kernels, which the run's own log refutes: every arm of every run logs
`Loaded 0 configs from ...` and `Saved 0 configs ... (0 new, 0 from previous config)` in each of its
three arms, so nothing was read from or written to that cache in the run that actually diverged.

**No mechanism is offered, because two attempts at one were both wrong.** The first blamed autotuned
kernels, which the log above refutes. The second quoted a slow JIT step that turned out to belong to a
*different* run, one that crashed before emitting a token. Something about a cold first run mattered;
what, precisely, was never isolated, and the third guess is not going in a README.

What the harness does about it is what matters, and it does not depend on knowing the cause: it
discards a warm-up run before any measured arm, and it establishes that the engine reproduces itself
across processes before it will interpret a difference at all - refusing to return a verdict in either
direction if it does not. Measured that way, pristine-vs-pristine and patched-vs-pristine are
identical. Reproduce with `verify/opt-out-e2e.sh`, noting its requirements below.

**A two-arm control over eight upstream sampler suites** shows the patch adds no failure:
43 failed / 276 passed / 3 skipped on the patched tree and *the same 43 tests* on pristine
`410f6da5c4`, and identically again on two further pairs of runs, one with `scipy`
installed and `TORCH_CUDA_ARCH_LIST` set and one at the shipped code. 34 of the 43 are in `test_topk_topp_sampler.py`, which the patch does not
touch. We deliberately do not explain why those 43 fail: two attempts at an explanation were wrong,
and neither setting `TORCH_CUDA_ARCH_LIST` nor installing the missing `scipy` changed the count. The
control's job is only to show the same tests fail with and without the patch. You will likely see a
different set on your own machine.

**llama.cpp fidelity is checkable in the suite you get.** `test_worked_example` reproduces llama.cpp's
own commented example from `src/llama-sampler.cpp`; `test_differential_against_oracle` agrees with a
brute-force reimplementation sharing no code with the fast path; `test_exponent_clamp_float32_semantics`
and `test_double_pow_saturation` pin the float32 exponent clamp and the `std::pow(float, int)`-in-double
behaviour that llama.cpp has and a naive port loses. The llama.cpp side was read at revision
`30b6a755e2`. Earlier work also ran a differential against the llama.cpp *binary* with no
disagreements, but that harness and its logs no longer exist, so it is not offered as evidence.

**Peak GPU memory**, measured by `verify/peak_memory.py`, which is shipped here precisely so this
table is reproducible rather than asserted: vocab 128256, window 2048, `dry_base=1.75`,
`dry_allowed_length=2`, an 8-symbol alphabet so repeats are guaranteed, on an RTX 5050. The figure is
the penalty call's own transient, with the baseline taken after the inputs are allocated. Measuring
it the other way includes the input tensors and overstates the call by 23% at R=32 rising to 119% at
R=384 - the gap is the `[R, vocab]` logits tensor, which dominates at large batch. That is why the
methodology is stated and the script is shipped rather than the table alone. Note the unit test uses `dry_base=1.1` and reports 76.2 MiB at
R=32 for that reason.

```
   R      peak     bytes per (row x vocab)
  32     71 MiB          18.0
 128     99 MiB           6.4
 256    131 MiB           4.2
 384    164 MiB           3.5
```

Before that change the same measurement was a flat 41 bytes per entry: 1931 MiB at R=384, and about
5 GiB extrapolated to R=1024, which is an out-of-memory crash triggered by nothing worse than
concurrency. `test_peak_memory_bounded` pins a ceiling and now also asserts that tokens were actually
penalised, because its earlier window drew from a 1000-symbol alphabet where no repeat ever occurred,
so it had been measuring the match scan and never the penalty path it exists to bound.

## Known limits, measured, that are not fixed

**Do not expose DRY parameters directly to untrusted clients.** With trusted callers the numbers below
are the cost you are choosing.

All three were measured on one RTX 5050 at vocab 128256 by calling `dry_core` directly, not through a
serving benchmark, and **no log for them is shipped in this directory** - you would have to re-measure
to check them. That is weaker provenance than everything in the section above, and it is why the
numbers are given as orders of magnitude rather than as guarantees.

- **A `dry_base` from `1.0` up to about `1.044` routes to a pure-Python scan.** Those bases push the
  exponent cap past the vectorised path's budget, so the request falls back to a Z-algorithm in Python
  plus a full `.tolist()` of the window, per request per decode step, holding the GIL. About 23 ms per
  step at a 64k context, so sixteen such requests cost roughly 0.37 s per decode step for the whole
  engine. Every input involved is legal, and the window defaults to the whole context.
- **Steady-state cost scales with batch and window.** On a 2048-token window at `dry_base=1.75`: about
  6 ms per call at batch 128 and 13 ms at batch 256; a 4000-id breaker set roughly doubles it. Cost is
  linear in `dry_penalty_last_n`, which defaults to the whole context, so set a bound if you serve long
  ones.
- **The first request against a tokenizer pays a one-off vocabulary decode.** Resolving sequence
  breakers to token ids decodes every vocabulary entry once, synchronously on the API server's event
  loop: about 220 ms on a 50k-token vocabulary, more on a larger one. The decoded texts are then cached
  per tokenizer, so subsequent breaker sets cost only a containment scan rather than another decode.
  That scan is not free at scale: a few milliseconds for a handful of breakers on a 50k vocabulary, but
  around 0.27 s for the maximum 64 breakers on a 128k one, and it runs on the event loop every time a breaker set is
  not already cached, which includes but is not limited to evictions from the 64-entry cache. An earlier version of this section claimed the *decode* recurred per
  breaker set, which overstated the cost by about two orders of magnitude.

## Not verified, and you should assume it is untested

- **`/v1/chat/completions` has no test.** The patch adds the five fields to
  `chat_completion/protocol.py` and nothing exercises them; both REST tests use `CompletionRequest`.
- **No HTTP-level or serving test at all.** The `curl` example above is asserted, not exercised.
- **No concurrent, batched, or mixed-parameter test.** The end-to-end run is one model, five prompts,
  one request at a time. Behaviour when DRY-enabled and DRY-disabled requests share a batch is
  covered only by unit tests of the state object, not end to end.
- **No long-context or long-generation run.** 64 tokens per prompt is short for a loop-avoidance
  feature; the thing DRY exists to prevent mostly happens later than that.
- **No accelerator other than CUDA, and one card only.**
- **Nothing was run with CUDA graphs or `torch.compile`.** Every end-to-end arm used
  `enforce_eager=True`, and the unit tests call `apply_dry` directly. A real server runs with graphs
  enabled, and `apply_dry` performs a synchronising device-to-host copy (`pos[active_t].cpu()`) on the
  per-step path, so that combination is unexercised here.
- **`verify/opt-out-e2e.sh` will not run from the Apply recipe above.** It needs two *committed* git
  refs in one clone and refuses a dirty tree, whereas `git apply` leaves a dirty tree and no patched
  ref. Its defaults also point at the machine it was written on: clone path, venv, branch name, log
  directory, `TORCH_CUDA_ARCH_LIST=12.0` (sm120, wrong for other cards) and `HF_HUB_OFFLINE=1`. All are
  overridable via `DRY_VLLM_REPO`, `VLLM_DRY_VENV`, `PATCHED_REF`, `BASE_REF`, `DRY_LOG_DIR` and
  `DRY_GPU_NAME_MATCH`; on a multi-GPU host it refuses to pick a card until you set one of the last.
- **Sequence breakers are dropped under `skip_tokenizer_init`**, since there is no tokenizer to
  resolve them against. A warning is logged once per worker process, and a breaker set that
  resolves to no token ids at all is dropped without any warning.

### Running the suite yourself

`pip install -e .` does not pull the test dependencies. `tblib` is required before pytest can load
vLLM's `tests/conftest.py` at all:

```bash
pip install pytest tblib pytest-asyncio scipy
pytest tests/v1/sample/test_dry.py      # about eight seconds
```

## Status upstream

There is an open pull request, [#50584](https://github.com/vllm-project/vllm/pull/50584), for DRY in
vLLM under the same authorship, but **it is not this code**: its head is `0a35713770` from 2026-08-05,
it carries a second implementation for the V1 model runner in `vllm/v1/sample/logits_processor/`, and
it is currently conflicting with `main`. This patch is the later single-implementation rewrite against
the V2 runner. Do not read the PR diff expecting to see what you are applying.

The feature request, [#8581](https://github.com/vllm-project/vllm/issues/8581), has been open since
2024-09-18 with 23 thumbs-up and 31 comments, none of them from a vLLM maintainer. #50584 has had no
human review. This patch exists because that is a poor reason for users to go without the feature, not
as a complaint: vLLM has over five thousand open pull requests and maintainer attention is the scarce
resource.

DRY from the same effort is upstream and **merged** in exllamav3 as
[SS_DRY](https://github.com/turboderp-org/exllamav3/pull/278) (a different codebase and a separate
implementation), if that engine suits you.

## Licence

Same as vLLM: Apache-2.0. The patch's files carry vLLM's SPDX headers.

`SHA256SUMS` covers every file in this directory, not only the patch, so you can verify the scripts
you are being told to run.
