"""End-to-end throughput with DRY off and on, which is the question a reviewer asks first.

TWO COSTS, and they are not the same question.

THIS MEASURES ONE COST ONLY: DRY on against DRY off, within a single patched build, which is what a
user opts into by enabling the feature. It does NOT measure what the patch costs when DRY is unused -
that needs a pristine build, which this script cannot load, because SamplingParams would reject the
dry_* fields. Do not read the "off" column as "unpatched".

FAIR COMPARISON, deliberately. ignore_eos=True with a fixed max_tokens so both arms emit exactly the
same token count: with EOS live, DRY changes which tokens are produced and therefore when the
sequence stops, and a throughput ratio would silently be comparing different amounts of work.

Three warm-up pairs are discarded before each batch's measured repetitions. A cold first run on this
rig has already produced a wrong answer once, in the opt-out harness, and this card idles at 225 MHz
against a 3135 MHz ceiling.

    python throughput.py --batches 1,8,32 --reps 3
"""

import argparse
import json
import os
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = torch.cuda.get_device_properties(0).name
    want = os.environ.get("DRY_GPU_NAME_MATCH")
    if want:
        assert want in name, f"REFUSING: device 0 is {name!r}, not matching {want!r}"

    from vllm import LLM, SamplingParams

    # PREFIX CACHING OFF. It defaults to True, and a first version of this script also sent the
    # SAME prompt to every batch slot - so vLLM deduplicated the prefill, every repetition hit a
    # warm cache, and the timings measured cache state rather than DRY. That run reported DRY as
    # 126% FASTER than no DRY at batch 8, which is impossible and is what exposed the mistake.
    llm = LLM(model=args.model, enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.75, disable_log_stats=True,
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    # A repetitive prompt, so DRY has matches to find: measuring its cost on text where it charges
    # nothing would understate the work it does.
    base = "The quick brown fox jumps over the lazy dog. " * 200
    ids = tok.encode(base)[: args.prompt_tokens]
    prompt = tok.decode(ids)

    def prompts_for(batch):
        # DISTINCT per slot, so nothing is shared even if caching is re-enabled later. The varying
        # part goes at the FRONT, where a shared-prefix optimisation would otherwise bite.
        return [f"[{i}] " + prompt for i in range(batch)]

    def run(batch, dry):
        kw = dict(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True, seed=0)
        if dry:
            kw.update(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2,
                      dry_penalty_last_n=-1)
        params = SamplingParams(**kw)
        prompts = prompts_for(batch)
        t = time.perf_counter()
        res = llm.generate(prompts, params, use_tqdm=False)
        dt = time.perf_counter() - t
        n = sum(len(r.outputs[0].token_ids) for r in res)
        assert n == batch * args.max_tokens, f"expected {batch * args.max_tokens} tokens, got {n}"
        return n / dt, [list(r.outputs[0].token_ids) for r in res]

    rows = []
    print(f"  {name}, {args.model}, prompt {args.prompt_tokens}, gen {args.max_tokens}, "
          f"ignore_eos, median of {args.reps}")
    print(f"  {'batch':>6} {'off tok/s':>11} {'on tok/s':>10} {'delta':>8} {'spread':>8}  {'DRY live':>8}")
    for b in [int(x) for x in args.batches.split(",")]:
        # Interleaved, not all-off-then-all-on: a machine that drifts during the run would
        # otherwise attribute the drift to DRY.
        # THREE discarded warm-ups per arm, not one. This card idles at 225 MHz against a
        # 3135 MHz maximum, so a short run spends most of its time ramping the clock: with one
        # warm-up the first measured repetition came in 2-3x slow and the run-to-run spread
        # reached 78%, far above the effect being measured. Long generations and back-to-back
        # repetitions keep the clock up.
        for _ in range(3):
            run(b, False); run(b, True)
        offs, ons, tok_off, tok_on = [], [], None, None
        for _ in range(args.reps):
            o, tok_off = run(b, False)
            n_, tok_on = run(b, True)
            offs.append(o); ons.append(n_)
        off, on = statistics.median(offs), statistics.median(ons)
        # Each arm's range against ITS OWN median. Normalising both by the off-arm median
        # understated the on-arm spread and made this guard fire less often than it should.
        spread = 100 * max((max(offs) - min(offs)) / off, (max(ons) - min(ons)) / on)
        # DRY must actually have changed the output, or the delta is the cost of a code path that
        # did nothing and the comparison is worthless. An earlier draft of this check compared a
        # list against None, which is true unconditionally and would have certified anything.
        # EVERY slot, not just the first. Checking res[0] alone certified a batch-32 row from one
        # prompt, which is the same shape of error as the `list(...) != None` guard this replaced.
        n_changed = sum(1 for x, y in zip(tok_off, tok_on) if x != y)
        changed = n_changed == len(tok_off)
        delta = 100.0 * (on - off) / off
        rows.append({"batch": b, "off_tok_s": off, "on_tok_s": on, "delta_pct": delta,
                     "off_all": offs, "on_all": ons, "spread_pct": spread,
                     "dry_changed_output": changed,
                     # Recorded, not merely printed: every row this script has ever produced was
                     # noise-dominated, and a consumer reading delta_pct out of the JSON had no way
                     # to know that.
                     "noise_dominated": spread > abs(delta)})
        print(f"  {b:>6} {off:>11.1f} {on:>10.1f} {delta:>7.1f}%  {spread:>7.1f}%  "
              f"{'yes' if changed else 'NO':>8}")
        if spread > abs(delta):
            print(f"         !! run-to-run spread ({spread:.1f}%) exceeds the effect "
                  f"({abs(delta):.1f}%); this row is noise, not a measurement")
        if not changed:
            raise SystemExit(
                f"  !! at batch {b} only {n_changed}/{len(tok_off)} prompts changed with DRY on, "
                f"so this row times a penalty path that did not charge on every slot. Use a more "
                f"repetitive prompt or a longer generation."
            )

    if args.out:
        json.dump({"device": name, "model": args.model, "prompt_tokens": args.prompt_tokens,
                   "max_tokens": args.max_tokens, "reps": args.reps, "rows": rows},
                  open(args.out, "w"), indent=1)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
