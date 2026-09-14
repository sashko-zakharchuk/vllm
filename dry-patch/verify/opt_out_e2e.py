"""Does the DRY patch change anything for someone who does not ask for it?

THE GUARANTEE THIS TESTS is the only one that matters to a stranger applying a patch to a running
server: with DRY left at its default, the patched build must emit exactly the tokens the unpatched
build emits. The in-patch suite checks the gate at unit level; this checks the whole engine.

TWO DIRECTIONS, because one of them alone proves nothing. "Identical with DRY off" is also what a
patch that never executes would produce, so the run with DRY ON must DIFFER. Absent that positive
control, a silently dead code path would pass.

Emits token ids as JSON. `opt-out-e2e.sh` runs this once per tree and compares; run standalone with
--arm off on a pristine build to produce the baseline.

    python opt_out_e2e.py --arm off --out /tmp/pristine.json
"""

import argparse
import json
import os
import sys

PROMPTS = [
    "Write a haiku about winter.",
    "def fibonacci(n):",
    "The capital of France is",
    "List three prime numbers:",
    # A prompt that invites repetition, so the DRY-on arm has something to bite on.
    "Repeat the following word forever: banana banana banana",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "on"],
                    help="off: default params, must match pristine. on: DRY enabled, must differ.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    # Recorded into the JSON so the verdict can assert the two "off" arms came from DIFFERENT code.
    # Without it nothing in the artifacts distinguishes the arms: vllm_version is baked in at
    # install time and is identical in every arm, pristine ones included.
    ap.add_argument("--head", default=None, help="git sha this arm's source tree was at")
    ap.add_argument("--repo", default=None,
                    help="the checkout this arm must import vllm from; asserted, not assumed")
    args = ap.parse_args()

    import torch

    # Assert the card in-process rather than trusting an index: CUDA orders devices fastest-first,
    # so device 0 is not necessarily the one that was selected. DRY_GPU_NAME_MATCH, when set, is the
    # substring the caller committed to; unset means the caller pinned CUDA_VISIBLE_DEVICES itself.
    name = torch.cuda.get_device_properties(0).name
    want = os.environ.get("DRY_GPU_NAME_MATCH")
    if want:
        assert want in name, f"REFUSING: device 0 is {name!r}, which does not match {want!r}"

    import vllm

    # THE ARM MUST BE RUNNING THE REPO'S CODE. An editable install hard-codes the directory it was
    # built from, so pointing this script at a different clone while using that venv makes every arm
    # import the SAME source: the two "off" arms then agree because nothing changed, the "on" arm
    # still differs because the venv happens to be patched, and the run reports PASS having proved
    # nothing. The verdict's head-comparison cannot catch it, since those shas come from git in the
    # repo rather than from the code that was imported.
    if args.repo:
        where = os.path.realpath(os.path.dirname(vllm.__file__))
        want = os.path.realpath(args.repo)
        assert where.startswith(want + os.sep) or where == want, (
            f"REFUSING: imported vllm from {where}, which is not inside {want}. "
            "This arm would not be measuring the checked-out code."
        )

    from vllm import LLM, SamplingParams

    # enforce_eager and a small context: this measures token equality, not throughput, and a
    # cudagraph capture on an 8 GB card is a long detour to the same answer.
    llm = LLM(model=args.model, enforce_eager=True, max_model_len=1024,
              gpu_memory_utilization=0.55, disable_log_stats=True)

    kw = dict(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    if args.arm == "on":
        # Values that bite: llama.cpp's typical multiplier, and a window over everything.
        kw.update(dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2, dry_penalty_last_n=-1)
    params = SamplingParams(**kw)

    outs = llm.generate(PROMPTS, params)
    record = {
        "arm": args.arm,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "vllm_version": __import__("vllm").__version__,
        "head": args.head,
        "device": name,
        # Token IDS, not text: detokenisation can mask a differing id, and ids are what the
        # sampler actually chose.
        "token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "prompts": PROMPTS,
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    total = sum(len(t) for t in record["token_ids"])
    print(f"  arm {args.arm}: {len(PROMPTS)} prompts, {total} tokens -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
