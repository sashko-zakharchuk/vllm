"""Peak GPU memory of the DRY penalty path, as a function of batch size.

WHY THIS IS SHIPPED. The README quotes a table of peak figures, and a table without the script that
produced it is not reproducible: two reasonable ways of measuring differ by 30-40% here, depending on
whether the input tensors are inside or outside the baseline. This file fixes the methodology so the
numbers mean something. It measures the penalty call ALONE - inputs are allocated and the baseline is
taken after them, so what is reported is the transient the call itself adds.

    python peak_memory.py                 # the README's table
    python peak_memory.py --base 1.1      # the configuration test_peak_memory_bounded uses

The shipped unit test bounds this at 128 MiB for its own configuration; this is the wider sweep.
"""

import argparse
import torch

from vllm.v1.sample.dry_core import dry_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--base", type=float, default=1.75)
    ap.add_argument("--allowed", type=int, default=2)
    ap.add_argument("--alphabet", type=int, default=8,
                    help="token id range; small values guarantee repeats, so the penalty path runs")
    ap.add_argument("--batches", default="32,128,256,384")
    args = ap.parse_args()

    dev = torch.device("cuda")
    name = torch.cuda.get_device_properties(0).name
    # max_exponent as the worker computes it (vllm/v1/worker/gpu/sample/dry.py), not
    # sampling_params, which never touches it.
    from vllm.v1.sample.dry_core import _J_BUDGET
    from vllm.v1.sample.dry_utils import max_exponent

    max_exp = max_exponent(args.base)
    # dry_core itself does no routing: apply_dry decides between the vectorised path and a
    # sequential Python scan, and only the vectorised one is reachable from here. Refuse a base the
    # engine would send down the other path rather than silently reporting a cost it never pays.
    if not (max_exp > 0 and args.allowed + max_exp <= _J_BUDGET):
        raise SystemExit(
            f"  !! dry_base={args.base} gives max_exponent={max_exp}, which the engine routes to "
            f"the sequential path (allowed + max_exp must be <= {_J_BUDGET} and max_exp > 0). "
            f"This script measures the vectorised path only."
        )

    print(f"  {name}, vocab {args.vocab}, window {args.window}, dry_base {args.base}, "
          f"allowed {args.allowed}, alphabet {args.alphabet}")
    print(f"  {'R':>5} {'peak MiB':>10} {'B/(R*vocab)':>12} {'charged':>9}")
    for R in [int(x) for x in args.batches.split(",")]:
        torch.cuda.empty_cache()
        logits = torch.zeros(R, args.vocab, dtype=torch.float32, device=dev)
        W = torch.randint(0, args.alphabet, (R, args.window), dtype=torch.int64, device=dev)
        col = lambda v, dt=torch.int64: torch.full((R,), v, dtype=dt, device=dev)  # noqa: E731
        torch.cuda.synchronize()
        # Baseline AFTER the inputs exist: the figure is the call's own transient.
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = dry_core(logits, torch.arange(R, device=dev), W, col(args.window),
                       col(args.allowed), col(max_exp),
                       col(0.8, torch.float32), col(args.base, torch.float32), [None] * R)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        charged = int((out < 0).sum().item())
        print(f"  {R:>5} {peak / 2**20:>10.1f} {peak / (R * args.vocab):>12.1f} {charged:>9}")
        # REFUSE A VACUOUS MEASUREMENT. With a large --alphabet no repeat occurs, dry_core returns
        # at its `if not charged` guard, and the figure above is the match scan with the penalty
        # path never entered - indistinguishable from a real result unless it is checked. This is
        # the same defect the shipped unit test had; shipping it again in the script written to
        # make that test's claim reproducible would be worse than not shipping the script.
        if charged == 0:
            raise SystemExit(
                f"  !! no token was penalized at R={R}, so the penalty path did not run and this "
                f"figure means nothing. Lower --alphabet or raise --window."
            )
        del logits, W, out


if __name__ == "__main__":
    main()
