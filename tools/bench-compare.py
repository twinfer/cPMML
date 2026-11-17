#!/usr/bin/env python3
"""The performance gate of docs/design/typed-core.md §6.4, over the raw log
that tools/bench-ab.sh writes.

    bench-compare.py RAW [--rerun RERUN] [--fail-list FILE] [--seed N]

Each line of RAW (and RERUN) is

    <build> <mode> <round> <side> <model> <model_benchmark.exe output>

with side "base" or "head"; build is "seq", "omp" or "build" and mode "eval" or
"read-all". Every (build, mode) group is judged on its own:

  * per model, r = median(head) / median(base) of the per-process mean
    evaluate time, and its 95% bootstrap interval (4 000 resamples, each side
    resampled independently). FAIL when the lower bound exceeds 1.03;
  * load time the same way, FAIL when the lower bound exceeds 1.10;
  * the geometric mean of the evaluate ratios over every model, with a 95%
    interval (4 000 resamples) from resampling rounds jointly: both sides of a
    model share each draw, every model draws uniformly over its own rounds,
    and the draws that fall in the rounds all models have are shared across
    models, so noise shared by a sweep stays paired (see joint_geomean). FAIL
    when it exceeds 1.01 and its lower bound exceeds 1.00.

A model that fails in RAW is judged again on RERUN alone when RERUN holds
fresh rounds for it ("re-run once; it fails the gate only if it fails again"),
and those fresh rounds also replace its first ones in the geomean.
Without RERUN, --fail-list FILE receives one "<build> <mode> <model>" line per
failing model, so that the caller can re-run exactly those.

"wide" after a verdict marks an evaluate interval wider than ±5%; in an A/A
run (the same build as base and head) such a model is a NOISY candidate in
benchmark/CMakeLists.txt.

Exit status: 0 the gate passes, 1 it fails (or models await a re-run), 2 the
input is unusable. Stdlib only; the seed is fixed so a report is repeatable.
"""

import argparse
import collections
import math
import random
import re
import statistics
import sys

EVAL_LIMIT = 1.03
LOAD_LIMIT = 1.10
GEO_LIMIT = 1.01
GEO_LOWER = 1.00
WIDE = 0.05  # an A/A interval wider than ±5% marks a model NOISY

LINE = re.compile(
    r"^(\S+) (\S+) (\d+) (base|head) (\S+)\s+Loading time: ([\d.]+) ms\s+Evaluate time: ([\d.]+) us"
)

med = statistics.median


def fatal(message):
    print(f"bench-compare.py: {message}", file=sys.stderr)
    sys.exit(2)


def parse(path):
    """{(build, mode): {model: {side: {round: (load_ms, eval_us)}}}}, model order."""
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: {"base": {}, "head": {}}))
    order = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LINE.match(line)
            if not m:
                fatal(f"{path}:{number}: not a benchmark line: {line!r}")
            build, mode, rnd, side, model, load, ev = m.groups()
            data[(build, mode)][model][side][int(rnd)] = (float(load), float(ev))
            if model not in order:
                order.append(model)
    return data, order


def ratio_ci(base, head, rng, n):
    """Ratio of medians head/base and its 95% interval, each side resampled on its own."""
    r = med(head) / med(base)
    rs = sorted(med(rng.choices(head, k=len(head))) / med(rng.choices(base, k=len(base))) for _ in range(n))
    return r, rs[int(0.025 * n)], rs[int(0.975 * n)]


def joint_geomean(models, rng, n):
    """Geomean of eval ratios and its 95% interval, rounds resampled jointly.

    A model with k paired rounds draws k of them per resample, uniformly over
    its own k, and base and head always use the same draws. Across models the
    draws are coupled: let c be the fewest rounds any model has (15; the first
    c rounds of every model ran in the same sweeps). Draw j of a model picks,
    with probability c/k, the shared round s_j (uniform over the first c,
    the same s_j for every model), and otherwise round c + floor(t_j (k - c))
    with t_j uniform in [0, 1). Each round thus has probability 1/k, and noise
    shared by a sweep stays paired across models wherever their draws fall in
    the common rounds."""

    def geo(sample_of):
        logs = []
        for rounds, base, head in models:
            idx = sample_of(len(rounds))
            logs.append(math.log(med([head[rounds[i]] for i in idx]) / med([base[rounds[i]] for i in idx])))
        return math.exp(sum(logs) / len(logs))

    common = min(len(rounds) for rounds, _, _ in models)
    longest = max(len(rounds) for rounds, _, _ in models)
    point = geo(lambda k: range(k))
    rs = []
    for _ in range(n):
        shared = [rng.randrange(common) for _ in range(longest)]
        coin = [rng.random() for _ in range(longest)]
        tail = [rng.random() for _ in range(longest)]

        def sample(k, shared=shared, coin=coin, tail=tail):
            return [shared[j] if coin[j] * k < common else common + int(tail[j] * (k - common)) for j in range(k)]

        rs.append(geo(sample))
    rs.sort()
    return point, rs[int(0.025 * n)], rs[int(0.975 * n)]


def paired(entry):
    """Rounds present on both sides, sorted."""
    return sorted(set(entry["base"]) & set(entry["head"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--rerun", help="fresh rounds for the models that failed in RAW")
    ap.add_argument("--fail-list", help="write the failing (build mode model) triples here")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resamples", type=int, default=4000)
    args = ap.parse_args()

    data, order = parse(args.raw)
    rerun = parse(args.rerun)[0] if args.rerun else {}
    if not data:
        fatal(f"{args.raw}: no benchmark lines")

    failed = False
    pending = []
    for group in sorted(data):
        build, mode = group
        rng = random.Random(args.seed)
        models = data[group]
        print(f"== build {build}, mode {mode}: head/base, ratio of medians [95% bootstrap interval]")
        print(
            f"{'model':38s} {'n':>3} {'base us':>9} {'head us':>9} {'eval r':>7} {'interval':>17}  "
            f"{'load r':>7} {'interval':>17}  verdict"
        )
        geo_input = []
        for model in [m for m in order if m in models]:
            entry = models[model]
            rounds = paired(entry)
            if len(rounds) < 3:
                fatal(f"{model} ({build}, {mode}): {len(rounds)} paired rounds; need at least 3")
            base_ev = [entry["base"][r][1] for r in rounds]
            head_ev = [entry["head"][r][1] for r in rounds]
            base_ld = [entry["base"][r][0] for r in rounds]
            head_ld = [entry["head"][r][0] for r in rounds]
            er, elo, ehi = ratio_ci(base_ev, head_ev, rng, args.resamples)
            lr, llo, lhi = ratio_ci(base_ld, head_ld, rng, args.resamples)

            notes = []
            fails = []
            if elo > EVAL_LIMIT:
                fails.append("eval")
            if llo > LOAD_LIMIT:
                fails.append("load")
            if elo < 1 - WIDE or ehi > 1 + WIDE:
                notes.append("wide")
            verdict = "ok"
            if fails:
                again = rerun.get(group, {}).get(model)
                if again is None:
                    verdict = "FAIL (" + ",".join(fails) + "; re-run pending)"
                    pending.append(f"{build} {mode} {model}")
                    failed = True
                else:
                    rr = paired(again)
                    if len(rr) < 3:
                        fatal(f"{model} ({build}, {mode}): re-run has {len(rr)} paired rounds; need at least 3")
                    still = []
                    if "eval" in fails:
                        r2, lo2, hi2 = ratio_ci(
                            [again["base"][r][1] for r in rr], [again["head"][r][1] for r in rr], rng, args.resamples
                        )
                        notes.append(f"re-run eval {r2:.3f} [{lo2:.3f},{hi2:.3f}]")
                        if lo2 > EVAL_LIMIT:
                            still.append("eval")
                    if "load" in fails:
                        r2, lo2, hi2 = ratio_ci(
                            [again["base"][r][0] for r in rr], [again["head"][r][0] for r in rr], rng, args.resamples
                        )
                        notes.append(f"re-run load {r2:.3f} [{lo2:.3f},{hi2:.3f}]")
                        if lo2 > LOAD_LIMIT:
                            still.append("load")
                    if still:
                        verdict = "FAIL (" + ",".join(still) + ", confirmed by re-run)"
                        failed = True
                    else:
                        verdict = "ok after re-run"
            # A re-run model enters the geomean with its fresh rounds.
            src = rerun.get(group, {}).get(model) if fails else None
            src_rounds = paired(src) if src is not None else rounds
            src = src if src is not None else entry
            geo_input.append(
                (
                    src_rounds,
                    {r: src["base"][r][1] for r in src_rounds},
                    {r: src["head"][r][1] for r in src_rounds},
                )
            )
            print(
                f"{model:38s} {len(rounds):>3} {med(base_ev):9.4f} {med(head_ev):9.4f} {er:7.3f} "
                f"[{elo:.3f},{ehi:.3f}]  {lr:7.3f} [{llo:.3f},{lhi:.3f}]  {verdict}"
                + ("  " + "; ".join(notes) if notes else "")
            )
        g, glo, ghi = joint_geomean(geo_input, rng, args.resamples)
        geo_fail = g > GEO_LIMIT and glo > GEO_LOWER
        failed |= geo_fail
        print(
            f"geomean eval r over {len(geo_input)} models: {g:.4f} [{glo:.4f},{ghi:.4f}]  "
            + ("FAIL" if geo_fail else "ok")
        )
        print()

    if args.fail_list:
        with open(args.fail_list, "w") as f:
            f.writelines(line + "\n" for line in pending)
    print("GATE " + ("FAIL" if failed else "PASS") + (f" ({len(pending)} model(s) await a re-run)" if pending else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
