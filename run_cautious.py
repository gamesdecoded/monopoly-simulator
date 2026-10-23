"""
Cautious-bot stress test for the Monopoly house-rules follow-up batch.

Identical engine, but bots play cautious: $800 cash reserve before ANY
building, max 3 houses per property (never a 4th), never bid above face
value at auction (already true for baseline; explicit here).

Checkpointed: safe to interrupt and resume with the same arguments.
Writes cautious-results.json. Seed 20261022 (differs from production).
Key question: does the nightmare-rate ranking hold under cautious bots?
"""

import argparse
import json
import os
import time

import numpy as np

import engine as E

HIST_BINS = E.MAX_TURNS + 1


def empty_agg():
    return {
        "games": 0,
        "hist": np.zeros(HIST_BINS, dtype=np.int64),
        "turns_sum": 0,
        "truncated": 0,
        "houses_sum": 0,
        "seat_wins": np.zeros(4, dtype=np.int64),
    }


AGG_KEYS = ("games", "hist", "turns_sum", "truncated", "houses_sum",
            "seat_wins")
INT_KEYS = ("games", "turns_sum", "truncated", "houses_sum")


def ckpt_path(out):
    return out + ".checkpoint.npz"


def save_ckpt(path, args, aggs, done):
    data = {"games": args.games, "seed": args.seed, "batch": args.batch,
            "variants": np.array(args.variants, dtype=np.int64),
            "done_variants": np.array(sorted(done), dtype=np.int64)}
    for v in args.variants:
        for k, val in aggs[v].items():
            data[f"v{v}_{k}"] = val
    np.savez_compressed(path, **data)


def load_ckpt(path, args):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    if int(z["games"]) != args.games or int(z["seed"]) != args.seed:
        return None
    if list(z["variants"]) != list(args.variants):
        return None
    aggs = {}
    for v in args.variants:
        aggs[v] = {k: z[f"v{v}_{k}"] for k in AGG_KEYS}
        for k in INT_KEYS:
            aggs[v][k] = int(aggs[v][k])
    return aggs, set(int(x) for x in z["done_variants"])


def pct_from_hist(hist, q):
    total = hist.sum()
    if total == 0:
        return 0.0
    target = total * q / 100.0
    cum = 0
    for t in range(HIST_BINS):
        cum += hist[t]
        if cum >= target:
            return float(t)
    return float(HIST_BINS - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000000)
    ap.add_argument("--variants", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--out", default="cautious-results.json")
    ap.add_argument("--never-ending-turns", type=int, default=1500)
    args = ap.parse_args()

    ckpt = ckpt_path(args.out)
    loaded = load_ckpt(ckpt, args)
    if loaded:
        aggs, done = loaded
        print("resumed checkpoint: %d games already banked" %
              sum(aggs[v]["games"] for v in args.variants), flush=True)
    else:
        aggs = {v: empty_agg() for v in args.variants}
        done = set()

    t0 = time.time()
    for v in args.variants:
        if v in done:
            print("variant %d (%s): already done, skipping" %
                  (v, E.VARIANT_NAMES[v]), flush=True)
            continue
        a = aggs[v]
        name = E.VARIANT_NAMES[v]
        while a["games"] < args.games:
            b = min(args.batch, args.games - a["games"])
            out = np.zeros((b, E.N_STATS), dtype=np.int64)
            seed0 = args.seed + v * 10_000_000 + a["games"]
            E.play_games(v, b, seed0, out, 1)  # cautious bots
            for i in range(b):
                t = int(out[i, E.ST_TURNS])
                a["hist"][min(t, E.MAX_TURNS)] += 1
            a["games"] += b
            a["turns_sum"] += int(out[:, E.ST_TURNS].sum())
            a["truncated"] += int(out[:, E.ST_TRUNCATED].sum())
            a["houses_sum"] += int(out[:, E.ST_HOUSES].sum())
            for s in range(4):
                a["seat_wins"][s] += int((out[:, E.ST_WINNER] == s).sum())
            save_ckpt(ckpt, args, aggs, done)
            el = time.time() - t0
            print("  [%s] %d/%d games  %.0f games/s  elapsed %.1fh" %
                  (name, a["games"], args.games,
                   a["games"] / max(el, 1e-9), el / 3600), flush=True)
        done.add(v)
        save_ckpt(ckpt, args, aggs, done)

    thr = args.never_ending_turns
    variants_out = {}
    for v in args.variants:
        a = aggs[v]
        n = a["games"]
        variants_out[E.VARIANT_NAMES[v]] = {
            "games": n,
            "turns": {"median": pct_from_hist(a["hist"], 50),
                      "mean": a["turns_sum"] / n},
            "never_ending_per_10000": a["hist"][thr:].sum() / n * 10000,
            "never_ending_pct": a["hist"][thr:].sum() / n * 100,
            "truncated_pct": a["truncated"] / n * 100,
            "houses_per_game": a["houses_sum"] / n,
            "seat_win_pct": [float(x) / n * 100 for x in a["seat_wins"]],
        }

    results = {
        "meta": {
            "game": "monopoly",
            "batch": "cautious-bot-stress-test",
            "games_per_variant": args.games,
            "variants_run": [E.VARIANT_NAMES[v] for v in args.variants],
            "seed": args.seed,
            "bots": "cautious ($800 build reserve, max 3 houses/property, "
                    "never bid above face value)",
            "never_ending_threshold_turns": thr,
            "engine": "engine.py (numba)",
        },
        "variants": variants_out,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
