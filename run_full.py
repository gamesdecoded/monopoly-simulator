"""
Production runner for the Monopoly house-rules simulation.

Runs N games per variant (default: all 5 variants), aggregates metrics, and
writes results.json. Checkpointed: safe to interrupt and resume with the same
arguments.

Usage:
    python run_full.py --games 500000 --out results.json
    python run_full.py --games 500000 --variants 0 1 --seed 7 --batch 20000

The title of the video uses the REAL total games run (sum over variants).
"""

import argparse
import json
import os
import time

import numpy as np

import engine as E

HIST_BINS = E.MAX_TURNS + 1  # 1-turn bins: exact percentiles from the histogram


def empty_agg():
    return {
        "games": 0,
        "hist": np.zeros(HIST_BINS, dtype=np.int64),
        "turns_sum": 0,
        "truncated": 0,
        "bankrupt_sum": 0,
        "houses_sum": 0,
        "hotels_sum": 0,
        "trades_sum": 0,
        "auctions_sum": 0,
        "pot_max": 0,
        "seat_wins": np.zeros(4, dtype=np.int64),
    }


def ckpt_path(out):
    return out + ".checkpoint.npz"


def save_ckpt(path, args, aggs, done_variants):
    data = {"games": args.games, "seed": args.seed, "batch": args.batch,
            "variants": np.array(args.variants, dtype=np.int64),
            "done_variants": np.array(sorted(done_variants), dtype=np.int64)}
    for v in args.variants:
        a = aggs[v]
        for k, val in a.items():
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
        aggs[v] = {k: z[f"v{v}_{k}"] for k in
                   ("games", "hist", "turns_sum", "truncated", "bankrupt_sum",
                    "houses_sum", "hotels_sum", "trades_sum", "auctions_sum",
                    "pot_max", "seat_wins")}
        # npz loads 0-d arrays for scalars; normalize
        for k in ("games", "turns_sum", "truncated", "bankrupt_sum",
                  "houses_sum", "hotels_sum", "trades_sum", "auctions_sum",
                  "pot_max"):
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
    ap.add_argument("--games", type=int, default=500000,
                    help="games per variant")
    ap.add_argument("--variants", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                    help="variant ids to run")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--batch", type=int, default=20000,
                    help="games per checkpoint batch")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--never-ending-turns", type=int, default=1500,
                    help="reporting threshold for the never-ending rate")
    args = ap.parse_args()

    ckpt = ckpt_path(args.out)
    loaded = load_ckpt(ckpt, args)
    if loaded:
        aggs, done = loaded
        print("resumed checkpoint: %d games already banked" %
              sum(aggs[v]["games"] for v in args.variants))
    else:
        aggs = {v: empty_agg() for v in args.variants}
        done = set()

    t0 = time.time()
    for v in args.variants:
        if v in done:
            print("variant %d (%s): already done, skipping" %
                  (v, E.VARIANT_NAMES[v]))
            continue
        a = aggs[v]
        name = E.VARIANT_NAMES[v]
        while a["games"] < args.games:
            b = min(args.batch, args.games - a["games"])
            out = np.zeros((b, E.N_STATS), dtype=np.int64)
            seed0 = args.seed + v * 10_000_000 + a["games"]
            E.play_games(v, b, seed0, out, 0)
            for i in range(b):
                t = int(out[i, E.ST_TURNS])
                a["hist"][min(t, E.MAX_TURNS)] += 1
            a["games"] += b
            a["turns_sum"] += int(out[:, E.ST_TURNS].sum())
            a["truncated"] += int(out[:, E.ST_TRUNCATED].sum())
            a["bankrupt_sum"] += int(out[:, E.ST_BANKRUPT].sum())
            a["houses_sum"] += int(out[:, E.ST_HOUSES].sum())
            a["hotels_sum"] += int(out[:, E.ST_HOTELS].sum())
            a["trades_sum"] += int(out[:, E.ST_TRADES].sum())
            a["auctions_sum"] += int(out[:, E.ST_AUCTIONS].sum())
            pm = int(out[:, E.ST_POTMAX].max())
            if pm > a["pot_max"]:
                a["pot_max"] = pm
            for s in range(4):
                a["seat_wins"][s] += int((out[:, E.ST_WINNER] == s).sum())
            save_ckpt(ckpt, args, aggs, done)
            el = time.time() - t0
            gps = a["games"] / max(el, 1e-9)
            print("  [%s] %d/%d games  %.0f games/s  elapsed %.1fh" %
                  (name, a["games"], args.games, gps, el / 3600), flush=True)
        done.add(v)
        save_ckpt(ckpt, args, aggs, done)

    # ---- aggregate to results.json ----
    total_games = 0
    variants_out = {}
    for v in args.variants:
        a = aggs[v]
        n = a["games"]
        total_games += n
        hist = a["hist"]
        thr = args.never_ending_turns
        variants_out[E.VARIANT_NAMES[v]] = {
            "games": n,
            "turns": {
                "median": pct_from_hist(hist, 50),
                "mean": a["turns_sum"] / n,
                "p90": pct_from_hist(hist, 90),
                "p99": pct_from_hist(hist, 99),
                "max": int(np.nonzero(hist)[0].max()),
            },
            "rounds": {  # 1 round = 4 individual turns
                "median": pct_from_hist(hist, 50) / 4,
                "mean": a["turns_sum"] / n / 4,
            },
            "truncated_at_cap": int(a["truncated"]),
            "truncated_pct": a["truncated"] / n * 100,
            "never_ending_threshold_turns": thr,
            "never_ending_pct": hist[thr:].sum() / n * 100,
            "seat_win_pct": [float(x) / n * 100 for x in a["seat_wins"]],
            "bankruptcies_per_game": a["bankrupt_sum"] / n,
            "houses_per_game": a["houses_sum"] / n,
            "hotels_per_game": a["hotels_sum"] / n,
            "trades_per_game": a["trades_sum"] / n,
            "auctions_per_game": a["auctions_sum"] / n,
            "max_free_parking_pot": int(a["pot_max"]),
        }

    results = {
        "meta": {
            "game": "monopoly",
            "title_uses_total_games": total_games,
            "games_per_variant": args.games,
            "variants_run": [E.VARIANT_NAMES[v] for v in args.variants],
            "seed": args.seed,
            "never_ending_threshold_turns": args.never_ending_turns,
            "engine": "engine.py (numba)",
            "methods": "METHODS.md",
        },
        "variants": variants_out,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote %s (%d total games)" % (args.out, total_games))


if __name__ == "__main__":
    main()
