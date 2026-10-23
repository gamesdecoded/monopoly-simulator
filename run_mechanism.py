"""
Mechanism-stats runner for the Monopoly house-rules follow-up batch.

Standard (non-cautious) bots, all 5 base variants. Instruments:
  - turn of first completed monopoly (ST_FIRSTMONO)
  - total rent paid (ST_RENT) -> avg rent per turn
  - rule-injected $ (ST_INJECTED), jackpot-recycled $ (ST_RECYCLED)
  - nightmare-game (>1500 turns) end-state: cash/houses/hotels per survivor

Checkpointed: safe to interrupt and resume with the same arguments.
Writes mechanism-results.json. Seed 20261022 (differs from production).
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
        "fm_hist": np.zeros(HIST_BINS, dtype=np.int64),
        "fm_never": 0,
        "rent_sum": 0,
        "injected_sum": 0,
        "recycled_sum": 0,
        "nm_count": 0,
        "nm_cash": 0,
        "nm_alive": 0,
        "nm_houses": 0,
        "nm_hotels": 0,
    }


AGG_KEYS = ("games", "hist", "turns_sum", "fm_hist", "fm_never", "rent_sum",
            "injected_sum", "recycled_sum", "nm_count", "nm_cash", "nm_alive",
            "nm_houses", "nm_hotels")
INT_KEYS = ("games", "turns_sum", "fm_never", "rent_sum", "injected_sum",
            "recycled_sum", "nm_count", "nm_cash", "nm_alive", "nm_houses",
            "nm_hotels")


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
    ap.add_argument("--games", type=int, default=2000000)
    ap.add_argument("--variants", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--out", default="mechanism-results.json")
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
            E.play_games(v, b, seed0, out, 0)  # standard bots
            for i in range(b):
                t = int(out[i, E.ST_TURNS])
                a["hist"][min(t, E.MAX_TURNS)] += 1
                fm = int(out[i, E.ST_FIRSTMONO])
                if fm > 0:
                    a["fm_hist"][min(fm, E.MAX_TURNS)] += 1
                else:
                    a["fm_never"] += 1
            a["games"] += b
            a["turns_sum"] += int(out[:, E.ST_TURNS].sum())
            a["rent_sum"] += int(out[:, E.ST_RENT].sum())
            a["injected_sum"] += int(out[:, E.ST_INJECTED].sum())
            a["recycled_sum"] += int(out[:, E.ST_RECYCLED].sum())
            a["nm_count"] += int(out[:, E.ST_NM_COUNT].sum())
            a["nm_cash"] += int(out[:, E.ST_NM_CASH].sum())
            a["nm_alive"] += int(out[:, E.ST_NM_ALIVE].sum())
            a["nm_houses"] += int(out[:, E.ST_NM_HOUSES].sum())
            a["nm_hotels"] += int(out[:, E.ST_NM_HOTELS].sum())
            save_ckpt(ckpt, args, aggs, done)
            el = time.time() - t0
            print("  [%s] %d/%d games  %.0f games/s  elapsed %.1fh" %
                  (name, a["games"], args.games,
                   a["games"] / max(el, 1e-9), el / 3600), flush=True)
        done.add(v)
        save_ckpt(ckpt, args, aggs, done)

    variants_out = {}
    for v in args.variants:
        a = aggs[v]
        n = a["games"]
        fm_total = a["fm_hist"].sum()
        variants_out[E.VARIANT_NAMES[v]] = {
            "games": n,
            "turns": {"median": pct_from_hist(a["hist"], 50),
                      "mean": a["turns_sum"] / n},
            "first_monopoly": {
                "median_turn": pct_from_hist(a["fm_hist"], 50),
                "mean_turn": (a["fm_hist"] * np.arange(HIST_BINS)).sum() / max(fm_total, 1),
                "never_share_pct": a["fm_never"] / n * 100,
            },
            "rent_per_turn": a["rent_sum"] / max(a["turns_sum"], 1),
            "injected_per_game": a["injected_sum"] / n,
            "recycled_per_game": a["recycled_sum"] / n,
            "nightmare_games": {
                "count": int(a["nm_count"]),
                "avg_cash_per_survivor": a["nm_cash"] / max(a["nm_alive"], 1),
                "avg_houses_per_survivor": a["nm_houses"] / max(a["nm_alive"], 1),
                "avg_hotels_per_survivor": a["nm_hotels"] / max(a["nm_alive"], 1),
            },
        }

    results = {
        "meta": {
            "game": "monopoly",
            "batch": "mechanism-stats",
            "games_per_variant": args.games,
            "variants_run": [E.VARIANT_NAMES[v] for v in args.variants],
            "seed": args.seed,
            "bots": "standard (non-cautious)",
            "engine": "engine.py (numba)",
        },
        "variants": variants_out,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
