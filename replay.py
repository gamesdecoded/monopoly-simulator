"""
Nightmare-game replay exporter for the Monopoly cold-open visual.

1. Scans snake_eyes-variant seeds for the most dramatic game
   (most turns; tie-break: most hotels on board).
2. Replays the winning seed turn-by-turn, recording a trajectory:
   [turn, cash_p0..p3, houses_p0..p3, hotels_p0..p3, free_parking_pot].
3. Writes nightmare-replay.json.

The replay reproduces play_game exactly (same init, same RNG call order),
so the trajectory matches what the engine would produce for that seed.
Determinism is verified before export.
"""

import json

import numba
import numpy as np

import engine as E

TRAJ_COLS = 14  # turn + 4 cash + 4 houses + 4 hotels + pot


@numba.njit
def play_game_trajectory(seed, variant, cautious, traj):
    """Mirror of engine.play_game with per-turn snapshots. Returns turns."""
    np.random.seed(seed)
    cash = np.empty(E.N_PLAYERS, np.int64)
    pos = np.empty(E.N_PLAYERS, np.int64)
    in_jail = np.empty(E.N_PLAYERS, np.int64)
    jail_rolls = np.empty(E.N_PLAYERS, np.int64)
    goojf = np.empty(E.N_PLAYERS, np.int64)
    alive = np.empty(E.N_PLAYERS, np.int64)
    for p in range(E.N_PLAYERS):
        cash[p] = E.START_CASH
        pos[p] = 0
        in_jail[p] = 0
        jail_rolls[p] = 0
        goojf[p] = 0
        alive[p] = 1
    owner = np.empty(E.N_SQ, np.int64)
    mort = np.empty(E.N_SQ, np.int64)
    houses = np.empty(E.N_SQ, np.int64)
    for s in range(E.N_SQ):
        owner[s] = -1
        mort[s] = 0
        houses[s] = 0
    bank = np.empty(3, np.int64)
    bank[0] = 32
    bank[1] = 12
    bank[2] = 0
    decks = np.empty((2, 16), np.int64)
    for i in range(16):
        decks[0, i] = i
        decks[1, i] = i
    E.shuffle_deck(decks[0])
    E.shuffle_deck(decks[1])
    dp = np.zeros(2, np.int64)
    stats = np.zeros(E.N_STATS, np.int64)

    turns = 0
    while turns < E.MAX_TURNS and E.alive_count(alive) > 1:
        for p in range(E.N_PLAYERS):
            if E.alive_count(alive) <= 1 or turns >= E.MAX_TURNS:
                break
            if not alive[p]:
                continue
            turns += 1
            E.take_turn(p, pos, cash, owner, mort, houses, alive, goojf,
                        in_jail, jail_rolls, bank, variant, stats, decks, dp,
                        cautious)
            r = turns - 1
            traj[r, 0] = turns
            for q in range(E.N_PLAYERS):
                traj[r, 1 + q] = cash[q]
            for q in range(E.N_PLAYERS):
                h = 0
                ht = 0
                for s in range(E.N_SQ):
                    if owner[s] == q:
                        if houses[s] == 5:
                            ht += 1
                        else:
                            h += houses[s]
                traj[r, 5 + q] = h
                traj[r, 9 + q] = ht
            traj[r, 13] = bank[2]
    return turns


def find_nightmare(base, n_scan, variant):
    """Scan seeds; return (seed, turns, hotels) of the most dramatic game."""
    best_seed, best_turns, best_hotels = base, -1, -1
    batch = 10000
    done = 0
    while done < n_scan:
        b = min(batch, n_scan - done)
        out = np.zeros((b, E.N_STATS), dtype=np.int64)
        E.play_games(variant, b, base + done, out, 0)
        t = out[:, E.ST_TURNS]
        h = out[:, E.ST_HOTELS]
        for i in range(b):
            ti, hi = int(t[i]), int(h[i])
            if ti > best_turns or (ti == best_turns and hi > best_hotels):
                best_turns, best_hotels = ti, hi
                best_seed = base + done + i
        done += b
        print("  scanned %d/%d seeds (best so far: %d turns)" %
              (done, n_scan, best_turns), flush=True)
    return best_seed, best_turns, best_hotels


def main():
    variant = E.SNAKE_EYES
    base = 20261022 + variant * 10_000_000
    print("scanning for the most dramatic snake_eyes game...", flush=True)
    seed, turns, hotels = find_nightmare(base, 50000, variant)
    print("winner: seed=%d turns=%d hotels=%d" % (seed, turns, hotels),
          flush=True)

    # determinism check: trajectory replay must match play_game exactly
    s = np.zeros(E.N_STATS, dtype=np.int64)
    E.play_game(seed, variant, 0, s)
    assert int(s[E.ST_TURNS]) == turns, "replay determinism mismatch!"
    print("determinism verified (turns match play_game)", flush=True)

    traj = np.zeros((E.MAX_TURNS, TRAJ_COLS), dtype=np.int64)
    got = play_game_trajectory(seed, variant, 0, traj)
    assert got == turns
    traj = traj[:turns]

    data = {
        "meta": {
            "game": "monopoly",
            "purpose": "cold-open nightmare replay visual",
            "seed": seed,
            "seed_scheme": "20261022 + 4*10_000_000 + scan_index",
            "variant": E.VARIANT_NAMES[variant],
            "turns": turns,
            "hotels_on_board": hotels,
        },
        "trajectory_columns": (
            ["turn"] +
            [f"cash_p{q}" for q in range(4)] +
            [f"houses_p{q}" for q in range(4)] +
            [f"hotels_p{q}" for q in range(4)] +
            ["free_parking_pot"]
        ),
        "trajectory": traj.tolist(),
    }
    with open("nightmare-replay.json", "w") as f:
        json.dump(data, f)
    print("wrote nightmare-replay.json (%d turns)" % turns, flush=True)


if __name__ == "__main__":
    main()
