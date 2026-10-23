# Monopoly House-Rules Simulator — "Do House Rules Ruin Monopoly?"

Monte Carlo simulation backing the YouTube video **"Do House Rules Ruin
Monopoly?"** Four identical heuristic bots play full Monopoly games under 5
rule sets, so every measured difference comes from the rules, not the
players. Every number in the video comes from this code. Run it yourself.

## The five variants

1. **Control (official rules)** — auctions, even building, 32 houses / 12
   hotels, 10% mortgage interest, jail, full Chance + Community Chest decks.
2. **Free Parking jackpot** — taxes and fines pile in the center; landing on
   Free Parking collects the pot.
3. **No auctions** — declined properties stay with the bank, unowned.
4. **Double GO ($400)** — landing exactly on GO pays $400 total.
5. **Snake Eyes ($500)** — rolling double ones pays $500 from the bank.

## Results (40M games per variant, 200M total)

| Variant | Median game length | Never-ending rate (>1,500 turns) | Seat win % |
|---|---|---|---|
| Control | 147 turns | 1 in 6,652 | 27.3 / 25.8 / 24.2 / 22.6 |
| Free Parking jackpot | 140 turns | 1 in 1,247 | 27.6 / 25.9 / 24.2 / 22.4 |
| No auctions | 155 turns | 1 in 6,641 | 27.3 / 25.7 / 24.2 / 22.8 |
| Double GO ($400) | 142 turns | 1 in 2,090 | 27.6 / 25.9 / 24.1 / 22.3 |
| Snake Eyes ($500) | 136 turns | 1 in 805 | 27.8 / 26.0 / 24.0 / 22.2 |

See `METHODS.md` for the full methodology: bot logic, the turn cap and
never-ending definition, validation, and limitations (no human-like
negotiation — same disclosure as Perfect Play Sim).

## Files

- `engine.py` — the simulation engine (numba-accelerated): full official
  rules, 5 variant flags, 4 identical heuristic bots
- `validate.py` — three validation gates: (a) movement vs the published
  Scientific American 1996 landing table, (b) full-game sanity (300+
  games/variant, zero crashes), (c) throughput benchmark + recommended run size
- `run_full.py` — the production run: N games per variant, checkpointed,
  writes `results.json`
- `results.json` — full production results: 40M games per variant
  (200M total), plus batch-2 files (`mechanism-results.json`,
  `cautious-results.json`, `combo-results.json`)
- `METHODS.md` — rules, bot logic, limitations

## Run it

```bash
python validate.py                 # all three gates must pass
python run_full.py --games 500000  # production run (checkpointed, resumable)
```

The video title uses the REAL total games run (sum over all 5 variants),
reported in `results.json` under `meta.title_uses_total_games`.

## Validation

`validate.py` checks the engine before any production run: a 2M-roll
movement sim reproduces the published 1996 landing-probability table's
"roll for doubles" column (Illinois Ave 2.97% vs 2.993% published; Jail
most-visited; top-4 order Jail → Illinois → GO → B&O exact), 300+
full games per variant run crash-free with sane economies and symmetric
seat win shares, and a ≥60 s throughput benchmark sizes the production run
to finish under ~24 h.
