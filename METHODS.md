# Methods — "Do House Rules Ruin Monopoly?"

Monte Carlo simulation of 4-player Monopoly under 5 rule sets. Four identical
heuristic bots play every game, so measured differences between variants come
from the **rules**, not the players. Engine: `engine.py` (Python + numba),
production runner: `run_full.py`.

## The five variants (locked)

1. **Control — official rules.** Auctions on declined property, even building,
   32 houses / 12 hotels (shortages enforced), mortgages cost 10% interest to
   lift, jail (exit via doubles / $50 / Get-Out-of-Jail-Free), full 16-card
   Chance and Community Chest decks, $200 flat income tax, $100 luxury tax.
2. **Free Parking jackpot.** Every tax payment (Income $200, Luxury $100) and
   every fine / card payment that would go to the bank (jail $50 fine,
   speeding $15, repairs, street repairs, doctor/hospital/school fees)
   instead accumulates in a center pot (starts at $0, no seed money).
   Landing exactly on Free Parking collects the whole pot.
3. **No auctions.** A declined property stays with the bank, unowned. No
   bidding of any kind.
4. **Double GO ($400).** Landing exactly on GO — by dice or by an "Advance to
   GO" card — collects **$400 total**. Passing GO still pays $200.
5. **Snake Eyes ($500).** Rolling double ones collects $500 from the bank, in
   addition to moving 2 and rolling again. (Applies even on the roll that
   sends you to jail on third doubles.)

## Official-rules baseline (applies to all variants)

- Standard 40-square US board. 4 players, $1500 start, $200 salary for
  passing GO. Fixed turn order, seats 0–3.
- Doubles grant another roll; three consecutive doubles → jail (turn ends).
- Jail: sent by third doubles, "Go to Jail" square, or card. Exit by rolling
  doubles (move, no extra roll), paying $50, or playing a Get-Out-of-Jail-Free
  card. After 3 failed rolls the $50 is paid automatically and the player
  moves. Rent collection, building, auctions, mortgages, and trades are all
  allowed while in jail (official rule).
- Declined unowned property is auctioned immediately among all alive players
  (proxy auction: winner pays second-highest maximum + $10). The declining
  player may bid.
- Buildings: even building only; no building on a set with a mortgaged
  property; 32 houses / 12 hotels total — when the bank runs out, nobody
  builds until buildings return. Hotel = 4 houses returned to the bank + one
  house-cost payment.
- Selling buildings: half price, sold from the most-built property first
  (keeps building roughly even). Selling a hotel breaks it back into 4 houses.
- Mortgages: mortgage value = half the face price; lifting = mortgage value
  + 10% interest.
- Rent: streets — rent table by building count, doubled base rent on an
  unbuilt complete set; railroads — $25/$50/$100/$200 by count owned;
  utilities — 4× / 10× the dice roll by count owned. No rent on mortgaged
  property. Utility rent on the "nearest utility" Chance card = 10× a fresh
  dice roll (official rule).
- "Advance to nearest railroad" Chance card: double the normal railroad rent
  if owned, else the bot may buy (or it goes to auction, except variant 3).
- Bankruptcy: a player who cannot raise funds goes bankrupt. Debts to a
  player transfer all assets (properties keep mortgaged flags; cash and
  Get-Out-of-Jail-Free cards transfer). Debts to the bank: buildings return
  to the supply and every property is auctioned immediately, unmortgaged
  (official rule).
- "Go Back 3 Spaces": the new square is resolved normally (one chained card
  draw at most, depth-guarded).
- Get-Out-of-Jail-Free cards are held and used; card trading between bots is
  not implemented (rare, immaterial).

## Bot logic (identical in all variants)

Competent baseline heuristics; deliberately simple so the rules, not
negotiation skill, drive results:

- **Buying:** buy any unowned landable property (street, railroad, utility)
  if cash after purchase stays ≥ $150.
- **Auctions:** bid up to face value for a property completing a color set
  (or a railroad / second utility); up to 70–80% of face otherwise; never
  below the $150 cash reserve.
- **Building:** at end of turn, while cash > build cost + $200 reserve:
  build on the complete unmortgaged set with the best marginal rent-per-dollar,
  keeping buildings even across the set; hotels when all four houses are up
  and a hotel is available.
- **Trades (the simple rule):** once per turn, a bot holding all-but-one of a
  color set (no buildings on it) offers face-value cash for the missing
  property — but only if a single other bot holds it as a lone dead asset
  (exactly 1 of that set) and it is unmortgaged. The offer is accepted
  automatically under those conditions. No haggling, no multi-property
  packages, no rent/immunity deals, no loans.
- **Jail:** use a Get-Out-of-Jail-Free card if held; pay $50 if cash > $300;
  otherwise roll for doubles.
- **Cash management:** when short, sell buildings (even-ish) then mortgage —
  non-set properties first, cheapest first. Unmortgage cheapest-first when
  cash > $400 (keeping a reserve).

## Game end and the "never-ending" definition

- A game ends when one player owns everything (all others bankrupt).
- Turn cap: **3000 individual player-turns** (750 full rounds). A game
  reaching the cap is recorded as **truncated**; its winner is the richest
  surviving player by net worth (cash + properties at face/mortgage value +
  buildings at half price).
- Reported "never-ending rate" = share of games exceeding a reporting
  threshold chosen from the control variant's turn distribution tail
  (default 1500 individual turns = 375 rounds; documented in
  `results.json`). This maps to Brute Force Lab's "1 in 6 Monopoly games
  never end" finding: we report our control rate on the same conceptual
  basis and compare qualitatively.

## Validation (see `validate.py`; all gates pass before any production run)

- **(a) Movement vs the published Scientific American 1996 landing table.**
  A movement-only sim (dice + movement cards + jail, 2M rolls) reproduces the
  published Markov table's "roll for doubles" column: Illinois Ave 2.97% vs
  2.993% published (and within the task's 3.18% ± 0.25pp headline check),
  B&O 2.895% vs 2.893%, Free Parking 2.805% vs 2.825%, Tennessee 2.820% vs
  2.821%; Jail the most-visited square; top-4 order Jail → Illinois → GO →
  B&O exactly as published; "Go To Jail" square itself at 0.00%.
  Note: the published 5.89% jail figure assumes leaving jail on the very
  next turn; our sim (and bots) roll for doubles up to 3 turns, giving ~9.5%
  in-jail + ~2.1% just-visiting = ~11.6% square-10 occupancy. Both are valid
  jail-exit models; ours matches the bots.
- **(b) Full-game sanity:** ≥300 games per variant, zero crashes; invariants
  hold (no negative cash, houses ≤ 32, hotels ≤ 12, every game terminates at
  the cap or by bankruptcy, winner holds > 0 net worth); seat win shares
  within 25% ± 8pp (identical bots ⇒ symmetry check).
- **(c) Throughput:** sustained games/sec measured single-process over ≥60 s
  on the production machine; the recommended run size keeps all 5 variants
  under ~24 h wall-clock. The video title uses the real total games run.

## Limitations (disclosed)

1. **No human-like negotiation.** Bots make only simple set-completing cash
   trades at face value (same limitation Perfect Play Sim discloses). Real
   players haggle, bluff, and form grudges; our results measure how the
   *rules* change the game between competent but artless opponents.
2. **Identical bots.** There is no skill differential to flatten — "winner
   concentration" is measured as per-seat win share (≈25% each by symmetry),
   and we report whether a rule amplifies or dampens the first-player
   advantage. A weak-vs-strong bot experiment is possible future work.
3. **Simplifications:** Get-Out-of-Jail-Free cards are never traded; the
   "Advance to nearest railroad/utility" rent is charged without a buy
   decision from opponents (standard); bankruptcy auctions value properties
   by bot heuristics, not true game-theoretic worth.
4. **Turn cap:** games hitting 3000 turns are truncated, not played out —
   the "never-ending" rate is therefore a lower bound on true non-termination.

## Follow-up batch (batch 2; seed 20261022 — production used 20261021)

All batch-2 runs use the same engine with added instruments; behavior for
variants 0–4 is unchanged (verified: control median/aggregates reproduce the
production run within sampling noise).

- **Stacked combo (variant 5):** Free Parking jackpot + No Auctions + $400 GO
  all at once (the typical family stack). Run separately via
  `run_full.py --variants 5` → `combo-results.json`.
- **Cautious-bot stress test** (`run_cautious.py` → `cautious-results.json`,
  3M games/variant): identical bots except (a) an $800 cash reserve is
  required before ANY building (vs $200), (b) never build a 4th house — max 3
  houses per property (vs hotels allowed), (c) never bid above face value at
  auction (already true for the baseline bot by construction; made explicit).
  Asks whether the nightmare-rate ranking survives more human-like hesitation.
- **Mechanism instruments** (`run_mechanism.py` → `mechanism-results.json`,
  2M games/variant, standard bots): per game the engine now also records
  (i) the turn index of the first completed street monopoly (0 if never),
  (ii) total rent $ paid by all players (→ average rent per turn),
  (iii) rule-printed $ (Snake Eyes $500s; the extra $200 on each exact GO
  landing), (iv) jackpot-recycled $ (pot payouts — tracked separately, not
  counted as printed), and (v) for games ending past 1500 turns, the
  surviving players' cash, houses, and hotels (→ per-survivor averages).
- **Nightmare replay** (`replay.py` → `nightmare-replay.json`): scans 50,000
  Snake Eyes seeds for the most dramatic game (most turns, tie-break most
  hotels) and exports its full per-turn trajectory (turn, cash/houses/hotels
  per player, pot size). The replay re-runs the engine's exact RNG stream
  for that seed; determinism is asserted before export.
