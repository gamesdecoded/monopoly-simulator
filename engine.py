"""
Monopoly house-rules Monte Carlo engine — "Do House Rules Ruin Monopoly?"

Simulates 4-player Monopoly under 5 rule variants with identical heuristic bots,
so measured differences come from the rules, not the players.

Variants (config flag on every game):
  0 CONTROL              - official rules baseline
  1 FREE_PARKING_JACKPOT - taxes/fines accumulate in a center pot; landing
                           exactly on Free Parking collects it
  2 NO_AUCTIONS          - declined properties stay with the bank, unowned
  3 DOUBLE_GO            - landing exactly on GO collects $400 (passing: $200)
  4 SNAKE_EYES           - rolling double ones pays $500 from the bank
  5 STACKED_COMBO        - family combo: jackpot + no auctions + $400 GO at once

Official-rules baseline implemented here: auctions on declined property,
even building, 32 houses / 12 hotels (shortages enforced), mortgages with
10% interest to lift, jail (doubles / $50 / Get-Out-of-Jail-Free), full
16-card Chance and Community Chest decks, $200 flat income tax, $100 luxury
tax, bankruptcy transfers to creditor (bank auctions to-bankrupt properties).

Board data: standard US board. Square indices 0..39 starting at GO.
"""

import numba
import numpy as np

# ---------------------------------------------------------------- variants
CONTROL = 0
FREE_PARKING_JACKPOT = 1
NO_AUCTIONS = 2
DOUBLE_GO = 3
SNAKE_EYES = 4
STACKED_COMBO = 5      # family combo: jackpot + no auctions + $400 GO together

VARIANT_NAMES = {
    CONTROL: "control (official rules)",
    FREE_PARKING_JACKPOT: "free_parking_jackpot",
    NO_AUCTIONS: "no_auctions",
    DOUBLE_GO: "double_go ($400 landing on GO)",
    SNAKE_EYES: "snake_eyes ($500 on double ones)",
    STACKED_COMBO: "stacked (jackpot + no auctions + $400 GO)",
}


@numba.njit
def is_jackpot(v):
    return v == FREE_PARKING_JACKPOT or v == STACKED_COMBO


@numba.njit
def is_noauctions(v):
    return v == NO_AUCTIONS or v == STACKED_COMBO


@numba.njit
def is_doublego(v):
    return v == DOUBLE_GO or v == STACKED_COMBO


@numba.njit
def is_snakeeyes(v):
    return v == SNAKE_EYES

# ------------------------------------------------------------ square types
SQ_GO = 0
SQ_STREET = 1
SQ_CHEST = 2
SQ_TAX = 3
SQ_RAIL = 4
SQ_CHANCE = 5
SQ_JAIL = 6        # just visiting
SQ_UTIL = 7
SQ_PARK = 8
SQ_GOTOJAIL = 9

# ------------------------------------------------------------------ board
# index: 0 GO, 10 Jail(visiting), 20 Free Parking, 30 Go To Jail
SQ_TYPE = np.array([
    SQ_GO,     SQ_STREET, SQ_CHEST,  SQ_STREET, SQ_TAX,    SQ_RAIL,   SQ_STREET, SQ_CHANCE,
    SQ_STREET, SQ_STREET, SQ_JAIL,   SQ_STREET, SQ_UTIL,   SQ_STREET, SQ_STREET, SQ_RAIL,
    SQ_STREET, SQ_CHEST,  SQ_STREET, SQ_STREET, SQ_PARK,   SQ_STREET, SQ_CHANCE, SQ_STREET,
    SQ_STREET, SQ_RAIL,   SQ_STREET, SQ_STREET, SQ_UTIL,   SQ_STREET, SQ_GOTOJAIL,
    SQ_STREET, SQ_STREET, SQ_CHEST,  SQ_STREET, SQ_RAIL,   SQ_CHANCE, SQ_STREET, SQ_TAX,
    SQ_STREET,
], dtype=np.int64)

PRICE = np.zeros(40, dtype=np.int64)
for _sq, _p in ((1, 60), (3, 60), (5, 200), (6, 100), (8, 100), (9, 120),
                (11, 140), (12, 150), (13, 140), (14, 160), (15, 200),
                (16, 180), (18, 180), (19, 200), (21, 220), (23, 220),
                (24, 240), (25, 200), (26, 260), (27, 260), (28, 150),
                (29, 280), (31, 300), (32, 300), (34, 320), (35, 200),
                (37, 350), (39, 400)):
    PRICE[_sq] = _p

HOUSE_COST = np.zeros(40, dtype=np.int64)
for _sq, _c in ((1, 50), (3, 50), (6, 50), (8, 50), (9, 50),
                (11, 100), (13, 100), (14, 100), (16, 100), (18, 100), (19, 100),
                (21, 150), (23, 150), (24, 150), (26, 150), (27, 150), (29, 150),
                (31, 200), (32, 200), (34, 200), (37, 200), (39, 200)):
    HOUSE_COST[_sq] = _c

# color groups 0..7 (streets), 8 railroads, 9 utilities, -1 otherwise
GROUP = np.full(40, -1, dtype=np.int64)
for _sq, _g in ((1, 0), (3, 0),
                (6, 1), (8, 1), (9, 1),
                (11, 2), (13, 2), (14, 2),
                (16, 3), (18, 3), (19, 3),
                (21, 4), (23, 4), (24, 4),
                (26, 5), (27, 5), (29, 5),
                (31, 6), (32, 6), (34, 6),
                (37, 7), (39, 7),
                (5, 8), (15, 8), (25, 8), (35, 8),
                (12, 9), (28, 9)):
    GROUP[_sq] = _g
GROUP_SIZE = np.array([2, 3, 3, 3, 3, 3, 3, 2, 4, 2], dtype=np.int64)

# rent[sq][0..4] = houses 0..4, rent[sq][5] = hotel
RENT = np.zeros((40, 6), dtype=np.int64)
_RENTS = {
    1: (2, 10, 30, 90, 160, 250), 3: (4, 20, 60, 180, 320, 450),
    6: (6, 30, 90, 270, 400, 550), 8: (6, 30, 90, 270, 400, 550),
    9: (8, 40, 100, 300, 450, 600),
    11: (10, 50, 150, 450, 625, 750), 13: (10, 50, 150, 450, 625, 750),
    14: (12, 60, 180, 500, 700, 900),
    16: (14, 70, 200, 550, 750, 950), 18: (14, 70, 200, 550, 750, 950),
    19: (16, 80, 220, 600, 800, 1000),
    21: (18, 90, 250, 700, 875, 1050), 23: (18, 90, 250, 700, 875, 1050),
    24: (20, 100, 300, 750, 925, 1100),
    26: (22, 110, 330, 800, 975, 1150), 27: (22, 110, 330, 800, 975, 1150),
    29: (24, 120, 360, 850, 1025, 1200),
    31: (26, 130, 390, 900, 1100, 1275), 32: (26, 130, 390, 900, 1100, 1275),
    34: (28, 150, 450, 1000, 1200, 1400),
    37: (35, 175, 500, 1100, 1300, 1500), 39: (50, 200, 600, 1400, 1700, 2000),
}
for _sq, _r in _RENTS.items():
    RENT[_sq] = np.array(_r, dtype=np.int64)
del _sq, _p, _c, _g, _r

TAX = np.zeros(40, dtype=np.int64)
TAX[4] = 200    # Income Tax (flat, modern official rule)
TAX[38] = 100   # Luxury Tax

N_SQ = 40
N_PLAYERS = 4
START_CASH = 1500
GO_SALARY = 200
MAX_TURNS = 3000          # individual player-turns; hitting it = truncated game
JAIL_SQ = 10

# ------------------------------------------------------------ chance cards
CH_BOARDWALK = 0
CH_GO = 1
CH_ILLINOIS = 2
CH_STCHARLES = 3
CH_RAIL_A = 4
CH_RAIL_B = 5
CH_UTIL = 6
CH_DIVIDEND = 7
CH_GOOJF = 8
CH_BACK3 = 9
CH_JAIL = 10
CH_REPAIRS = 11
CH_SPEEDING = 12
CH_READING = 13
CH_CHAIRMAN = 14
CH_LOAN = 15

# ----------------------------------------------------- community chest cards
CC_GO = 0
CC_BANKERROR = 1
CC_DOCTOR = 2
CC_GOOJF = 3
CC_JAIL = 4
CC_OPERA = 5
CC_HOLIDAY = 6
CC_TAXREFUND = 7
CC_BIRTHDAY = 8
CC_LIFEINS = 9
CC_HOSPITAL = 10
CC_SCHOOL = 11
CC_CONSULT = 12
CC_STREETREPAIR = 13
CC_BEAUTY = 14
CC_INHERIT = 15

# stats indices (per-game output row)
ST_TURNS = 0
ST_WINNER = 1
ST_TRUNCATED = 2
ST_BANKRUPT = 3
ST_HOUSES = 4
ST_TRADES = 5
ST_AUCTIONS = 6
ST_POTMAX = 7
ST_HOTELS = 8
# --- batch-2 instruments (mechanism stats) ---
ST_FIRSTMONO = 9   # turn index of first completed street monopoly (0 = never)
ST_RENT = 10       # total rent $ paid by all players this game
ST_INJECTED = 11   # rule-printed $ this game (snake eyes $500s, GO $200 extras)
ST_RECYCLED = 12   # jackpot pot payouts $ this game (recycled, not printed)
ST_NM_COUNT = 13   # 1 if turns > 1500 (nightmare game), else 0
ST_NM_CASH = 14    # sum of cash of alive players at end (nightmare games only)
ST_NM_ALIVE = 15   # alive players at end (nightmare games only)
ST_NM_HOUSES = 16  # houses on alive players' properties (nightmare games only)
ST_NM_HOTELS = 17  # hotels on alive players' properties (nightmare games only)
N_STATS = 18


# ============================================================ njit helpers
@numba.njit
def shuffle_deck(deck):
    n = deck.shape[0]
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        t = deck[i]
        deck[i] = deck[j]
        deck[j] = t


@numba.njit
def count_owned(p, gid, owner):
    c = 0
    for s in range(N_SQ):
        if GROUP[s] == gid and owner[s] == p:
            c += 1
    return c


@numba.njit
def owns_all(p, gid, owner):
    return count_owned(p, gid, owner) == GROUP_SIZE[gid]


@numba.njit
def group_has_houses(gid, houses, owner):
    for s in range(N_SQ):
        if GROUP[s] == gid and owner[s] >= 0 and houses[s] > 0:
            return True
    return False


@numba.njit
def group_mortgaged(gid, mort, owner):
    for s in range(N_SQ):
        if GROUP[s] == gid and owner[s] >= 0 and mort[s]:
            return True
    return False


@numba.njit
def street_rent(sq, p, houses, owner):
    h = houses[sq]
    if h == 0 and owns_all(p, GROUP[sq], owner):
        return RENT[sq][0] * 2
    return RENT[sq][h]


@numba.njit
def net_worth(p, cash, owner, mort, houses):
    w = cash[p]
    for s in range(N_SQ):
        if owner[s] == p:
            if mort[s]:
                w += PRICE[s] // 2
            else:
                w += PRICE[s]
            w += houses[s] * (HOUSE_COST[s] // 2)
    return w


@numba.njit
def alive_count(alive):
    c = 0
    for p in range(N_PLAYERS):
        if alive[p]:
            c += 1
    return c


# ------------------------------------------------------- money / bankruptcy
@numba.njit
def sell_buildings_for_cash(p, need, cash, owner, mort, houses, bank):
    """Sell houses (even-ish: from the most-built property) until cash>=need.
    Returns True when cash covers need."""
    while cash[p] < need:
        best = -1
        best_h = 0
        for s in range(N_SQ):
            if owner[s] == p and not mort[s] and houses[s] > best_h:
                best_h = houses[s]
                best = s
        if best < 0:
            break
        houses[best] -= 1
        cash[p] += HOUSE_COST[best] // 2
        if best_h == 5:  # sold a hotel: it breaks back into 4 houses
            bank[1] += 1          # hotel returns to supply
            bank[0] -= 4          # 4 houses leave the supply
            houses[best] = 4
        else:
            bank[0] += 1
    return cash[p] >= need


@numba.njit
def mortgage_for_cash(p, need, cash, owner, mort, houses):
    """Mortgage properties (least strategic first) until cash>=need."""
    # pass 1: non-set properties (railroads/utilities/single streets), cheapest first
    for _pass in range(2):
        while cash[p] < need:
            best = -1
            best_price = 10 ** 12
            for s in range(N_SQ):
                if owner[s] == p and not mort[s] and houses[s] == 0 and PRICE[s] > 0:
                    g = GROUP[s]
                    is_set_piece = g < 8 and owns_all(p, g, owner)
                    if (_pass == 0 and is_set_piece) or (_pass == 1 and not is_set_piece):
                        continue
                    if PRICE[s] < best_price:
                        best_price = PRICE[s]
                        best = s
            if best < 0:
                break
            mort[best] = 1
            cash[p] += PRICE[best] // 2
    return cash[p] >= need


@numba.njit
def do_bankrupt(p, creditor, cash, owner, mort, houses, alive, goojf, bank):
    """Resolve bankruptcy of p. creditor: player idx, or -1 = the bank."""
    alive[p] = 0
    # buildings return to the bank (no cash: bankrupt)
    for s in range(N_SQ):
        if owner[s] == p:
            if houses[s] == 5:
                bank[1] += 1
                houses[s] = 0
            elif houses[s] > 0:
                bank[0] += houses[s]
                houses[s] = 0
    if creditor >= 0:
        cash[creditor] += cash[p]
        goojf[creditor] += goojf[p]
        for s in range(N_SQ):
            if owner[s] == p:
                owner[s] = creditor  # mortgaged flags transfer as-is
    else:
        # bank: auction every property immediately (unmortgaged, official rule)
        for s in range(N_SQ):
            if owner[s] == p:
                owner[s] = -1
                mort[s] = 0
    cash[p] = 0
    goojf[p] = 0


@numba.njit
def spend(p, amount, creditor, to_pot, cash, owner, mort, houses, alive, goojf,
          bank, stats):
    """p pays `amount`. creditor: player idx, -1 = bank, -2 = center pot.
    to_pot=1 routes bank-bound fines/taxes to the pot (variant 1).
    Returns True if p is still alive afterwards."""
    if amount <= 0:
        return True
    if cash[p] >= amount:
        cash[p] -= amount
    else:
        need = amount
        if not sell_buildings_for_cash(p, need, cash, owner, mort, houses, bank):
            if not mortgage_for_cash(p, need, cash, owner, mort, houses):
                do_bankrupt(p, creditor if creditor >= 0 else -1, cash, owner,
                            mort, houses, alive, goojf, bank)
                stats[ST_BANKRUPT] += 1
                return False
        cash[p] -= amount
    if creditor >= 0:
        cash[creditor] += amount
    elif to_pot:
        bank[2] += amount
    return True


# ==================================================== bot: auctions/trades
@numba.njit
def bot_max_bid(p, prop, cash, owner, cautious):
    g = GROUP[prop]
    n = count_owned(p, g, owner)
    if g == 8:
        base = PRICE[prop]
    elif g == 9:
        base = PRICE[prop] if n > 0 else (PRICE[prop] * 7) // 10
    elif n == GROUP_SIZE[g] - 1:
        base = PRICE[prop]          # completes my set
    elif n > 0:
        base = (PRICE[prop] * 8) // 10
    else:
        base = (PRICE[prop] * 7) // 10
    if cautious and base > PRICE[prop]:
        base = PRICE[prop]  # never bid above face value (baseline already
                            # satisfies this by construction; explicit here)
    cap = cash[p] - 150
    if cap < 10:
        return 0
    return base if base < cap else cap


@numba.njit
def run_auction(prop, cash, owner, mort, houses, alive, bank, stats, cautious):
    """Proxy (second-price +$10) auction among all alive players."""
    m1 = 0
    m2 = 0
    w = -1
    for p in range(N_PLAYERS):
        if not alive[p]:
            continue
        b = bot_max_bid(p, prop, cash, owner, cautious)
        if b > m1:
            m2 = m1
            m1 = b
            w = p
        elif b > m2:
            m2 = b
    if w < 0 or m1 < 10:
        return  # no sale: stays with the bank
    price = m2 + 10
    if price > m1:
        price = m1
    if price < 10:
        price = 10
    cash[w] -= price
    owner[prop] = w
    stats[ST_AUCTIONS] += 1


@numba.njit
def try_trade(p, cash, owner, mort, houses, alive, stats):
    """One set-completing cash offer per call (the disclosed simple-trade rule)."""
    for g in range(8):
        if group_has_houses(g, houses, owner):
            continue
        mine = 0
        missing = -1
        bad = False
        for s in range(N_SQ):
            if GROUP[s] != g:
                continue
            if owner[s] == p:
                if mort[s]:
                    bad = True
                    break
                mine += 1
            elif owner[s] >= 0:
                if mort[s]:
                    continue
                if missing == -1:
                    missing = s
                else:
                    bad = True  # split across 2+ owners: skip
                    break
        if bad or mine + 1 != GROUP_SIZE[g] or missing < 0:
            continue
        q = owner[missing]
        if not alive[q]:
            continue
        if count_owned(q, g, owner) != 1:
            continue  # q only sells a dead single asset at face value
        offer = PRICE[missing]
        if cash[p] < offer + 150:
            continue
        cash[p] -= offer
        cash[q] += offer
        owner[missing] = p
        stats[ST_TRADES] += 1
        return


@numba.njit
def build_phase(p, cash, owner, mort, houses, bank, stats, cautious):
    """Build while affordable: best marginal rent-per-dollar, even building.
    Cautious: $800 cash reserve before any building, max 3 houses/property."""
    reserve = 800 if cautious else 200
    max_h = 3 if cautious else 5
    for _ in range(200):  # safety bound; normally exits via break
        best_s = -1
        best_gain = 0
        best_cost = 1
        best_hotel = False
        for g in range(8):
            if not owns_all(p, g, owner):
                continue
            if group_mortgaged(g, mort, owner):
                continue
            mh = 99
            for s in range(N_SQ):
                if GROUP[s] == g and owner[s] == p and houses[s] < mh:
                    mh = houses[s]
            if mh >= max_h:
                continue
            for s in range(N_SQ):
                if GROUP[s] != g or owner[s] != p or houses[s] != mh:
                    continue
                cost = HOUSE_COST[s]
                if mh < 4:
                    if bank[0] <= 0:
                        continue
                    gain = RENT[s][mh + 1] - RENT[s][mh]
                    hotel = False
                else:
                    if bank[1] <= 0:
                        continue
                    gain = RENT[s][5] - RENT[s][4]
                    hotel = True
                if cash[p] < cost + reserve:
                    continue
                if gain * best_cost > best_gain * cost:
                    best_gain = gain
                    best_cost = cost
                    best_s = s
                    best_hotel = hotel
        if best_s < 0:
            break
        houses[best_s] += 1
        cash[p] -= best_cost
        if best_hotel:
            bank[1] -= 1
            bank[0] += 4
            stats[ST_HOTELS] += 1
        else:
            bank[0] -= 1
        stats[ST_HOUSES] += 1


@numba.njit
def unmortgage_phase(p, cash, owner, mort):
    while cash[p] > 400:
        best = -1
        best_price = 10 ** 12
        for s in range(N_SQ):
            if owner[s] == p and mort[s] and PRICE[s] < best_price:
                best_price = PRICE[s]
                best = s
        if best < 0:
            break
        cost = (PRICE[best] * 11) // 20  # mortgage value + 10% interest
        if cash[p] < cost + 250:
            break
        cash[p] -= cost
        mort[best] = 0


# ============================================================ cards & moves
@numba.njit
def nearest_rail(pos):
    if pos < 5:
        return 5
    if pos < 15:
        return 15
    if pos < 25:
        return 25
    if pos < 35:
        return 35
    return 5


@numba.njit
def nearest_util(pos):
    if pos < 12:
        return 12
    if pos < 28:
        return 28
    return 12


@numba.njit
def send_to_jail(p, pos, in_jail, jail_rolls):
    pos[p] = JAIL_SQ
    in_jail[p] = 1
    jail_rolls[p] = 0


@numba.njit
def card_move_to(p, target, pos, cash, variant, stats):
    """Card-directed move; collects $200 when passing GO ($400 in variant 3
    when the card sends you to GO itself)."""
    if target < pos[p]:
        cash[p] += GO_SALARY
    pos[p] = target
    if target == 0 and is_doublego(variant):
        cash[p] += GO_SALARY  # land-on-GO bonus: $400 total
        stats[ST_INJECTED] += GO_SALARY  # the extra $200 over the normal salary


@numba.njit
def apply_chance(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                 in_jail, jail_rolls, bank, variant, stats, decks, dp, depth,
                 cautious):
    cdeck = decks[0]
    c = cdeck[dp[0]]
    dp[0] += 1
    if dp[0] >= 16:
        shuffle_deck(cdeck)
        dp[0] = 0
    to_pot = is_jackpot(variant)
    if c == CH_BOARDWALK:
        card_move_to(p, 39, pos, cash, variant, stats)
        land_on(p, 39, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, depth + 1, cautious)
    elif c == CH_GO:
        card_move_to(p, 0, pos, cash, variant, stats)
    elif c == CH_ILLINOIS:
        card_move_to(p, 24, pos, cash, variant, stats)
        land_on(p, 24, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, depth + 1, cautious)
    elif c == CH_STCHARLES:
        card_move_to(p, 11, pos, cash, variant, stats)
        land_on(p, 11, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, depth + 1, cautious)
    elif c == CH_RAIL_A or c == CH_RAIL_B:
        t = nearest_rail(pos[p])
        card_move_to(p, t, pos, cash, variant, stats)
        o = owner[t]
        if o == -1:
            offer_unowned(p, t, cash, owner, mort, houses, alive, bank,
                          variant, stats, cautious)
        elif o != p and not mort[t]:
            n = count_owned(o, 8, owner)
            if n == 1:
                base = 25
            elif n == 2:
                base = 50
            elif n == 3:
                base = 100
            else:
                base = 200
            spend(p, base * 2, o, 0, cash, owner, mort, houses, alive, goojf,
                  bank, stats)
    elif c == CH_UTIL:
        t = nearest_util(pos[p])
        card_move_to(p, t, pos, cash, variant, stats)
        o = owner[t]
        if o == -1:
            offer_unowned(p, t, cash, owner, mort, houses, alive, bank,
                          variant, stats, cautious)
        elif o != p and not mort[t]:
            r1 = np.random.randint(1, 7)
            r2 = np.random.randint(1, 7)
            mult = 10 if count_owned(o, 9, owner) == 2 else 4
            spend(p, (r1 + r2) * mult, o, 0, cash, owner, mort, houses, alive,
                  goojf, bank, stats)
    elif c == CH_DIVIDEND:
        cash[p] += 50
    elif c == CH_GOOJF:
        goojf[p] += 1
    elif c == CH_BACK3:
        pos[p] = (pos[p] - 3 + N_SQ) % N_SQ
        land_on(p, pos[p], d1, d2, pos, cash, owner, mort, houses, alive,
                goojf, in_jail, jail_rolls, bank, variant, stats, decks, dp,
                depth + 1, cautious)
    elif c == CH_JAIL:
        send_to_jail(p, pos, in_jail, jail_rolls)
    elif c == CH_REPAIRS:
        nh = 0
        nht = 0
        for s in range(N_SQ):
            if owner[s] == p:
                if houses[s] == 5:
                    nht += 1
                else:
                    nh += houses[s]
        spend(p, nh * 25 + nht * 100, -1, to_pot, cash, owner, mort, houses,
              alive, goojf, bank, stats)
    elif c == CH_SPEEDING:
        spend(p, 15, -1, to_pot, cash, owner, mort, houses, alive, goojf,
              bank, stats)
    elif c == CH_READING:
        card_move_to(p, 5, pos, cash, variant, stats)
        land_on(p, 5, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, depth + 1, cautious)
    elif c == CH_CHAIRMAN:
        for q in range(N_PLAYERS):
            if q != p and alive[q]:
                if not spend(p, 50, q, 0, cash, owner, mort, houses, alive,
                             goojf, bank, stats):
                    break
    elif c == CH_LOAN:
        cash[p] += 150


@numba.njit
def apply_chest(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, depth):
    hdeck = decks[1]
    c = hdeck[dp[1]]
    dp[0] += 1
    if dp[0] >= 16:
        shuffle_deck(hdeck)
        dp[0] = 0
    to_pot = is_jackpot(variant)
    if c == CC_GO:
        card_move_to(p, 0, pos, cash, variant, stats)
    elif c == CC_BANKERROR:
        cash[p] += 200
    elif c == CC_DOCTOR:
        spend(p, 50, -1, to_pot, cash, owner, mort, houses, alive, goojf,
              bank, stats)
    elif c == CC_GOOJF:
        goojf[p] += 1
    elif c == CC_JAIL:
        send_to_jail(p, pos, in_jail, jail_rolls)
    elif c == CC_OPERA:
        for q in range(N_PLAYERS):
            if q != p and alive[q]:
                spend(q, 50, p, 0, cash, owner, mort, houses, alive,
                      goojf, bank, stats)
    elif c == CC_HOLIDAY:
        cash[p] += 100
    elif c == CC_TAXREFUND:
        cash[p] += 20
    elif c == CC_BIRTHDAY:
        for q in range(N_PLAYERS):
            if q != p and alive[q]:
                spend(q, 10, p, 0, cash, owner, mort, houses, alive,
                      goojf, bank, stats)
    elif c == CC_LIFEINS:
        cash[p] += 100
    elif c == CC_HOSPITAL:
        spend(p, 100, -1, to_pot, cash, owner, mort, houses, alive, goojf,
              bank, stats)
    elif c == CC_SCHOOL:
        spend(p, 150, -1, to_pot, cash, owner, mort, houses, alive, goojf,
              bank, stats)
    elif c == CC_CONSULT:
        cash[p] += 25
    elif c == CC_STREETREPAIR:
        nh = 0
        nht = 0
        for s in range(N_SQ):
            if owner[s] == p:
                if houses[s] == 5:
                    nht += 1
                else:
                    nh += houses[s]
        spend(p, nh * 40 + nht * 115, -1, to_pot, cash, owner, mort, houses,
              alive, goojf, bank, stats)
    elif c == CC_BEAUTY:
        cash[p] += 10
    elif c == CC_INHERIT:
        cash[p] += 100


# ==================================================== landing / turn / game
@numba.njit
def offer_unowned(p, prop, cash, owner, mort, houses, alive, bank, variant,
                  stats, cautious):
    if cash[p] - PRICE[prop] >= 150:
        cash[p] -= PRICE[prop]
        owner[prop] = p
    elif not is_noauctions(variant):
        run_auction(prop, cash, owner, mort, houses, alive, bank, stats,
                    cautious)


@numba.njit
def land_on(p, sq, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
            in_jail, jail_rolls, bank, variant, stats, decks, dp, depth,
            cautious):
    if depth > 3 or not alive[p]:
        return
    t = SQ_TYPE[sq]
    if t == SQ_STREET or t == SQ_RAIL or t == SQ_UTIL:
        o = owner[sq]
        if o == -1:
            offer_unowned(p, sq, cash, owner, mort, houses, alive, bank,
                          variant, stats, cautious)
        elif o != p and not mort[sq]:
            if t == SQ_STREET:
                rent = street_rent(sq, o, houses, owner)
            elif t == SQ_RAIL:
                n = count_owned(o, 8, owner)
                if n == 1:
                    rent = 25
                elif n == 2:
                    rent = 50
                elif n == 3:
                    rent = 100
                else:
                    rent = 200
            else:
                r1 = np.random.randint(1, 7)
                r2 = np.random.randint(1, 7)
                mult = 10 if count_owned(o, 9, owner) == 2 else 4
                rent = (r1 + r2) * mult
            stats[ST_RENT] += rent
            spend(p, rent, o, 0, cash, owner, mort, houses, alive, goojf,
                  bank, stats)
    elif t == SQ_TAX:
        spend(p, TAX[sq], -1, is_jackpot(variant), cash, owner,
              mort, houses, alive, goojf, bank, stats)
    elif t == SQ_CHANCE:
        apply_chance(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                     in_jail, jail_rolls, bank, variant, stats, decks, dp,
                     depth, cautious)
    elif t == SQ_CHEST:
        apply_chest(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                    in_jail, jail_rolls, bank, variant, stats, decks, dp,
                    depth)
    elif t == SQ_PARK:
        if is_jackpot(variant) and bank[2] > 0:
            if bank[2] > stats[ST_POTMAX]:
                stats[ST_POTMAX] = bank[2]
            stats[ST_RECYCLED] += bank[2]  # pot payout: recycled, not printed
            cash[p] += bank[2]
            bank[2] = 0
    elif t == SQ_GOTOJAIL:
        send_to_jail(p, pos, in_jail, jail_rolls)
    # SQ_GO / SQ_JAIL: nothing happens


@numba.njit
def do_move(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf, in_jail,
            jail_rolls, bank, variant, stats, decks, dp, cautious):
    steps = d1 + d2
    new = pos[p] + steps
    salary = 0
    if new >= N_SQ:
        new -= N_SQ
        salary = GO_SALARY
    if is_doublego(variant) and new == 0:
        salary = GO_SALARY * 2  # $400 total for landing exactly on GO
        stats[ST_INJECTED] += GO_SALARY  # the extra $200 over the normal salary
    cash[p] += salary
    pos[p] = new
    land_on(p, new, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
            in_jail, jail_rolls, bank, variant, stats, decks, dp, 0, cautious)


@numba.njit
def end_of_turn(p, pos, cash, owner, mort, houses, alive, goojf, in_jail,
                jail_rolls, bank, variant, stats, decks, dp, cautious):
    if not alive[p]:
        return
    try_trade(p, cash, owner, mort, houses, alive, stats)
    if not alive[p]:
        return
    build_phase(p, cash, owner, mort, houses, bank, stats, cautious)
    unmortgage_phase(p, cash, owner, mort)


@numba.njit
def take_turn(p, pos, cash, owner, mort, houses, alive, goojf, in_jail,
              jail_rolls, bank, variant, stats, decks, dp, cautious):
    to_pot = is_jackpot(variant)
    if in_jail[p]:
        if goojf[p] > 0:
            goojf[p] -= 1
            in_jail[p] = 0
        elif cash[p] > 300:
            if not spend(p, 50, -1, to_pot, cash, owner, mort, houses, alive,
                         goojf, bank, stats):
                return
            in_jail[p] = 0
        else:
            d1 = np.random.randint(1, 7)
            d2 = np.random.randint(1, 7)
            jail_rolls[p] += 1
            if d1 == d2 or jail_rolls[p] >= 3:
                if jail_rolls[p] >= 3 and d1 != d2:
                    if not spend(p, 50, -1, to_pot, cash, owner, mort,
                                 houses, alive, goojf, bank, stats):
                        return
                in_jail[p] = 0
                do_move(p, d1, d2, pos, cash, owner, mort, houses, alive,
                        goojf, in_jail, jail_rolls, bank, variant, stats,
                        decks, dp, cautious)
                if not alive[p]:
                    return
            end_of_turn(p, pos, cash, owner, mort, houses, alive, goojf,
                        in_jail, jail_rolls, bank, variant, stats, decks, dp,
                        cautious)
            return
    # normal dice sequence (also used immediately after leaving jail)
    doubles = 0
    while True:
        d1 = np.random.randint(1, 7)
        d2 = np.random.randint(1, 7)
        if is_snakeeyes(variant) and d1 == 1 and d2 == 1:
            cash[p] += 500
            stats[ST_INJECTED] += 500
        if d1 == d2:
            doubles += 1
        else:
            doubles = 0
        if doubles >= 3:
            send_to_jail(p, pos, in_jail, jail_rolls)
            break
        do_move(p, d1, d2, pos, cash, owner, mort, houses, alive, goojf,
                in_jail, jail_rolls, bank, variant, stats, decks, dp, cautious)
        if not alive[p]:
            break
        if d1 != d2:
            break
    end_of_turn(p, pos, cash, owner, mort, houses, alive, goojf, in_jail,
                jail_rolls, bank, variant, stats, decks, dp, cautious)


@numba.njit
def play_game(seed, variant, cautious, stats):
    np.random.seed(seed)
    cash = np.empty(N_PLAYERS, np.int64)
    pos = np.empty(N_PLAYERS, np.int64)
    in_jail = np.empty(N_PLAYERS, np.int64)
    jail_rolls = np.empty(N_PLAYERS, np.int64)
    goojf = np.empty(N_PLAYERS, np.int64)
    alive = np.empty(N_PLAYERS, np.int64)
    for p in range(N_PLAYERS):
        cash[p] = START_CASH
        pos[p] = 0
        in_jail[p] = 0
        jail_rolls[p] = 0
        goojf[p] = 0
        alive[p] = 1
    owner = np.empty(N_SQ, np.int64)
    mort = np.empty(N_SQ, np.int64)
    houses = np.empty(N_SQ, np.int64)
    for s in range(N_SQ):
        owner[s] = -1
        mort[s] = 0
        houses[s] = 0
    bank = np.empty(3, np.int64)
    bank[0] = 32   # houses left
    bank[1] = 12   # hotels left
    bank[2] = 0    # free-parking pot
    decks = np.empty((2, 16), np.int64)
    for i in range(16):
        decks[0, i] = i
        decks[1, i] = i
    shuffle_deck(decks[0])
    shuffle_deck(decks[1])
    dp = np.zeros(2, np.int64)
    for k in range(N_STATS):
        stats[k] = 0

    turns = 0
    while turns < MAX_TURNS and alive_count(alive) > 1:
        for p in range(N_PLAYERS):
            if alive_count(alive) <= 1 or turns >= MAX_TURNS:
                break
            if not alive[p]:
                continue
            turns += 1
            take_turn(p, pos, cash, owner, mort, houses, alive, goojf,
                      in_jail, jail_rolls, bank, variant, stats, decks, dp,
                      cautious)
            if stats[ST_FIRSTMONO] == 0:
                # first completed street monopoly by any player this game
                for g in range(8):
                    found = False
                    for q in range(N_PLAYERS):
                        if owns_all(q, g, owner):
                            stats[ST_FIRSTMONO] = turns
                            found = True
                            break
                    if found:
                        break

    stats[ST_TURNS] = turns
    if turns > 1500:
        # nightmare-game end-state snapshot (surviving players only)
        stats[ST_NM_COUNT] = 1
        nalive = 0
        for p in range(N_PLAYERS):
            if alive[p]:
                nalive += 1
                stats[ST_NM_CASH] += cash[p]
        stats[ST_NM_ALIVE] = nalive
        for s in range(N_SQ):
            if owner[s] >= 0 and alive[owner[s]]:
                if houses[s] == 5:
                    stats[ST_NM_HOTELS] += 1
                else:
                    stats[ST_NM_HOUSES] += houses[s]
    if alive_count(alive) > 1:
        stats[ST_TRUNCATED] = 1
        best = 0
        bw = -1
        for p in range(N_PLAYERS):
            if alive[p]:
                w = net_worth(p, cash, owner, mort, houses)
                if w > bw:
                    bw = w
                    best = p
        stats[ST_WINNER] = best
    else:
        for p in range(N_PLAYERS):
            if alive[p]:
                stats[ST_WINNER] = p


@numba.njit
def play_games(variant, n, seed0, out, cautious):
    for i in range(n):
        s = np.empty(N_STATS, np.int64)
        play_game(seed0 + i, variant, cautious, s)
        for k in range(N_STATS):
            out[i, k] = s[k]


# ------------------------------------------------- movement-only validation
@numba.njit
def movement_sim(n_rolls, counts):
    """Token movement with dice + movement cards + jail. No economy.
    counts[sq] accumulates one count per roll's final square (jail rolls
    count as occupying square 10)."""
    pos = 0
    in_jail = 0
    jr = 0
    doubles = 0
    for _ in range(n_rolls):
        d1 = np.random.randint(1, 7)
        d2 = np.random.randint(1, 7)
        if in_jail:
            jr += 1
            if d1 == d2 or jr >= 3:
                # leaving jail: this roll counts as the landing square, not jail
                in_jail = 0
                jr = 0
                pos += d1 + d2
                if pos >= N_SQ:
                    pos -= N_SQ
            else:
                counts[JAIL_SQ] += 1
                continue
        else:
            if d1 == d2:
                doubles += 1
            else:
                doubles = 0
            if doubles >= 3:
                in_jail = 1
                jr = 0
                doubles = 0
                pos = JAIL_SQ
                counts[JAIL_SQ] += 1
                continue
            pos += d1 + d2
            if pos >= N_SQ:
                pos -= N_SQ
        for _ in range(6):
            if pos == 7 or pos == 22 or pos == 36:
                c = np.random.randint(0, 16)
                if c == CH_BOARDWALK:
                    pos = 39
                elif c == CH_GO:
                    pos = 0
                elif c == CH_ILLINOIS:
                    pos = 24
                elif c == CH_STCHARLES:
                    pos = 11
                elif c == CH_RAIL_A or c == CH_RAIL_B:
                    pos = nearest_rail(pos)
                elif c == CH_UTIL:
                    pos = nearest_util(pos)
                elif c == CH_BACK3:
                    pos = (pos - 3 + N_SQ) % N_SQ
                elif c == CH_JAIL:
                    in_jail = 1
                    jr = 0
                    pos = JAIL_SQ
                    break
                else:
                    break
            elif pos == 2 or pos == 17 or pos == 33:
                c = np.random.randint(0, 16)
                if c == CC_GO:
                    pos = 0
                elif c == CC_JAIL:
                    in_jail = 1
                    jr = 0
                    pos = JAIL_SQ
                    break
                else:
                    break
            elif pos == 30:
                in_jail = 1
                jr = 0
                pos = JAIL_SQ
                break
            else:
                break
        counts[pos] += 1
