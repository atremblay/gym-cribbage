# -*- coding: utf-8 -*-

SUITS = "♤♡♧♢"
RANKS = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K"]

# Starting the idx at 1 because 0 will be used as padding
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS, 1)}
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS, 1)}

# Render: Used to render player-specific stats.
TABLE_MP = """--- Player1 Player2 Player3 Player4
Hand {hand1} {hand2} {hand3} {hand4}
Played {played1} {played2} {played3} {played4}
Score {score1} {score2} {score3} {score4}
--- --- --- --- ---"""
ROW_MP = "{:12s} {:20s} {:20s} {:20s} {:20s}"

# Render: Used to render the in-play cards.
TABLE = """Crib {crib}
Table {table}
Discarded {discarded}"""
ROW = "{:12s} {:60s}"

MAX_TABLE_VALUE = 31  # Max points allowed before hand reset.
MAX_ROUND_VALUE = 121  # Max points allowed before game ends.

