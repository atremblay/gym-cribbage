# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2019-03-03 16:47:17
# @Last Modified by:   Joseph D Viviano
# @Last Modified time: 2019-03-22 19:51:20

from copy import copy
from itertools import product, combinations
import gym
import logging
import numpy as np
import random
from collections import defaultdict

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

# For debug information.
logging.basicConfig(
    level=logging.WARN, format="[%(lineno)s: %(funcName)24s] %(message)s")


class Card(object):
    """Card, french style"""

    def __init__(self, rank, suit):
        super(Card, self).__init__()
        self.rank = rank
        self.suit = suit

    @property
    def value(self):
        if isinstance(self.rank, str):
            if self.rank == "A":
                return 1
            if self.rank in "JQK":
                return 10
        return self.rank

    @property
    def rank_value(self):
        if self.rank == "A":
            return 1
        elif self.rank == 'J':
            return 11
        elif self.rank == 'Q':
            return 12
        elif self.rank == 'K':
            return 13

        return self.rank

    @property
    def state(self):
        # One-hot encode the card
        s = np.zeros(52)
        idx = SUITS.index(self.suit) * 13 + RANKS.index(self.rank)
        s[idx] = 1
        return s

    @property
    def compact_state(self):
        # [0:3] encode suit, [4:16] encode rank
        suit = np.zeros(4)
        rank = np.zeros(13)
        suit[SUITS.index(self.suit)] = 1
        rank[RANKS.index(self.rank)] = 1
        return suit, rank

    @staticmethod
    def rank_suit_from_idx(idx):
        idx_rank = idx % 13
        idx_suit = (idx-idx_rank)//13
        return RANKS[idx_rank], SUITS[idx_suit]

    def __repr__(self):
        return "{}{}".format(self.rank, self.suit)

    def __str__(self):
        return "{}{}".format(self.rank, self.suit)

    def __eq__(self, card):
        return self.rank == card.rank and self.suit == card.suit

    def __ge__(self, card):
        return self.rank_value >= card.rank_value

    def __gt__(self, card):
        return self.rank_value > card.rank_value

    def __le__(self, card):
        return self.rank_value <= card.rank_value

    def __lt__(self, card):
        return self.rank_value < card.rank_value


class Deck(object):
    """Deck of 52 cards. Automatically suffles at creation."""

    def __init__(self):
        super(Deck, self).__init__()
        self.cards = [Card(rank, suit) for rank, suit in product(RANKS, SUITS)]

        random.shuffle(self.cards)

    def deal(self):
        try:
            return self.cards.pop(0)
        except IndexError:
            return None

    def remove(self, card):
        new_cards = []
        for c in self.cards:
            if c != card:
                new_cards.append(c)
        new_deck = Deck()
        new_deck.cards = new_cards
        return new_deck

    def remove_(self, card):
        for i, c in enumerate(self.cards):
            if c == card:
                self.cards.pop(i)

    def __len__(self):
        return len(self.cards)


class Stack(object):
    """A generic stack of cards."""

    @staticmethod
    def from_stack(stack):
        return Stack(cards=stack.cards.copy())

    def __init__(self, cards=None):
        super(Stack, self).__init__()
        if cards is None:
            self.cards = []
        else:
            self.cards = cards

    def play(self, card):
        for i, c in enumerate(self.cards):
            if card == c:
                return self.cards.pop(i)
        raise ValueError("{} not in hand. Cannot play this".format(card))

    def discard(self, card):
        """
        Same thing as play(). It's just for semantics
        """
        self.play(card)

    @property
    def state(self):
        # One-hot encode the hand
        s = np.zeros(52)
        for card in self.cards:
            s += card.state
        return s

    @property
    def compact_state(self):
        """
        Possibly for state aggregation.
        """
        # [0:3] encode suit, [4:16] encode rank
        suit = np.zeros((4, len(self)), dtype=np.float32)
        rank = np.zeros((13, len(self)), dtype=np.float32)
        for i, card in enumerate(self.cards):
            s, r = card.compact_state
            suit[:, i] = s
            rank[:, i] = r

        argsort = np.argsort([c.rank_value for c in self])
        suit = suit[:, argsort]
        rank = rank[:, argsort]
        return suit, rank

    def add(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        return Stack(cards=self.cards + [card])

    def add_(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        self.cards.append(card)

    def remove(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        return Stack(cards=[c for c in self.cards if c != card])

    def remove_(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        self.cards = [c for c in self.cards if c != card]

    def __repr__(self):
        if len(self.cards) == 0:
            return("empty")
        else:
            return("-".join([str(c) for c in self.cards]))

    def __iter__(self):
        for card in self.cards:
            yield card

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, idx):
        if not isinstance(idx, (int, slice)):
            raise ValueError("Index must be integer or slice")
        return self.cards[idx]


class State(object):
    """
    Contains the state of the current hand. The state tells the external world:
    1) The hand (playable cards only) of the current player.
    2) The ID of the current player.
    3) The ID of the player who recieves this turn's reward.
    4) The phase of the game {0: the deal, 1: the play, 2: the show}.
    """

    def __init__(self, hand, hand_id, reward_id, phase,
                 player_score, opponent_score):
        self.hand = hand
        self.hand_id = hand_id
        self.reward_id = reward_id
        self.phase = phase
        self.player_score = player_score
        self.opponent_score = opponent_score

class CribbageEnv(gym.Env):
    """
    Cribbage class calculates the points during the pegging phase.
    When a player step through the environment, this class returns the state as
    a tuple of cards that are currently used in this pegging round and the
    previous cards played in previous pegging rounds. The reward is the number
    of point following the last card played. TODO: The
    """

    def __init__(self, n_players=2, verbose=False):
        super(CribbageEnv, self).__init__()

        self.n_players = n_players
        if self.n_players < 2 or self.n_players > 4:
            raise ValueError("Cribbage is played by 2-4 players.")

        if self.n_players == 2:
            self._cards_per_hand = 6
        else:
            self._cards_per_hand = 5

        self.logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        self.initialized = False

    def reset(self, dealer=None):
        """
        Resets the hand, additionally clearing the scoreboard.
        """
        self.logger.debug("New Game!")

        # Reset the persistant scores of all players.
        self.scores = np.zeros(self.n_players, dtype=np.int8)

        # Allows the user to see whether we are dealing with a new hand.
        self.new_hand = True

        # Pick dealer, clear table, shuffle, deal cards.
        reward, done, _ = self._reset_hand(dealer=dealer)

        self.initialized = True

        return(self.state, reward, done, "Reset Game!")

    def step(self, card):
        """
        Add the card to the current play and calculates the reward for it.
        Never checks if the move is illegal (i.e. total card value is
        higher than 31)

        Params
        ======
            card: Card
                A card object

        Returns
        =======
            (current play, past plays), points, total: (Stack, Stack), int, int
            points is the reward for the last card played
            total is the cummulative value of the card played in the current
            play.
        """
        if not self.initialized:
            raise Exception("Need to CribbageEnv.reset() before first step.")

        done = False
        self.new_hand = False
        debug = "step!"

        # The Deal.
        if self.phase == 0:
            self.logger.debug(
                "Player {} discards {} to the crib".format(self.player, card)
            )
            # Move card from hand to crib.
            self.hands[self.player].discard(card)
            self.crib.add_(card)

            # Keep track of number of cards in play.
            counts, playable_hands = self._count_playable_cards()
            reward = 0
            self.last_player = copy(self.player)

            # The crib is complete.
            if sum(counts) / float(self.n_players) == 4:
                self.phase = 1
                self.starter = [self.deck.deal()]

                self.logger.debug("Starter drawn={}".format(self.starter))

                # 2 for his heels.
                if self.starter[0].rank == "J":
                    reward = 2
                    self.logger.debug("Two for his heels!")

                # Start next phase from the left of the dealer.
                self.player = self.next_player(self.player, from_dealer=True)
                # self.logger.debug("Crib: {}".format(self.crib))
                self.logger.debug(
                    f"Crib complete: {self.crib}  Move to The Play."
                )

            else:
                self.player = self.next_player(self.player)

            # Keep track of the player's total score.
            self.scores[self.last_player] += reward
            assert (self.scores>=0).all()
            player_score, opponent_scores = self._get_scores()
            self.state = State(
                Stack(playable_hands[self.player]),
                self.player,
                self.last_player,
                self.phase,
                player_score,
                opponent_scores
            )

        # The Play.
        elif self.phase == 1:
            self.logger.debug(
                "Player {} plays {}".format(self.player, card)
            )
            # Move card from player's hand to table. Keep track of player's
            # played cards in "played", which we need for The Show.
            self.hands[self.player].discard(card)
            self.played[self.player].add_(card)
            self.table.add_(card)
            reward = self._evaluate_play()
            assert reward>=0
            self._update_table_value()

            # Check to see who can play next.
            counts, playable_hands = self._count_playable_cards()

            # self.last_player recieves the reward.
            self.last_player = copy(self.player)

            # Go! If no one else can play, give this player an extra 2 points.
            if sum(counts) == 0:

                # Reward player for placing the last card.
                if self.table_value == MAX_TABLE_VALUE:
                    reward += 2
                    self.logger.debug("reward+2 for MAX_TABLE_VALUE.")
                else:
                    reward += 1
                    self.logger.debug("reward+1 for last player.")


                remaining_cards = self._count_remaining_cards()

                # Move onto The Show.
                if remaining_cards == 0:
                    self.logger.debug("No cards left, time for The Show.")
                    self.phase = 2
                    self.player = self.next_player(self.player,
                                                   from_dealer=True)

                # Reset the table and playable cards.
                else:
                    self.logger.debug(
                        "Resetting table! table_value={} n_cards={}".format(
                            self.table_value, remaining_cards)
                    )
                    self._reset_table()
                    self.player = self.next_player(self.player)
                    counts, playable_hands = self._count_playable_cards()
                    self._next_avail_player(counts, playable_hands)

            # Go! Skip to the next player who has a playable hand.
            else:
                self.player = self.next_player(self.player)

                self._next_avail_player(counts, playable_hands)

            # Keep track of the player's total score.
            self.scores[self.last_player] += reward
            assert (self.scores>=0).all()

            # When self.phase == 2, playable_hands[self.player] will be empty.
            player_score, opponent_scores = self._get_scores()
            self.state = State(
                Stack(playable_hands[self.player]),
                self.player,
                self.last_player,
                self.phase,
                player_score,
                opponent_scores
            )

            self.prev_phase = 1

        # The Show.
        elif self.phase == 2:


            # Calculate points for self.player.
            reward = self._evaluate_show()
            assert reward>=0

            # Went around the circle once. This hand is over.
            if self.player == self.dealer:
                self.new_hand = True

            self.last_player = copy(self.player)
            self.player = self.next_player(self.player)

            # Keep track of the player's total score.
            self.scores[self.last_player] += reward
            assert (self.scores>=0).all()
            player_score, opponent_scores = self._get_scores()
            self.state = State(
                Stack([]),
                self.player,
                self.last_player,
                self.phase,
                player_score,
                opponent_scores
            )

            self.prev_phase = 2

        # If any player, at any time, gets a winning amount of points.
        if any(self.scores >= MAX_ROUND_VALUE):
            done = True

            # Forces user to reset the environment for the next game.
            self.new_hand = False
            self.initialized = False

        # If we go around the circle once during Phase 2.
        elif self.new_hand:

            # The next hand is dealt by the person next to the dealer.
            next_dealer = self.next_player(self.dealer)
            self._reset_hand(dealer=next_dealer, reward_id=self.state.reward_id)

        return(self.state, reward, done, debug)

    def next_player(self, player, from_dealer=False):
        """
        Increments through the players. Increments forever, but can be set
        to start from the dealer.
        """
        if from_dealer:
            player = copy(self.dealer)

        player += 1
        if player > self.n_players - 1:
            player = 0

        self.logger.debug("Player={}".format(self.player))
        return player

    def render(self, mode='human'):
        """Renders a table of the current game."""
        # Get information from each player.
        mp_dict = {}
        mp_dict.update(self._get_item_dict("hand", self.hands))
        mp_dict.update(self._get_item_dict("played", self.played))
        mp_dict.update(self._get_item_dict("score", self.scores))

        # Common information.
        table_dict = {"crib": self.crib, "table": self.table,
                      "discarded": self.discarded}

        table_mp = TABLE_MP.format(**mp_dict)
        table = TABLE.format(**table_dict)

        # Split input data by row and then on spaces
        rows_mp = self._get_rows(table_mp)
        rows = self._get_rows(table)

        # Print each row using the associated format
        for row in rows_mp:
            print(ROW_MP.format(*row))
        for row in rows:
            print(ROW.format(*row))

    def close(self):
        pass

    def _get_scores(self):
        player_score = self.scores[self.player]
        opponent_scores = self.scores[np.setdiff1d(range(self.n_players),
                                                         self.player)]

        return(player_score, opponent_scores)

    def _get_rows(self, iterable):
        """Split input data by row and then on spaces."""
        return([ line.strip().split(' ') for line in iterable.split('\n') ])

    def _get_item_dict(self, name, items):
        """
        Used for getting dicts representing the internal state of the
        environment, for the render function.
        """
        item_dict = {
            "{}{}".format(name, i+1): items[i] for i in range(self.n_players)}
        if self.n_players < 4:
            empty_dict = {
                "{}{}".format(name, i+1): "N/A" for i in range(
                    self.n_players, 4)}
            item_dict.update(empty_dict)

        return(item_dict)

    def _update_table_value(self):
        """
        Calculates the value of all cards played on the table.
        """
        self.table_value = np.array([c.value for c in self.table]).sum()

    def _count_playable_cards(self):
        """
        Counts the number of cards in each player's hand that can be
        legally played, i.e., adding them to the table would not make
        the table go over 31.
        """
        counts, playable_hands = [], []

        for hand in self.hands:
            count = 0
            playable_hand = []

            for card in hand:
                if self.table_value + card.value <= MAX_TABLE_VALUE:
                    count += 1
                    playable_hand.append(card)

            counts.append(count)
            playable_hands.append(playable_hand)

        self.logger.debug("Table={}, playable cards={}".format(
            self.table_value, playable_hands)
        )

        return(counts, playable_hands)

    def _count_remaining_cards(self):
        """Counts the sum of the cards in all hands."""
        remaining_cards = 0

        for hand in self.hands:
            remaining_cards += len(hand)

        self.logger.debug('Total remaining cards={}'.format(remaining_cards))

        return(remaining_cards)

    def _next_avail_player(self, counts, playable_hands):
        """
        Finds the next available player if has any
        """
        if sum(counts) != 0:
            while counts[self.player] == 0:
                self.logger.debug("Go! Skip player {}, hand={}/{}".format(
                    self.player,
                    playable_hands[self.player],
                    self.hands[self.player])
                )
                self.player = self.next_player(self.player)

    def _reset_table(self):
        """
        This method moves all cards on the table to a discard pile and
        clears the table by initialzing an empty stack. Called when
        no player can play or the total points on the table is 31.
        """
        for card in self.table.cards:
            self.discarded.add_(card)

        self.table_value = 0
        self.table = Stack()

    def _reset_hand(self, dealer=None, reward_id=None):
        """
        All the steps required to start a new hand. Shuffles the deck, deals
        cards to each of the n_player's hands, and randomly selects the
        dealer. Each user receives the appropriate number of cards.
        """
        self.logger.debug("New hand!")

        self.deck = Deck()

        # Stores the playable cards in each player's hand.
        self.hands = [Stack() for i in range(self.n_players)]

        # Stores the cards played by each player.
        self.played = [Stack() for i in range(self.n_players)]

        # Stores the cards played by each player (in order) for The Play.
        self.table = Stack()

        # Stores the crib generated during The Deal.
        self.crib = Stack()
        self.starter = Stack()
        self.discarded = Stack()

        # Randomly select the dealer. Initalize the player to be the same.
        self.dealer = random.randint(0, self.n_players - 1) if dealer is None \
                                                            else dealer

        self.logger.debug("Player {} has the crib".format(self.dealer))
        self.player = copy(self.dealer)
        self.last_player = copy(self.dealer)

        self.table_value = 0
        self.phase = 0  # 0: the deal, 1: the play, 2: the show.
        self.prev_phase = 0  # To catch phase transitions

        # Deal cards to all users.
        for i in range(self.n_players):
            for j in range(self._cards_per_hand):
                self.hands[i].add_(self.deck.deal())
            self.logger.debug("Player {}'s hand: {}".format(i, self.hands[i]))

        # Return the hand of the dealer.
        player_score, opponent_scores = self._get_scores()
        self.state = State(
            self.hands[self.player],
            self.player,
            reward_id,
            self.phase,
            player_score,
            opponent_scores
        )

        reward = 0
        done = False

        return(reward, done, "Reset Hand!")

    def _evaluate_play(self):
        """
        Evaluates points for the last-played card during The Play.
        These calculations do not include the starter.
        """
        points = evaluate_table(self.table)

        self.logger.debug('PLAY: player {} earned {} points'.format(
            self.player, points)
        )

        return(points)

    def _evaluate_show(self):
        """
        Evaluates points for a given set of cards during The Show.
        These calculations include the starter. If the player is the dealer,
        also add the points from the crib.
        """
        points = evaluate_cards(
            self.played[self.player],
            starter=self.starter[0]
        )

        self.logger.debug('SHOW: player {} earned {} points'.format(
            self.player, points)
        )

        if self.player == self.dealer:
            crib_points = evaluate_cards(
                self.crib,
                starter=self.starter[0],
                is_crib=True
            )
            points += crib_points
            self.logger.debug('SHOW CRIB: player {} earned {} points'.format(
                self.player, points)
            )

        return(points)


def evaluate_table(cards):
    points = 0

    if sum(c.value for c in cards) == 15:
        points += 2

    # Pair points
    pair_point = 0
    for i in range(-2, -min(len(cards)+1, 5), -1):
        if cards[-1].rank == cards[i].rank:
            if i == -2:
                pair_point = 2
            elif i == -3:
                pair_point = 6
            elif i == -4:
                pair_point = 12
        else:
            break

    points += pair_point

    # Run points
    for i in reversed(range(-3, -len(cards)-1, -1)):
        if is_sequence(cards[i:]):
            points += len(cards[i:])
            break

    return points


def evaluate_cards(cards, starter=None, is_crib=False):
    """
    This is to evaluate the number of points in a hand. Optionally with the
    knob
    """

    points = 0

    # If only one card on the table.
    if len(cards) == 1:
        return(points)

    # Sequence of cards, with or without the starter.
    cards_without_starter = Stack.from_stack(cards)
    if starter is not None:
        cards = cards.add(starter)

    # List of all card combinations: 2 in n, 3 in n, ..., n in n
    all_combinations = defaultdict(list)
    for i in range(2, len(cards) + 1):
        all_combinations[i].extend(list(combinations(cards, i)))

    # Check for pairs. Check only the combinations of two cards
    for combination in all_combinations[2]:
        left, right = combination

        if left.rank_value == right.rank_value:
            points += 2

    # Check for suits, starting with full deck. Minimum 3 cards.
    # Since we reverse through all_combinations, finds the longest sequence.
    sequence_found = False

    for length in reversed(sorted(all_combinations.keys())):
        if length < 3:
            continue

        for combination in all_combinations[length]:
            if is_sequence(combination):
                points += len(combination)
                sequence_found = True

        if sequence_found:
            break

    points += same_suit_points(cards_without_starter, starter, is_crib)

    # Check for 15s, starting with two card combinations.
    for length in sorted(all_combinations.keys()):
        for combination in all_combinations[length]:
            cards = list(sorted(combination))
            if sum(c.value for c in cards) == 15:
                points += 2

    # Check for cards with same suit as starter.
    if starter is not None:
        for card in cards_without_starter:
            if card.rank == "J" and card.suit == starter.suit:
                points += 1

    return(points)


def is_sequence(cards):
    # Need at least 3 cards
    if len(cards) < 3:
        return False

    rank_values = list(sorted([c.rank_value for c in cards]))
    for i, val in enumerate(rank_values[:-1], 1):
        if (val + 1) != rank_values[i]:
            return False
    return True


def same_suit_points(hand, knob, is_crib=False):
    # Check if all same suit. Small detail, if is_crib, you need all 5 cards
    # to be of the same suit. Otherwise you only need the cards in your hands
    # to be of the same suit. If the knob is also of the same suit then you
    # get an extra point

    points = 0
    hand_without_knob = Stack.from_stack(hand)
    if knob is not None:
        hand = hand.add(knob)

    if is_crib:
        # Checking the hand that includes the knob
        if len(set([c.suit for c in hand])) == 1:
            points += len(hand)
    else:
        if len(set([c.suit for c in hand_without_knob])) == 1:
            points += len(hand_without_knob)
            if knob is not None and knob.suit == hand_without_knob[0].suit:
                points += 1
    return points


def card_to_idx(card):
    return (RANK_TO_IDX[card.rank], SUIT_TO_IDX[card.suit])


def stack_to_idx(stack):
    return tuple(
        zip(*[card_to_idx(c) for c in stack])
    )


if __name__ == "__main__":

    env = CribbageEnv(verbose=True)
    state, reward, done, debug = env.reset()

    while not done:

        if env.phase < 2:
            state, reward, done, debug = env.step(state.hand[0])
        else:
            state, reward, done, debug = env.step([])
        env.render()
