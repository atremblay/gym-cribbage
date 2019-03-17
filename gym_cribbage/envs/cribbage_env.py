# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2019-03-03 16:47:17
# @Last Modified by:   Alexis Tremblay
# @Last Modified time: 2019-03-09 14:07:20


from itertools import product, combinations
import random
import numpy as np
import gym


SUITS = "♤♡♧♢"
RANKS = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K"]

# Starting the idx at 1 because 0 will be used as padding
rank_to_idx = {r: i for i, r in enumerate(RANKS, 1)}
suit_to_idx = {s: i for i, s in enumerate(SUITS, 1)}


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
    def short_state(self):
        # [0:3] encode suit, [4:16] encode rank
        s = np.zeros(4 + 13)
        s[SUITS.index(self.suit)] = 1
        s[RANKS.index(self.rank) + 4] = 1
        return s

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
        return self.cards.pop(0)

    def __len__(self):
        return len(self.cards)


class Stack(object):
    """A generic stack of cards."""

    @staticmethod
    def from_stack(stack):
        return Stack(cards=stack.cards)

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
    def short_state(self):
        """
        Possibly for state aggregation.
        """
        # [0:3] encode suit, [4:16] encode rank
        s = np.zeros(4 + 13)
        for card in self.cards:
            s += card.state
        return s

    def add(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        return Stack(cards=self.cards + [card])

    def add_(self, card):
        if not isinstance(card, Card):
            raise ValueError("Can only add card to a hand.")
        self.cards.append(card)

    def __repr__(self):
        return " | ".join([str(c) for c in self.cards])

    def __iter__(self):
        for card in self.cards:
            yield card

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, idx):
        if not isinstance(idx, (int, slice)):
            raise ValueError("Index must be integer or slice")
        return self.cards[idx]


class CribbageEnv(gym.Env):
    """
    Cribbage class calculates the points during the pegging phase.
    When a player step through the environment, this class returns the state as
    a tuple of cards that are currently used in this pegging round and the
    previous cards played in previous pegging rounds. The reward is the number
    of point following the last card played. The
    """

    def __init__(self):
        super(CribbageEnv, self).__init__()

    def reset(self):
        """
        Call this when a new deck is created.
        """
        self.past_plays = Stack()
        self.current_play = Stack()

    def new_play(self):
        """
        This method transfers the cards from the current play stack to the
        past play stack and resets the current play stack. Call this when
        no player can play or the total points on the table is 31.
        """
        for card in self.current_play:
            self.past_plays.add_(card)
        self.current_play = Stack()

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
        self.current_play.add_(card)
        reward = self._evaluate_current_play()
        total = np.array([c.value for c in self.current_play]).sum()

        return (self.current_play, self.past_plays), reward, total

    def _evaluate_current_play(self):
        points = 0
        # Check if last cards are of the same rank value
        num_cards = len(self.current_play)
        rank_values = [c.rank_value for c in self.current_play]
        values = [c.value for c in self.current_play]
        for i in range(num_cards - 1):
            if len(set(rank_values[i:])) == 1:
                x = num_cards - i
                points += x * (x - 1)
                break

        if sum(values) == 15:
            points += 2

        for i in range(num_cards - 2):
            n = num_cards - i
            if is_sequence(self.current_play[i:]):
                points += n
                break

        return points


def evaluate_hand(hand, knob=None, is_crib=False):
    """
    This is to evaluate the number of points in a hand. Optionnally with the
    knob
    """
    hand_without_knob = Stack.from_stack(hand)
    if knob is not None:
        hand = hand.add(knob)

    # List of all card combinations: 2 in n, 3 in n, ..., n in n
    all_combinations = []
    for i in range(2, len(hand) + 1):
        all_combinations.append(list(combinations(hand, i)))

    points = 0
    # Check for pairs. Check only the combinations of two cards
    for left, right in all_combinations[0]:
        if left.rank_value == right.rank_value:
            points += 2

    # Check for suits
    # Starting with full deck
    # Need at least 3 cards for this
    found_sequence = False
    for comb in reversed(all_combinations[1:]):
        for comb_ in comb:
            if is_sequence(comb_):
                points += len(comb_)
                found_sequence = True
        if found_sequence:
            break

    points += same_suit_points(hand_without_knob, knob, is_crib)

    # Check for 15
    # Starting with every two cards combinations
    for comb in all_combinations:
        for comb_ in comb:
            cards = list(sorted(comb_))
            if sum(c.value for c in cards) == 15:
                points += 2

    if knob is not None:
        for card in hand_without_knob:
            if card.rank == "J" and card.suit == knob.suit:
                points += 1

    return points


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
    return (rank_to_idx[card.rank], suit_to_idx[card.suit])


def stack_to_idx(stack):
    return tuple(
        zip(*[card_to_idx(c) for c in stack])
    )
