# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2019-03-03 16:47:17
# @Last Modified by:   Alexis Tremblay
# @Last Modified time: 2019-03-09 14:07:20

import unittest

from gym_cribbage.envs.cribbage_env import (
    is_sequence,
    Card,
    RANKS,
    SUITS,
    Stack,
    evaluate_cards,
    CribbageEnv,
    card_to_idx,
    stack_to_idx
)


class CribbageEnvTest(unittest.TestCase):

    def is_a_sequence(self):
        self.assertTrue(
            is_sequence(
                cards=[
                    Card(RANKS[4], SUITS[0]),
                    Card(RANKS[5], SUITS[1]),
                    Card(RANKS[6], SUITS[2]),
                    Card(RANKS[7], SUITS[3])
                ]
            ))

    def is_not_a_sequence(self):
        self.assertFalse(
            is_sequence(
                cards=[
                    Card(RANKS[4], SUITS[0]),
                    Card(RANKS[4], SUITS[1]),
                    Card(RANKS[4], SUITS[2]),
                    Card(RANKS[10], SUITS[3])
                ]
            ))

    def test_evaluate_cards(self):
        hand = Stack(
            cards=[
                Card(RANKS[0], SUITS[0]),
                Card(RANKS[2], SUITS[0]),
                Card(RANKS[3], SUITS[0]),
                Card(RANKS[5], SUITS[0])
            ]
        )

        # For a regular hand, if all cards in hand are of the same suit, then you
        # get the points. If the starter is also of the same suit you get an
        # extra point
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[7], SUITS[0])), 9)
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[7], SUITS[1])), 8)

        # Crib hand needs to have all same suit including the starter, otherwise
        # you don't get the points for the same suit
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[7], SUITS[0]), is_crib=True), 9)
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[7], SUITS[1]), is_crib=True), 4)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in range(5)])
        self.assertEqual(evaluate_cards(hand), 12)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in range(4)])
        starter = Card(RANKS[4], SUITS[0])
        self.assertEqual(evaluate_cards(hand, starter), 12)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in range(4)])
        starter = Card(RANKS[5], SUITS[0])
        self.assertEqual(evaluate_cards(hand, starter), 11)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in [1, 1, 2, 3, 4]])
        self.assertEqual(evaluate_cards(hand), 15)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in [1, 1, 1, 2, 3]])
        self.assertEqual(evaluate_cards(hand), 20)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in [1, 3, 5, 7]])
        starter = Card(RANKS[9], SUITS[0])
        self.assertEqual(evaluate_cards(hand, starter), 5)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in [1, 3, 5, 7]])
        starter = Card(RANKS[9], SUITS[1])
        self.assertEqual(evaluate_cards(hand, starter), 4)

        hand = Stack(cards=[Card(RANKS[i], SUITS[0]) for i in [1, 3, 5, 10]])
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[-1], SUITS[0])), 6)

        # Best possible hand
        hand = Stack(
            cards=[
                Card(RANKS[4], SUITS[0]),
                Card(RANKS[4], SUITS[1]),
                Card(RANKS[4], SUITS[2]),
                Card(RANKS[10], SUITS[3])
            ]
        )
        self.assertEqual(evaluate_cards(hand, starter=Card(RANKS[4], SUITS[3])), 29)

        hand = Stack(
            cards=[
                Card(RANKS[8], SUITS[3]),
                Card(RANKS[7], SUITS[1]),
                Card(RANKS[6], SUITS[1]),
                Card(RANKS[12], SUITS[3])
            ]
        )
        self.assertEqual(
            evaluate_cards(hand, starter=Card(RANKS[3], SUITS[0])), 5
        )

    # def test_cribbage_step(self):

    #     cribbage = CribbageEnv()
    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[1]))[1], 2)
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[2]))[1], 6)
    #     self.assertEqual(cribbage.step(Card(RANKS[1], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[2], SUITS[2]))[1], 3)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[1], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[2], SUITS[2]))[1], 3)
    #     self.assertEqual(cribbage.step(Card(RANKS[8], SUITS[2]))[1], 2)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[1]))[1], 2)
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[2]))[1], 6)
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[3]))[1], 12)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[4], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[9], SUITS[1]))[1], 2)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[4], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[10], SUITS[1]))[1], 2)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[4], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[11], SUITS[1]))[1], 2)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[4], SUITS[0]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[12], SUITS[1]))[1], 2)

    #     cribbage.reset()
    #     self.assertEqual(cribbage.step(Card(RANKS[0], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[2], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[3], SUITS[2]))[1], 0)
    #     self.assertEqual(cribbage.step(Card(RANKS[1], SUITS[2]))[1], 4)

    #     # Transfert cards from the current play to past plays
    #     cribbage.reset()
    #     cribbage.step(Card(RANKS[0], SUITS[2]))
    #     self.assertEqual(len(cribbage.current_play), 1)
    #     self.assertEqual(cribbage.current_play[0].suit, SUITS[2])
    #     self.assertEqual(cribbage.current_play[0].rank, RANKS[0])
    #     self.assertEqual(len(cribbage.past_plays), 0)
    #     cribbage.new_play()
    #     self.assertEqual(len(cribbage.current_play), 0)
    #     self.assertEqual(len(cribbage.past_plays), 1)
    #     self.assertEqual(cribbage.past_plays[0].suit, SUITS[2])
    #     self.assertEqual(cribbage.past_plays[0].rank, RANKS[0])

    def test_card_to_idx(self):
        self.assertEqual(card_to_idx(Card(RANKS[0], SUITS[2])), (1, 3))

    def test_stack_to_idx(self):
        hand = Stack(
            cards=[
                Card(RANKS[4], SUITS[0]),
                Card(RANKS[4], SUITS[1]),
                Card(RANKS[4], SUITS[2]),
                Card(RANKS[10], SUITS[3])
            ]
        )

        assert stack_to_idx(hand) == ((5, 5, 5, 11), (1, 2, 3, 4))


if __name__ == '__main__':
    unittest.main()
