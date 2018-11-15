from card import Card
from random import shuffle
from player import Human, Baseline, Idiot, Oracle
from agents import ModelPlayer, QModel, ModelTest

import numpy as np


class Game:
    """
    Stateful class holding the entire game state
    """

    END_SCORE = 500 - 1
    NUM_PLAYERS = 4

    def __init__(self, args):
        self.deck = ([Card(i) for i in range(Card.NUM_CARDS)])
        shuffle(self.deck)
        self.players = [Baseline([], "AI Baseline " + str(i + 1))
                        for i in range(0, Game.NUM_PLAYERS - 1)]
        if args.human:
            self.players.append(Human([], "Human"))
        elif args.oracle:
            self.players.append(Oracle([], "Oracle"))
        else:
            self.players.append(ModelTest([], "ModelTest"))
            #self.players.append(Baseline([], "Test"))
        shuffle(self.players)
        self.pile = []


    def getPlayerGameState(self, player, playerCursor):
        """
        @param Player player: _____
        @param int playerCursor: index of player who won last round (i.e. starts this round)
        """
        oracleGameState = self.getOracleGameState(playerCursor)
        # Remove other player hands...
        # replace with just our player's hand
        if not player.name == "Oracle":  # @GriffinKardos...when you make your oracle class change this
            oracleGameState[0] = player.hand
        return oracleGameState

    def getOracleGameState(self, playerCursor):
        """
        Get interesting state related to round.
        For the three lists, the 0 index is yourself, each successive element follows
        the ordering around the table clockwise.
        """
        playerHands = []
        playerClaimedCards = []
        playerBids = []
        playerBags = []
        for i in range(len(self.players)):
            player = self.players[(playerCursor + i) % 4]
            playerHands.append(player.hand)
            playerClaimedCards.append(player.claimed)
            playerBids.append(player.bid)
            playerBags.append(player.sandbags)
        return [playerHands, playerClaimedCards, playerBids, playerBags, self.pile, ]

    @staticmethod
    def stateFeatureExtractor(state, actions):
        '''
        @return np.array feature vector
        Features:
        1. playerHandF: [int], len 52
        2. claimedF: [int], len 52, among all players
        3. playerBids: [int], len 4
        4. pileF: [map], len 52
        5. tricksF: [int], len 4, number of bids each player has
        NOTE: do not call this method using the oracle
        NOTE: bags not included
        '''
        playerHand, playerClaimedCards, playerBids, playerBags, pile = state

        playerHandF = [0] * Card.NUM_CARDS
        for card in playerHand:
            playerHandF[card.index] = 1

        claimedF = [0] * Card.NUM_CARDS
        for cards in playerClaimedCards:
            for card in cards:
                claimedF[card.index] = 1
        
        pileF = [0] * Card.NUM_CARDS
        for i, card in enumerate(pile):
            pileF[card.index] = i + 1

        tricksF = [len(c) // 4 for c in playerClaimedCards] # should divide evenly

        return np.array([playerHandF, claimedF, playerBids, pileF, tricksF])