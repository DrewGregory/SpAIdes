from card import Card
from random import shuffle, randint
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
        self.players = [Baseline([], "Oracle AI" + str(i + 1))
                        for i in range(0, Game.NUM_PLAYERS - 1)]
        if args.human:
            self.players.append(Human([], "Human"))
            self.players[-2] = ModelTest([], "Model Test")
        elif args.oracle:
            self.players.append(Oracle([], "Oracle"))
        elif args.idiot:
            self.players.append(Idiot([], "Idiot"))
        else:
            self.players.append(ModelTest([], "Model Test"))
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
        if not "Oracle" in player.name:  # @GriffinKardos...when you make your oracle class change this
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
    def stateFeatureExtractor(state, action):
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
        
        
        actions = [0] * 52
        if action:
            actions[action.index] = 1
        playerHandF = [float(0)] * Card.NUM_CARDS
        for card in playerHand:
            playerHandF[card.index] = float(1)

        claimedF = [float(0)] * Card.NUM_CARDS
        for cards in playerClaimedCards:
            for card in cards:
                claimedF[card.index] = float(1)

        """bidIndicators = [0] * Game.NUM_PLAYERS * 14
        for i in range (0, Game.NUM_PLAYERS):
            bidIndicators[i * 14 + playerBids[i]] = 1
        """
        pileF = [float(0)] * Card.NUM_CARDS
        for i, card in enumerate(pile):
            pileF[card.index] = float(i + 1)

        tricksF = [float(len(c) // 4) for c in playerClaimedCards] # should divide evenly
        return playerHandF + claimedF + playerBids +  pileF  + tricksF + playerBags + actions
        '''
        cards = [float(0)]*52
        for card in playerHand:
            cards[card.index] = 1
        for card in pile:
            cards[card.index] = 5
        for i in range(len(playerClaimedCards)):
            claimed = playerClaimedCards[i]
            for card in claimed:
                cards[card.index] = i + 2
        return cards + [float(b) for b in playerBids] + [float(action.index if not action is None else -1)]# 52 + 4 + 1
        '''
        

    @staticmethod
    def genActionParams(state):
        playerHand, playerClaimedCards, playerBids, playerBags, pile = state

        broke = any(p.getSuit() == Card.SPADES_SUIT for p in pile)
        return playerHand, pile, broke
