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
        self.players = [Baseline([], "AI Baseline" + str(i + 1))
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
        if not "Oracle" in player.name:
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
        @return [float] feature vector as one concatenated list
        Features:
        1. actions: [float], indicator for card played
        2. playerHandF: [float], indicators for player's hand
        3. claimedF: [float], indicators for all players' claimed
        4. playerBids: [float], bids/player
        5. pileF: [float], indicators for all cards in pile
        6. tricksF: [float], tricks/player
        7. playerBags: [float], bags/player
        NOTE: This method should not be called using the oracle, has no
        support for multiple player hands
        '''
        playerHand, playerClaimedCards, playerBids, playerBags, pile = state
        
        actions = [0] * Card.NUM_CARDS
        if action:
            actions[action.index] = float(1)
        
        playerHandF = [float(0)] * Card.NUM_CARDS


        for card in playerHand:
            playerHandF[card.index] = float(1)

        claimedF = [float(0)] * Card.NUM_CARDS
        for cards in playerClaimedCards:
            for card in cards:
                claimedF[card.index] = float(1)

        # value corresponds to index of card in pile
        pileF = [float(0)] * Card.NUM_CARDS
        for i, card in enumerate(pile):
            pileF[card.index] = float(i + 1)

        tricksF = [float(len(c) // 4) for c in playerClaimedCards] # should divide evenly

        playerBids = [0, 0, 0, 0,] # for training the Trick Winner
        return playerHandF + claimedF + playerBids +  pileF + tricksF + playerBags + actions

    @staticmethod
    def genActionParams(state):
        playerHand, playerClaimedCards, playerBids, playerBags, pile = state
        broke = any(p.getSuit() == Card.SPADES_SUIT for p in pile)
        return playerHand, pile, broke
