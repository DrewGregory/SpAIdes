from card import Card
from random import shuffle
from player import Human, Baseline, Idiot, Oracle

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
        self.players = [Baseline([], "AI Baseline " + str(i + 1))  for i in range(0, Game.NUM_PLAYERS - 1)]
        if args.human:
            self.players.append(Human([], "Human"))
        elif args.oracle:
            self.players.append(Oracle([], "Oracle"))
        else: 
            self.players.append(Baseline([], "Test"))
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
        if False and not player.name == "Oracle": # @GriffinKardos...when you make your oracle class change this
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
        return [playerHands, playerClaimedCards, playerBids, playerBags, self.pile]

    @staticmethod
    def stateFeatureExtractor(state):
        playerHands, playerClaimedCards, playerBids, playerBags, pile = state

        # Binary indicators for each card for each hand/set of cards
        def vectorizeHand(hand):
            indicators = [0] * Card.NUM_CARDS
            for card in hand:
                indicators[card] = 1
        
        vectors = []
        for hand in playerHands:
            vectors.append(vectorizedHand(hand))
        for claim in playerClaimedCards:
            vectors.append(vectorizeHand(claim))
        
        vectors.append(vectorizeHand(pile))
        vectors.append(playerBids) # just keep as numbers
        vectors.append(playerBags)
        return np.array(vectors, dtype=uint8)

