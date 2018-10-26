from card import Card
from random import shuffle
from player import Human, Baseline, Idiot, Oracle
class Game:

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
        self.pile = []


    def getPlayerGameState(self, player, playerCursor):
        oracleGameState = self.getOracleGameState(playerCursor)
        # Remove other player hands...
        # replace with just our player's hand
        if False and not player.name == "Oracle": # @GriffinKardos...when you make your oracle class change this
            oracleGameState[0] = player.hand
        return oracleGameState

    def getOracleGameState(self, playerCursor):
        """
        Get interesting state related to round.
        For the three rays, the 0 index is yourself, each successive element follows
        the ordering around the table clockwise.
        """
        playerHands = []
        playerClaimedCards = []
        playerBids = []
        for i in range(0, len(self.players)):
            player = self.players[(playerCursor + i) % 4]
            playerHands.append(player.hand)
            playerClaimedCards.append(player.claimed)
            playerBids.append(player.bid)
        return [playerHands, playerClaimedCards, playerBids, self.pile]
