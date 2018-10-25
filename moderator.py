from player import Human, Baseline, Idiot
from card import Card
from random import shuffle
from game import Game
from statistics import mean

class Moderator:

    def __init__(self, args):
        self.deck = ([Card(i) for i in range(Card.NUM_CARDS)])
        shuffle(self.deck)
        self.players = [Idiot([], "Idiot " + str(i + 1))  for i in range(0, Game.NUM_PLAYERS - 1)]
        if args.human:
            self.players.append(Human([], "Human"))
        elif args.oracle:
            self.players.append(Oracle([], "Oracle"))
        else: 
            self.players.append(Baseline([], "Baseline"))
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


    def playGame(self):
        """
        Rounds go as follows:
        1) Rotate positions of players
        2) Deal cards to players
        3) Ask each player for bid
        4) Play hands until out of cards
        """
        avgScoreDifferential = []

        while max((Game.END_SCORE,) + tuple([player.score for player in self.players])) == Game.END_SCORE:
            self.roundCursor = 0
            shuffle(self.deck)
            for i in range(0, len(self.players)):
                self.players[i].hand = self.deck[i * 13 : (i + 1) * 13]
                playerState = self.getPlayerGameState(self.players[i], i)
                self.players[i].declareBid(playerState)
            self.playerCursor = (self.roundCursor + 1) % 4 # left of dealer
            brokeSpades = False
            while sum([len(p.hand) for p in self.players]) > 0:
                self.pile = []
                for i in range(0, len(self.players)):
                    player = self.players[(self.playerCursor + i) % len(self.players)]
                    actions = Game.genActions(player.hand, self.pile, brokeSpades)
                    playerState = self.getPlayerGameState(player, self.playerCursor + i)
                    self.pile.append(player.playCard(playerState, actions, self.pile))
                winIndex = (self.playerCursor + Game.determineWinCardIndex(self.pile)) % 4
                for card in self.pile:
                    if card.index // 13 == 0:
                        brokeSpades = True
                    self.players[winIndex].claimed.add(card)
                self.playerCursor = (self.playerCursor + winIndex - 1) % 4
            # Calculate scores
            print("SCORES:")
            print("--------")
            avgIdiotScore = mean([x.calculateScore() for x in self.players if "Idiot" in x.name])
            avgBaselineScore = mean([x.calculateScore() for x in self.players if "Baseline" in x.name])
            avgScoreDifferential.append(avgBaselineScore - avgIdiotScore)
            break
        print(mean(avgScoreDifferential))



