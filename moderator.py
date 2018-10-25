from card import Card
from random import shuffle
from game import Game
from statistics import mean
from utils import genActions, determineWinCardIndex

class Moderator:

    def __init__(self, args):
        self.game = Game(args)

    def playGame(self):
        """
        Rounds go as follows:
        1) Rotate positions of players
        2) Deal cards to players
        3) Ask each player for bid
        4) Play hands until out of cards
        """
        avgScoreDifferential = []

        self.roundCursor = 0
        # while max((Game.END_SCORE,) + tuple([player.score for player in self.players])) == Game.END_SCORE:
        for _ in range(5000):
            shuffle(self.game.deck)
            for i in range(0, len(self.game.players)):
                self.game.players[i].hand = self.game.deck[i * 13 : (i + 1) * 13]
                playerState = self.game.getPlayerGameState(self.game.players[i], i)
                self.game.players[i].declareBid(playerState)
            self.playerCursor = (self.roundCursor + 1) % 4 # left of dealer
            brokeSpades = False
            while sum([len(p.hand) for p in self.game.players]) > 0:
                self.game.pile = []
                for i in range(0, len(self.game.players)):
                    player = self.game.players[(self.playerCursor + i) % len(self.game.players)]
                    actions = genActions(player.hand, self.game.pile, brokeSpades)
                    playerState = self.game.getPlayerGameState(player, self.playerCursor + i)
                    self.game.pile.append(player.playCard(playerState, actions, self.game.pile))
                winIndex = (self.playerCursor + determineWinCardIndex(self.game.pile)) % 4
                for card in self.game.pile:
                    if card.index // 13 == 0:
                        brokeSpades = True
                    self.game.players[winIndex].claimed.add(card)
                self.playerCursor = (self.playerCursor + winIndex - 1) % 4
            # Calculate scores
            print("SCORES:")
            print("--------")
            bestScore = mean([x.calculateScore() for x in self.game.players if "AI" in x.name])
            testScore = ([x.calculateScore() for x in self.game.players if "Oracle" in x.name or "Test" in x.name])[0]
            print(testScore - bestScore)
            avgScoreDifferential.append(testScore - bestScore)
            self.roundCursor = self.roundCursor + 1 % 4
        print(mean(avgScoreDifferential))



