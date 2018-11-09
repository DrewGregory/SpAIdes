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
            # Initialize round
            shuffle(self.game.deck)
            for i in range(self.game.NUM_PLAYERS):
                self.game.players[i].hand = self.game.deck[i * 13: (i + 1) * 13]
                playerState = self.game.getPlayerGameState(self.game.players[i], i)
                self.game.players[i].declareBid(playerState)
            self.playerCursor = (self.roundCursor + 1) % self.game.NUM_PLAYERS  # left of dealer
            brokeSpades = False

            # Play round through to completion
            while sum([len(p.hand) for p in self.game.players]) > 0:
                self.game.pile = []
                for i in range(self.game.NUM_PLAYERS):
                    player = self.game.players[(self.playerCursor + i) % self.game.NUM_PLAYERS]
                    actions = genActions(player.hand, self.game.pile, brokeSpades)
                    playerState = self.game.getPlayerGameState(player, self.playerCursor + i)
                    self.game.pile.append(player.playCard(playerState, actions, self.game.pile))
                # playerCursor now represents the *index of the winning player*
                self.playerCursor = (self.playerCursor +
                                     determineWinCardIndex(self.game.pile)) % self.game.NUM_PLAYERS
                if any(card.index < Card.NUM_PER_SUIT for card in self.game.pile):
                    brokeSpades = True
                self.game.players[self.playerCursor].claimed.update(self.game.pile)
            # Calculate scores
            print("SCORES: \n --------")
            bestScore = mean([x.calculateScore() for x in self.game.players if "AI" in x.name])
            testScore = ([x.calculateScore()
                          for x in self.game.players if "Oracle" in x.name or "Test" in x.name])[0]
            print(testScore - bestScore)
            avgScoreDifferential.append(testScore - bestScore)
            self.roundCursor = self.roundCursor + 1 % self.game.NUM_PLAYERS

            # Incorporate Feedback From Game Score
            for i in range(self.game.NUM_PLAYERS):
                player = self.game.players[(self.playerCursor + i) % self.game.NUM_PLAYERS]
                playerState = self.game.getPlayerGameState(player, self.playerCursor + i)
                player.incorporateFeedback(player.calculateScore(), playerState)

        print(mean(avgScoreDifferential))
