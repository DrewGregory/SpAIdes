import numpy as np

from card import Card
from random import shuffle
from game import Game
from statistics import mean
from utils import genActions, determineWinCardIndex

class Moderator:

    NUM_GAMES = 15000
    
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
        for _ in range(Moderator.NUM_GAMES):
            # Initialize round, deals cards and bids
            
            shuffle(self.game.deck)
            if _%100 == 0:
                print("BIDS: \n ------")

            for i in range(self.game.NUM_PLAYERS):
                self.game.players[i].hand = self.game.deck[i * 13 : (i + 1) * 13]
                playerState = self.game.getPlayerGameState(self.game.players[i], i)
                self.game.players[i].declareBid(playerState)

                if _%100==0:
                    print(self.game.players[i].name , "bid " + str(self.game.players[i].bid))

            self.playerCursor = (self.roundCursor + 1) % self.game.NUM_PLAYERS # left of dealer
            brokeSpades = False

            # Play round through to completion
            while sum([len(p.hand) for p in self.game.players]) > 0:
                for i in range(self.game.NUM_PLAYERS):
                    player = self.game.players[(self.playerCursor + i) % self.game.NUM_PLAYERS]
                    actions = genActions(player.hand, self.game.pile, brokeSpades)
                    playerState = self.game.getPlayerGameState(player, (self.playerCursor + i) % self.game.NUM_PLAYERS)
                    self.game.pile.append(player.playCard(playerState, actions ,self.game.pile))
                # winnerIndex now represents the *index of the winning player*
                winnerIndex = (self.playerCursor + determineWinCardIndex(self.game.pile)) % self.game.NUM_PLAYERS
                if any (card.index < Card.NUM_PER_SUIT for card in self.game.pile):
                    brokeSpades = True
                # update winning player's claimed cards
                self.game.players[winnerIndex].claimed.update(self.game.pile)

                # incorporate feedback based on trick winning
                for i in range(self.game.NUM_PLAYERS):
                    playerIndex = (self.playerCursor+i) % self.game.NUM_PLAYERS
                    player = self.game.players[playerIndex]
                    playerState = self.game.getPlayerGameState(player, playerIndex)
                    
                    reward = -1  # default reward for no tricks won

                    # Give reward for winning, but penalize if it's overbidding
                    if playerIndex == winnerIndex:
                        if player.tricksWon(self.game.NUM_PLAYERS) > player.bid:
                            reward = -1.1
                        else:
                            reward = 1
                    player.incorporateFeedback(playerState, reward)

                self.game.pile = []
            otherScores = [ ( x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "AI" in x.name]
            bestScore = mean([x[0] for x in otherScores])
            testScore = ([ (x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "Test" in x.name])[0]
            
            avgScoreDifferential.append(testScore[0] - bestScore)
            self.roundCursor = self.roundCursor + 1 % self.game.NUM_PLAYERS
            if _ % 100 == 0:
                # Calculate scores
                print("SCORES: \n --------")
                for score  in otherScores:
                    print("Baseline score: " + str(score[2]), "Bid:", score[1], "Tricks Won:", score[2] )
                print("Model Score: " + str(testScore[2]), "Bid: ", testScore[1], "Tricks Won: ", testScore[2] )
                print(testScore[0] - bestScore)
                


        print(mean(avgScoreDifferential))