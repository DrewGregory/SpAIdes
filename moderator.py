import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import random
from random import shuffle
from statistics import mean


from card import Card
from game import Game
from utils import genActions, determineWinCardIndex
import utils


class Moderator:

    TEST = 1 # when random.seed uncommented, sets how many possible game configs
    NUM_GAMES = int(1e8) # number of rounds to simulate: with fixed Q-learning model objectives
    LOGGING = True

    
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
        model_total = []
        self.roundCursor = 0
        for _ in range(Moderator.NUM_GAMES):

            # used for diagnosing learning issues, restricts to small number of possible configs
            #random.seed(_ % Moderator.TEST) 
            self.roundCursor = 0
            self.game.deck = [Card(i) for i in range(Card.NUM_CARDS)]
            # Initialize round, deals cards and bids
            
            shuffle(self.game.deck)
            if _==0:
                print(self.game.deck)
            if _%100 == 0:
                print("BIDS: \n ------")

            for i in range(self.game.NUM_PLAYERS):
                self.game.players[i].hand = self.game.deck[i * 13 : (i + 1) * 13]
                playerState = self.game.getPlayerGameState(self.game.players[i], i)
                self.game.players[i].declareBid(playerState)

                # prints bidding along with scores down below for shell logging
                if _%100==0:
                    print(self.game.players[i].name , "bid " + str(self.game.players[i].bid))

            self.playerCursor = (self.roundCursor + 1) % self.game.NUM_PLAYERS # left of dealer
            brokeSpades = False
            
            numRotations = 0
            # Play round through to completion
            while sum([len(p.hand) for p in self.game.players]) > 0:
                numRotations += 1
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
                    
                    reward = 0
                    # Give reward for winning, but penalize if it's overbidding
                    if playerIndex == winnerIndex:
                        reward = 1

                    player.incorporateFeedback(playerState, reward)
                    
                self.playerCursor = winnerIndex
                self.game.pile = []


            ### Logging ####
            if Moderator.LOGGING:
                mt = [ p for p in self.game.players if p.name=="Model Test"][0]
                utils.TWriter.add_scalar('data/bid', mt.bid, _)
                logScores = {}
                for p in self.game.players:
                    logScores[p.name] = p.score
                utils.TWriter.add_scalars('data/scores'+str(_%Moderator.TEST), logScores, _)
                
            #### END Logging ####
            otherScores = [ ( x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "AI" in x.name]
            bestScore = mean([x[2] for x in otherScores])
            testScore = ([ (x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "Test" in x.name])[0]
            avgScoreDifferential.append(testScore[2] - bestScore)
            self.roundCursor = self.roundCursor + 1 % self.game.NUM_PLAYERS
            
            # Shell logging output: for diagnosing scoring issues
            if _ % 100 == 0:
                if Moderator.LOGGING:
                    mt.save()
                # Calculate scores
                print("SCORES: \n --------  " + str(_/Moderator.NUM_GAMES))
                for player in self.game.players:
                    print(player.name + ":  " + str(player.score))
                for score  in otherScores:
                    print("Baseline score: " + str(score[2]), "Bid:", score[1], "Tricks Won:", score[0] )
                    print("TOTAL: " + str())
                if Moderator.LOGGING:
                    model_total.append(mt.score)
                print("Model Score: " + str(testScore[2]), "Bid: ", testScore[1], "Tricks Won: ", testScore[0] )
        print(mean(avgScoreDifferential))