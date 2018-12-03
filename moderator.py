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

    TEST = 1
    NUM_GAMES = int(1e8)
    LOGGING = True
    VERBOSE_PLAY = True

    
    def __init__(self, args):
        self.game = Game(args)
       # self.writer = SummaryWriter()


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
        # while max((Game.END_SCORE,) + tuple([player.score for player in self.players])) == Game.END_SCORE:
        for _ in range(Moderator.NUM_GAMES):
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
                    card = player.playCard(playerState, actions ,self.game.pile)
                    if Moderator.VERBOSE_PLAY:
                        print(player.name + " played " + str(card))
                    self.game.pile.append(card)
                # winnerIndex now represents the *index of the winning player*
                winnerIndex = (self.playerCursor + determineWinCardIndex(self.game.pile)) % self.game.NUM_PLAYERS
                if any (card.index < Card.NUM_PER_SUIT for card in self.game.pile):
                    brokeSpades = True
                # update winning player's claimed cards
                if Moderator.VERBOSE_PLAY:
                    print(self.game.players[winnerIndex].name + " won the trick")
                    print()
                self.game.players[winnerIndex].claimed.update(self.game.pile)

                # incorporate feedback based on trick winning
                for i in range(self.game.NUM_PLAYERS):
                    playerIndex = (self.playerCursor+i) % self.game.NUM_PLAYERS
                    player = self.game.players[playerIndex]
                    playerState = self.game.getPlayerGameState(player, playerIndex)
                    
                    reward = 0#( min(player.tricksWon(self.game.NUM_PLAYERS) - player.bid, 0) ) /(13 - numRotations + 1) # default reward for no tricks won
                   
                    # Give reward for winning, but penalize if it's overbidding
                    if playerIndex == winnerIndex:
                        if player.tricksWon(self.game.NUM_PLAYERS) > player.bid:
                            reward = -0.5
                        else:
                            reward = 1
                        reward = 1
                    '''
                    if numRotations ==13:
                        reward = player.calculateScore(reset=False, scoreFunction=lambda s,t:player.simpleScore(t))
                    '''
                    player.incorporateFeedback(playerState, reward)
                    
                self.playerCursor = winnerIndex
                self.game.pile = []


            ### Logging ####
            if Moderator.LOGGING and not Moderator.VERBOSE_PLAY:
                mt = [ p for p in self.game.players if p.name=="Model Test"][0]
                utils.TWriter.add_scalar('data/bid', mt.bid, _)
                logScores = {}
                for p in self.game.players:
                    logScores[p.name] = p.score
                utils.TWriter.add_scalars('data/scores'+str(_%Moderator.TEST), logScores, _)
                
            #### END Logging ####
            for player in self.game.players:
                print(player.name + ": " + str(player.calculateScore()))
            """
            otherScores = [ ( x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "AI" in x.name]
            bestScore = mean([x[2] for x in otherScores])
            testScore = ([ (x.tricksWon(self.game.NUM_PLAYERS), x.bid, x.calculateScore()) for x in self.game.players if "Test" in x.name])[0]
            avgScoreDifferential.append(testScore[2] - bestScore)
            """
            self.roundCursor = self.roundCursor + 1 % self.game.NUM_PLAYERS
            """
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
            """
        print(mean(avgScoreDifferential))