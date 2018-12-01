import random
from card import Card
from utils import genActions, determineWinCardIndex


class Player:
    """
    State:
        - Hand
        - Claimed Cards
        - Score
        - Sandbags

    Behavior: behavior object
        - Bid
        - Play Card
    """

    # SCORING CONSTANTS
    USE_BAGS = True
    BAGGING_COST = 100

    def __init__(self, hand, name=""):
        # TODO
        self.hand = hand
        self.claimed = set()
        self.sandbags = 0
        self.score = 0
        self.bid = 0
        self.name = name
        self.bags = 0
        self.playHistory = []

    def declareBid(self, state):
        raise NotImplementedError("declareBid not implemented")

    def removeCard(self, card):
        if card in self.hand:
            self.hand = [x for x in self.hand if x.index != card.index]
            return True
        else:
            return False

    def playCard(self, state, actions, pile=None):
        raise NotImplementedError("playCard not implemented")

    def regressionScore(self, tricks):
        subScore = 0
        if tricks < self.bid:
            subScore = 10 * (tricks - self.bid)
        else:
            subScore = 10 * self.bid + (tricks - self.bid)
            self.bags += (tricks - self.bid)
            while self.bags >= 10:
                subScore -= self.BAGGING_COST
                self.bags -= 10
        return subScore
    
    def justTricksScore(self, tricks):
        return tricks

    def tricksWon(self, numPlayers=4):
        return len(self.claimed) // numPlayers

    def simpleScore(self, tricks):
        sub = -abs(tricks-self.bid)*10 + min(tricks-self.bid, 0)
        if tricks >= self.bid:
            sub += self.bid*10
        return sub

    def calculateScore(self, reset=True, scoreFunction=simpleScore):
        tricks = len(self.claimed) // 4
        subScore = scoreFunction(self, tricks)
        # Reset round state
        if reset:
            self.resetRound()
        self.score += subScore
        
        return subScore

    def incorporateFeedback(self, newState, reward):
        pass

    def resetRound(self):
        self.hand = []
        self.claimed = set()
        self.bid = 0


class Human(Player):

    def declareBid(self, state):
        print(self.name + "\'s turn:")
        print(self.hand)
        self.bid = int(input("What is your bid? "))
        return self.bid

    def playCard(self, state, actions, pile):
        print(self.name + "\'s turn:")
        print("Pile: " + str(pile))
        print("Possible Cards: " + str(actions))
        print("Claimed cards: " + str(self.claimed))
        print("Tricks so far: %d \t Bid: %d" % (len(self.claimed)/4, self.bid))
        print(state)
        overly_large_index = 1000
        chosenIndex = overly_large_index
        while chosenIndex >= len(actions):
            chosenIndex = int(input("Which card do you want to play (Index)? ")
                              or overly_large_index)
        self.removeCard(actions[chosenIndex])
        print(" ")
        return actions[chosenIndex]


class Baseline(Player):
    def declareBid(self, state):
        # Let number of cards above queen be our bid #.
        for card in self.hand:
            if card.index % 13 >= 11:
                self.bid += 1
            elif card.getSuit() == Card.SPADES_SUIT:
                self.bid += 0.3 # add small amount for low spades
        self.bid = int(min(self.bid , 13))
        return self.bid

    def playCard(self, state, actions, pile):
        card = None
        if len(pile) == 0:
            card = actions[0]#random.choice(actions)
        else:
            # find highest card
            chosenIndex = 0
            for i in range(0, len(actions)):
                if actions[i].index > actions[chosenIndex].index:
                    chosenIndex = i
            card = actions[chosenIndex]
        self.removeCard(card)

        self.playHistory.append((state, card))

        return card


class Idiot(Player):
    def declareBid(self, state):
        choice = random.choice([i for i in range(Card.NUM_CARDS // Game.NUM_PLAYERS + 1)])
        self.bid = choice
        return choice

    def playCard(self, state, actions, pile):
        card = random.choice(actions)
        self.removeCard(card)
        return card


class Oracle(Player):
    def declareBid(self, state):
        # Let number of cards above jack be our bid #.
        for card in self.hand:
            if card.index % 13 >= 10:
                self.bid += 1
        return self.bid

    def canBeat(self, actions, otherActions, suit):
        '''
        Looking at another player's actions, return None if we can't beat them.
        Otherwise return a pair of True, and the lowest card that can beat the other player's
        best play.
        '''
        otherSpades = list(filter(lambda x: x.getSuit() == 0, otherActions))
        otherSuit = list(filter(lambda x: x.getSuit() == suit, otherActions))
        mySpades = list(filter(lambda x: x.getSuit() == 0, actions))
        # mySuit == actions or [] by the rules of the game
        mySuit = list(filter(lambda x: x.getSuit() == suit, actions))
        if len(mySpades) == 0 and len(mySuit) == 0:
            # I have no card that can win
            return None
        if(len(otherSpades) > 0):
            # Opponent can play spades
            if len(mySpades) == 0:
                # I can't play spades
                return None
            oppBest = max(otherSpades, key=lambda x: x.getValue())
            myBest = max(mySpades, key=lambda x: x.getValue())
            if(oppBest.getValue() > myBest.getValue()):
                # opponent has better spade
                return None
            # Return least card that wins
            return min(filter(lambda x: x.getValue() > oppBest.getValue(), mySpades), key=lambda x: x.getValue())
        if len(otherSuit) == 0 and len(mySuit) > 0:
            # Other dude has no cards that can take the trick
            return min(actions, key=lambda x: x.getValue())
        if len(mySpades) > 0:
            # I can play spades and opponent can't
            # Implicitly I can't play in suit
            return min(mySpades, key=lambda x: x.getValue())
        oppBest = max(otherActions, key=lambda x: x.getValue())
        myBest = max(actions, key=lambda x: x.getValue())
        if(oppBest.getValue() > myBest.getValue()):
            # Both of us are in suit, he has higher card
            return None
        return min(filter(lambda x: x.getValue() > oppBest.getValue(), actions), key=lambda x: x.getValue())

    def playCard(self, state, actions, pile):
        if(len(self.claimed)//4 == self.bid):
            return self.playWorst(actions)
        if(len(pile) == 0):
            for i in range(4):
                card = max(filter(lambda x: x.getSuit() == i, actions),
                           default=None, key=lambda x: x.getValue())
                if(card is None):
                    continue
                play = True
                for j in range(1, 4):
                    playerHand = state[0][i]
                    playerActions = genActions(playerHand, [card], True)
                    maxP = max(filter(lambda x: x.getSuit() == i, playerActions),
                               default=None, key=lambda x: x.getValue())
                    if maxP is None:
                        if i > 0 and len([x for x in playerActions if x.getSuit() == 0]):
                            play = False
                            break
                    else:
                        if maxP.getValue() > card.getValue():
                            play = False
                            break
                if play:
                    self.removeCard(card)
                    return card
            return self.playWorst(actions)
        suit = pile[0].getSuit()
        numPlayed = len(pile)
        # bestCard will be the max of the mins -- the least card I can play which will beat everyone
        bestCard = pile[determineWinCardIndex(pile)]
        better = list(filter(lambda x: x.getSuit() == bestCard.getSuit()
                             and x.getValue() > bestCard.getValue(), actions))
        if(len(better) == 0):
            if(bestCard.getSuit() == 0):
                # I have no card that can beat the current spade
                return self.playWorst(actions)
            better = list(filter(lambda x: x.getSuit() == 0, actions))
            if(len(better) == 0):
                # I have no spades or in suit higher cards
                return self.playWorst(actions)
        bestCard = min(better, key=lambda x: x.getValue())
        for i in range(numPlayed + 1, len(state[0])):
            # The players are already in the order that they play
            playerHand = state[0][i]
            playerActions = genActions(playerHand, pile, True)
            nextBest = self.canBeat(actions, playerActions, suit)
            if(nextBest == None):
                return self.playWorst(actions)
            if(nextBest.getSuit() == bestCard.getSuit()):
                if(nextBest.getValue() > bestCard.getValue()):
                    bestCard = nextBest
            elif(nextBest.getSuit() == 0):
                bestCard = nextBest
            # else bestCard is a spade and the nextBest is not

        self.removeCard(bestCard)
        return bestCard

    def playWorst(self, actions):
        '''Plays and removes the worst card from possible actions. Returns said card'''
        worst = min(filter(lambda x: x.getSuit() != 0, actions),
                    default=None, key=lambda x: x.getValue())
        if(worst is not None):
            self.removeCard(worst)
            return worst
        worst = min(actions, key=lambda x: x.getValue())
        self.removeCard(worst)
        return worst
