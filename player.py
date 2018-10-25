import random
from card import Card


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

    def __init__(self, hand, name=""):
        # TODO
        self.hand = hand
        self.claimed = set()
        self.sandbags = 0
        self.score = 0
        self.bid = 0
        self.name = name
        self.bags = 0

    def declareBid(self):
        raise NotImplementedError("declareBid not implemented")

    def removeCard(self, card):
        if card in self.hand:
            self.hand = [x for x in self.hand if x.index != card.index]
            return True
        else:
            return False

    def playCard(self, actions, pile=None):
        raise NotImplementedError("playCard not implemented")

    def regressionScore(self, tricks):
        subScore = 0
        if tricks < self.bid:
            subScore = 10 * (tricks - self.bid)
        else:
            subScore = 10 * self.bid + (tricks - self.bid)
            self.bags += (tricks - self.bid)
            while self.bags >= 10:
                subScore -= 100
                self.bags -= 10
        return subScore

    def calculateScore(self, scoreFunction=regressionScore):
        tricks = len(self.claimed) / 4
        subScore = scoreFunction(self, tricks)
        # Reset round state
        self.hand = []
        self.claimed = set()
        self.bid = 0
        self.score += subScore
        print(self.name + " score: " + str(self.score))
        return subScore

class Human(Player):

    def declareBid(self):
        print(self.name + "\'s turn:")
        print(self.hand)
        self.bid = int(input("What is your bid?"))
        return self.bid

    def playCard(self, actions, pile):
        print(self.name + "\'s turn:")
        print("Pile: "+ str(pile))
        print("Possible Cards: " + str(actions))
        # TODO: print indices underneath card list
        print(self.claimed)
        print("Tricks so far: %d \t Bid: %d" % (len(self.claimed)/4, self.bid))
        chosenIndex = int(input("Which card do you want to play (Index)?"))
        self.removeCard(actions[chosenIndex])
        print(" ")
        return actions[chosenIndex]

class Baseline(Player):
    def declareBid(self):
        # Let number of cards above jack be our bid #.
        for card in self.hand:
            if card.index % 13 > 10:
                self.bid += 1
        return self.bid

    def playCard(self, actions, pile):
        print(str(self.hand) + " actions: " + str(actions))
        card = None
        if len(pile) == 0:
            card = random.choice(actions)
        else:
            # find highest card
            chosenIndex = 0
            for i in range(0, len(actions)):
                if actions[i].index > actions[chosenIndex].index:
                    chosenIndex = i
            card = actions[chosenIndex]
        self.removeCard(card)
        return card


class Idiot(Player):
    def declareBid(self):
        return random.choice([i for i in range(Card.NUM_CARDS)])
    
    def playCard(self, actions, pile):
        card = random.choice(actions)
        self.removeCard(card)
        return card