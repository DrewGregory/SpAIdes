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

    def playCard(self, actions, pile=None):
        raise NotImplementedError("playCard not implemented")

    def calculateScore(self):
        tricks = len(self.claimed) / 4
        if tricks < self.bid:
            self.score += 10 * (tricks - self.bid)
        else:
            self.score += 10 * self.bid + (tricks - self.bid)
            self.bags += (tricks - self.bid)
            while self.bags >= 10:
                self.score -= 100
                self.bags -= 10
        print(self.name + " score: " + str(self.score))
        # Reset round state
        self.hand = []
        self.claimed = set()
        self.bid = 0
    
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
        # Remove card from hand
        self.hand = filter(lambda x: x.index != actions[chosenIndex].index, self.hand)
        print(" ")
        return actions[chosenIndex]

        