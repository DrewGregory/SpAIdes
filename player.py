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
    def __init__(self, hand):
        # TODO
        self.hand = hand
        self.claimed = set()
        self.sandbags = 0
        self.score = 0
        self.bid = 0

    def declareBid(self):
        raise NotImplementedError("declareBid not implemented")

    def playCard(self, pile):
        raise NotImplementedError("playCard not implemented")

    
class Human(Player):

    def declareBid(self):
        print(self.hand)
        self.bid = int(input("What is your bid?"))
        return self.bid

    def playCard(self, actions):
        print(actions)
        # TODO: print indices underneath card list
        print("Tricks so far: %d \t Bid: %d" % (len(self.claimed)/4, self.bid))
        chosenIndex = int(input("Which card do you want to play?"))
        return actions[chosenIndex]

        