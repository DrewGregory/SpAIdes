# coding=utf-8
class Card:

    # Card Constants
    NUM_SUITS = 4
    NUM_CARDS = 52
    NUM_PER_SUIT = 13
    SPADES_SUIT = 0
    def __init__(self, index):
        """ 
        Indices: 0 - 12 -> Ace thru King of Spades
        0 -> 2 of Spades
        1 -> 3 of Spades
        ...
        10 -> Queen of Spades
        11 -> King of Spades
        12 -> Ace of Spades
        13 - 25 -> Clubs
        26 - 38 -> Hearts
        39 - 51 -> Diamonds
        """
        self.index = index

    def getValue(self):
        # Face value... since 2 is index 0
        return (self.index % 13) + 2

    def getSuit(self):
        return self.index // Card.NUM_PER_SUIT

    def __str__(self):
        """
        Print out card in human-readable format
        """
        d = {9: "J", 10: "Q", 11: "K", 12: "A"}
        s = ["♠", "♣", "♥", "♦"]
        num_val = self.index % 13
        num_val = str(num_val + 2) if num_val < 9 else d[num_val]
        suit = s[self.index // 13]
        return num_val+suit

    __repr__  = __str__