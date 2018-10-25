# coding=utf-8
class Card:

    def __init__(self, index):
        """ 
        Indices: 0 - 12 -> Ace thru King of Spades
        13 - 25 -> Clubs
        26 - 38 -> Hearts
        39 - 51 -> Diamonds
        """
        self.index = index


    def __str__(self):
        """
        Print out card in human-readable format
        """
        num = self.index % 13
        if num == 9:
            num = "J"
        elif num == 10:
            num = "Q"
        elif num == 11:
            num = "K"
        elif num == 12:
            num = "A"
        else:
            num = str(num + 2)
        suit = self.index // 13
        if suit == 0:
            suit = "♠"
        elif suit == 1:
            suit = "♣"
        elif suit == 2:
            suit = "♥"
        else:
            suit = "♦"
        return num+suit
        
    __repr__  = __str__