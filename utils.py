

from tensorboardX import SummaryWriter

#TWriter = SummaryWriter()


def genActions(hand, pile, brokeSpades):
    """
    @param list hand: player's hand
    @param list pile: pile of cards in center
    @param bool brokeSpades
    @return list: all possible cards the player can play. 
        If user has cards of leading suit, only return cards of same suit.
        Otherwise, return entire hand.
    """
    actions = []

    if len(pile) == 0:
        allCardsSpades = True
        for card in hand:
            if card.index >= 13:
                allCardsSpades = False
        if brokeSpades or allCardsSpades:
            return hand
        else:
            return [x for x in hand if x.index >= 13]
    bottomSuit = pile[0].index // 13
    for card in hand:
        suit = card.index // 13
        if suit == bottomSuit:
            actions.append(card)
    if len(actions) == 0:
        return hand
    return actions


def determineWinCardIndex(pile):
    """
    @param list pile: pile of cards in center
    @return int: index of winning card -
        highest index of leading suit, or if there are spades, highest spades index
    """
    bestCard = (pile[0], 0)
    for i in range(1, len(pile)):
        bestCardSuit = bestCard[0].index // 13
        cardSuit = pile[i].index // 13
        if (bestCardSuit == cardSuit and pile[i].index > bestCard[0].index) or \
                (bestCardSuit != 0 and cardSuit == 0):
            bestCard = (pile[i], i)
    #print("Out of " + str(pile) + " Best card: " + str(bestCard[0]))
    return bestCard[1]