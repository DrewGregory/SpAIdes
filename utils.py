def genActions(hand, pile):
    """
    Given a player's hand and the pile of cards in the center, generate a list of possible cards the player can
    play.
    """
    actions = []
    bottomSuit = pile[0].index / 13
    for card in hand:
        suit = card.index / 13
        if suit == 0 or suit == bottomSuit:
            actions.append(card)
    return actions

def determineWinCardIndex(pile):
    bestCard = (pile[0], 0)
    for i in range(1, 4):
        bestCardSuit = bestCard.index/13
        cardSuit = pile[i].index/13
        if (bestCardSuit == cardSuit and pile[i].index > bestCard.index) or \
            (cardSuit == 0 and pile[i].index > bestCard.index % 13):
            bestCard = (pile[i], i)
    return bestCard[1]