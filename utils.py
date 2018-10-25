def genActions(hand, pile, brokeSpades):
    """
    Given a player's hand and the pile of cards in the center, generate a list of possible cards the player can
    play.
    If user has cards of  that suit, only show cards of that suit. Otherwise, give anything.
    """
    actions = []

    if len(pile) == 0:
        allCardsSpades = True
        for card in hand:
            if card.index / 13 != 0:
                allCardsSpades = False
        if brokeSpades or allCardsSpades:
            return hand
        else:
            return filter(lambda x: x.index / 13 > 0, hand)
    bottomSuit = pile[0].index / 13
    for card in hand:
        suit = card.index / 13
        if suit == bottomSuit:
            actions.append(card)
    if len(actions) == 0:
        return hand
    return actions

def determineWinCardIndex(pile):
    bestCard = (pile[0], 0)
    for i in range(1, len(pile)):
        bestCardSuit = bestCard[0].index/13
        cardSuit = pile[i].index/13
        if (bestCardSuit == cardSuit and pile[i].index > bestCard[0].index) or \
            (bestCardSuit != 0 and cardSuit == 0):
            bestCard = (pile[i], i)
    print("Out of " + str(pile) + " Best card: " + str(bestCard[0]))
    return bestCard[1]