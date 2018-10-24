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
    