from player import Human
from card import Card
from random import shuffle
from utils import genActions, determineWinCardIndex
deck = ([Card(i) for i in range(0, 52)])
shuffle(deck)
players = [Human([], "Human " + str(i)) for i in range(0, 4)]

"""
Rounds go as follows:
1) Rotate positions of players
2) Deal cards to players
3) Ask each player for bid
4) Play hands until out of cards
"""

while max((499,) + tuple([player.score for player in players])) == 499:
    roundCursor = 0
    shuffle(deck)
    for i in range(0, len(players)):
        players[i].hand = deck[i * 13 : (i + 1) * 13]
        players[i].declareBid()
    playerCursor = (roundCursor + 1) % 4 # left of dealer
    while sum([len(p.hand) for p in players]) > 0:
        pile = []
        for i in range(0, len(players)):
            player = players[(playerCursor + i) % len(players)]
            actions = genActions(player.hand, pile)
            pile.append(player.playCard(actions, pile))
        winIndex = (playerCursor + determineWinCardIndex(pile)) % 4
        for card in pile:
            players[winIndex].claimed.add(card)
        playerCursor = (playerCursor + winIndex) % 4
        

