from player import Human
from card import Card
from random import shuffle
from utils import genActions
deck = ([Card(i) for i in range(0, 52)])
shuffle(deck)
players = [Human([]) for i in range(0, 4)]

"""
Rounds go as follows:
1) Rotate positions of players
2) Deal cards to players
3) Ask each player for bid
4) Play hands until out of cards
"""

while max((499,) + tuple([player.score for player in players])) == 499:
    shuffle(deck)
    players = [-1] + players[:-1]
    for i in range(0, len(players)):
        player[i].hand = deck[i * 13 : (i + 1) * 13]
        player.declareBid()
    playerCursor = 1 # left of dealer
    while sum([len(p.hand) for p in players]) > 0:
        pile = []
        for i in range(0, len(players)):
            player = players[(playerCursor + i) % len(players)]
            actions = genActions(player.hand, pile)
            pile.append(player.playCard(actions))
        winIndex = (determineWinCardIndex(pile) - 1) % 4
        players[winIndex].claimed.add(tuple(pile))

    
