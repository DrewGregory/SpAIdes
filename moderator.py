from player import Human, Baseline, Idiot
from card import Card
from random import shuffle
from utils import genActions, determineWinCardIndex
from statistics import mean


deck = ([Card(i) for i in range(0, 52)])
shuffle(deck)
players = [Idiot([], "Idiot " + str(i + 1))  for i in range(0, 3)]
players.append(Baseline([], "Baseline"))

"""
Rounds go as follows:
1) Rotate positions of players
2) Deal cards to players
3) Ask each player for bid
4) Play hands until out of cards
"""

avgScoreDifferential = []

while max((499,) + tuple([player.score for player in players])) == 499:
    roundCursor = 0
    shuffle(deck)
    for i in range(0, len(players)):
        players[i].hand = deck[i * 13 : (i + 1) * 13]
        players[i].declareBid()
    playerCursor = (roundCursor + 1) % 4 # left of dealer
    brokeSpades = False
    while sum([len(p.hand) for p in players]) > 0:
        pile = []
        for i in range(0, len(players)):
            player = players[(playerCursor + i) % len(players)]
            actions = genActions(player.hand, pile, brokeSpades)
            pile.append(player.playCard(actions, pile))
        winIndex = (playerCursor + determineWinCardIndex(pile)) % 4
        for card in pile:
            if card.index / 13 == 0:
                brokeSpades = True
            players[winIndex].claimed.add(card)
        playerCursor = (playerCursor + winIndex - 1) % 4
    # Calculate scores
    print("SCORES:")
    print("--------")
    avgScoreDifferential = mean()
    for player in players:
        player.calculateScore()

