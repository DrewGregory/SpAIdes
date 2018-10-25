from player import Human, Baseline, Idiot
from card import Card
from random import shuffle
from game import Game
from statistics import mean

class Moderator:

    def __init__(self):
                
        self.deck = ([Card(i) for i in range(Card.NUM_CARDS)])
        shuffle(self.deck)
        self.players = [Idiot([], "Idiot " + str(i + 1))  for i in range(0, Game.NUM_PLAYERS - 1)]
        self.players.append(Baseline([], "Baseline"))


    def playGame(self):

        """
        Rounds go as follows:
        1) Rotate positions of players
        2) Deal cards to players
        3) Ask each player for bid
        4) Play hands until out of cards
        """

        avgScoreDifferential = []

        while max((Game.END_SCORE,) + tuple([player.score for player in self.players])) == Game.END_SCORE:
            roundCursor = 0
            shuffle(self.deck)
            for i in range(0, len(self.players)):
                self.players[i].hand = self.deck[i * 13 : (i + 1) * 13]
                self.players[i].declareBid()
            playerCursor = (roundCursor + 1) % 4 # left of dealer
            brokeSpades = False
            while sum([len(p.hand) for p in self.players]) > 0:
                pile = []
                for i in range(0, len(self.players)):
                    player = self.players[(playerCursor + i) % len(self.players)]
                    actions = Game.genActions(player.hand, pile, brokeSpades)
                    pile.append(player.playCard(actions, pile))
                winIndex = (playerCursor + Game.determineWinCardIndex(pile)) % 4
                for card in pile:
                    if card.index // 13 == 0:
                        brokeSpades = True
                    self.players[winIndex].claimed.add(card)
                playerCursor = (playerCursor + winIndex - 1) % 4
            # Calculate scores
            print("SCORES:")
            print("--------")
            avgIdiotScore = mean([x.calculateScore() for x in self.players if "Idiot" in x.name])
            avgBaselineScore = mean([x.calculateScore() for x in self.players if "Baseline" in x.name])
            avgScoreDifferential.append(avgBaselineScore - avgIdiotScore)
        print(mean(avgScoreDifferential))



