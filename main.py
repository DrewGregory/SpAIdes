from moderator import Moderator
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='Play a game of spades')
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument("--idiot", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # sets how many games to play
    # each game is Moderator.NUM_GAMES rounds long, with a fixed objective Q-learning model
    num_games = 10**0
    for i in range(num_games):
        print("\n\n\n\nITERATION ", str(i)+"/"+str(num_games) , " STARTING \n\n\n")
        m = Moderator(parseArgs())
        m.playGame()
