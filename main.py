from moderator import Moderator
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description='Play a game of spades')
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    m = Moderator(parseArgs())
    m.playGame()
