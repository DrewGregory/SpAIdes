## How to Run
` python3 main.py --human` to play as a human against 3 baseline characters

`python3 main.py --oracle` to have an "oracle" play against 3 baseline characters

## Main: Driver class
The main class is a simple driver class for running and playing games. It handles argument configurations, and allows for specifying the number of games to play (via a for loop) which is useful for training our DQN, which updates its fixed objective functions after every game.


## Moderator: Performs logic on Game

The moderator class encompasses the majority of the game simulation code. It handles all of the dealing, iterates through the players to have them play each hand, and even handles the bidding process after shuffling and dealing. Majority of the game rules are implemented within the Moderator class via the simulation code.

- playGame --simulates the games, with a variable number of rounds

## Game: Stateful class

The purpose of the Game class is to provide a high level structure according to the rules of Spades. The responsibilities of the Game class include: establishing which players are to play the Game, keeping track of the game overall game state, giving the right game state information personalized for each player, and vectorizing a player games state for use with our learning models.

- getPlayerGameState
- stateFeatureExtractor
- getOracleGameState

## Utils: Helpful Functions
Helpful functions such as generating actions from a given state, making probabilistically weighted decisions, and determining the index of the winner from a given hand.

-genActions: generating all of the possible actions a player can take on a given game state
-determineWinCardWindex --helps determine who won each hand to help score feedbacks
-weightedChoice: deprecated function used for making weighted random choices used for model exploration policies.

## Player: base OOP class for game players

The player class is an abstract class that implements functionality for a lot of general methods required of any player extensions. This includes keeping track of the current player information such as score, current bid, current claimed cards, current tricks won, current hand, an identification name, and a play history. In addition, methdos such as __playCard__, __calculateScore__, __resetRound__, __incorporateFeedback__ and __declareBid__ are implemented to provide basic functionality. Subclasses tyically override these implementations to reflect the strategy they are meant to employ: for example __playCard__, __incorporateFeedback__, and __declareBid__ are most prominently modified in our AI agents to use the Q-Learning evaluation function to help make decisions.

### Idiot: low baseline for playing

__playCard__: Chooses from available actions randomly.
__declareBid__: Chooses a random valid bid value.

### Human: Interactive player interface
__declareBid__: Prints dealt hand, and prompts for user input for bid value
__playCard__: Prints current visible game state and prompts user input for card playing value

### Baseline: Our baseline metric player
__declareBid__: Bid the number of cards the player has with value greater than Queen plus the number of Spades cards divided by two. 
__playCard__: Plays the highest valid card they have. High cards are more valuable in the beginning of the game and will maximize the chance that the person wins a trick. This is often the strategy that many amateur players perform.

### Oracle: Our upper bound metric

__declareBid__: Since predicting the number of tricks won perfectly is difficult, we use the same bidding structure as in the Baseline class, but for scoring, we treat the bid value as the number of tricks won to mimic the effect of our Oracle perfectly bidding every time.
__playCard__: Using perfect information, plays the best card available at each hand after determining whether or not it would be able to win with it. Else, it plays its least valuable card possible.

## Card: A simple utility class
- useful for printing in a human-readable format


# Agents.py: Learning agents

Class for storing all of the AI based learning model logic. Contains player subclasses, neural net configuration logic, and training and updating logic as well.

## ModelPlayer:Generic AI Player

This is a generalized AI Player subclass that is meant to work with any type of evaluation function. It is initialized with exploration, discount, featureExtractor, action generator and model parameters. The majority of these follow the same structure as their representations within the CS221 Assignments. Albeit, the __model__ parameter is expected to represent any evaluation function that supports two functions: predict and update. These two functions are used to score state-action pairs and are used to update the model based on the incorporateFeedback function which is called to reflect the rewards for each state-action pair choice. The structure of the action choices and __incorporateFeedback__ method are implemented to accurately reflect the Q-Learning algorithm with a random action policy proportional to the inverse-square root of the number of played iterations. The __declareBid__ method follows a similar pattern (since we use a DQN for the bidding policy) and implements an upper-confidence bound on scoring the best actions as well.

## QModel: Wrapper class for PyTorch Neural Nets

This class wraps around a PyTorch neural net, and implements the __predict__ and __update__ functions that are expected of the __model__ parameter in the ModelPlayer class.

## ModelTest: Wrapper class for DQN
This class subclasses ModelPlayer, and simply initializes the ModelPlayer class with a QModel parameter. The QModel parameter is initialized with a customized neural net configuration. This is the class we used for actual gameplay.

## BidderModel: Wrapper class for QModel

Essentially wrapped the functionality for the QModel class with saving/load configurations to be used for training and playing with a DQN bidding policy.