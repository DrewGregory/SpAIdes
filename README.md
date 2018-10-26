## How to Run
` python3 main.py --human` to play as a human against 3 baseline characters
`python3 main.py --oracle` to have an "oracle" play against 3 baseline characters
## Moderator: Performs logic on Game
- playGame
- determineWinCardWindex
- genActions
## Game: Stateful class
- getPlayerGameState
- getOracleGameState
## Player
- Idiot
- Human
- Baseline
- Oracle
## Card
- useful for printing in a human-readable format
