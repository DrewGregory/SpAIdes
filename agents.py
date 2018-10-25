from player import Player



class ModelPlayer(Player):

    '''
    Policy Evaluator class follows a predefined policy
    and attempts to learn a value function
    '''
    def __init__(self, bidPolicy, playPolicy):
        super(self)
        self.bidPolicy = bidPolicy
        self.playPolicy = playPolicy

    def declareBid(self, state):
        return self.bidPolicy(self, state)

    def playCard(self, state, actions, pile=None):
        action = self.playPolicy(state, actions, pile)
        return action


    
    


    