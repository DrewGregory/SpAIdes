import numpy as np
import torch.nn as nn

from player import Player, Baseline
import random
import game
from card import Card


class ModelPlayer(Baseline):
    '''
    General Q-Learning class

    discount -- policy discount terms
    qmodel -- a learning model class, that supports prediction and updating
    featureExtractor -- an extractor for the state
    exploreProb -- eps greedy policy
    '''

    def __init__(self, discount, model, actions, featureExtractor, exploreProb=0):
        super(self)
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploreProb = exploreProb
        self.numiters = 0
        self.model = model # evaluation function class

    def getQ(self, state, actions):
        vector_features = self.featureExtractor(state, actions)
        output =  self.model.predict(vector_features)
        return output

    def getStepSize(self):
        return 1.0 / self.numiters

    def incorporateFeedback(self, terminalState, reward):

        lastState, lastAction = self.playHistory[-1]
        self.personalFeedback(lastState, lastAction, reward, terminalState)

        for i in reversed(range(len(self.playHistory))):
            nextState, nextAction = self.playHistory[i]
            state, action = self.playHistory[i-1]
            self.personalFeedback(state, action, 0, next_state)


    def personalFeedback(self, state, action, reward, newState):
        self.numiters += 1
        currentEstimate = self.getQ(state, action)
        target = reward + self.discount * self.getQ(newState, action)
        diff = self.getStepSize() * (currentEstimate - target)
        self.model.update(diff)


    def playCard(self, state, actions, pile=None):
        if random.random() < self.exploreProb:
            return random.choice(actions)

        # tuple hax
        score, chosen =  max( [ (self.getQ(state,action), action) for action in actions ] )

        self.hand.remove(chosen)
        self.playHistory.append((state, chosen))


'''

Class used to encapsulate generic evaluation functions that we might want
to test out for Q_opt.

E.X linearmodel = nn.Sequential(
    nn.Linear(),
    nn.ReLU(),
    nn.Linear() 
    )

    def pred(features):
        with(torch.no_grad()):
            result = model(features)
        return result
    def upd(features):
        result = model(features)
        loss = loss_criterion(result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    td = TDModel(model, p, u)c
'''

class QModel:

    def __init__(self, model, predict_lambda, update_lambda):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
    
    def predict(self, features):
        return self.predict_lambda(features)
    
    def update(self, diff):
        self.update_lambda(diff)
    



### Play test around

hidden = 100
weights = nn.Sequential(
    nn.Linear(52, hidden ),
    nn.ReLU(),
    nn.Linear(hidden, 52)
)

