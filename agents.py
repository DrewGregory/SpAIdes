import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch


from player import Player, Baseline
import random
import game
from card import Card
import utils


class ModelPlayer(Baseline):
    '''
    General Q-Learning class

    discount -- policy discount terms
    qmodel -- a learning model class, that supports prediction and updating
    featureExtractor -- an extractor for the state
    exploreProb -- eps greedy policy
    '''

    def __init__(self, discount, model, actions, featureExtractor, exploreProb, hand, name="",):
        super().__init__(hand, name)
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploreProb = exploreProb
        self.numiters = 0
        self.model = model # evaluation function class
    """
    def declareBid(self, state):
        # which bid gives us our best q?
        if random.random() < .05:
            choice = random.choice(range(13))
            #print("random choice: " + str(choice))
            self.bid = choice
            return self.bid
        bestQ = (float("-inf"), None)
        for i in range(0, 14):
            newStateArr = [0] * 14
            newStateArr[i] = 1
            state[2] = newStateArr + state[2][14:]
            newQ = float(self.getQ(state, None))
            #print("NEWQ : " + str(newQ))
            bestQ = max(bestQ, (newQ, i))
        # Don't need to revert our bid cuz it will be overwritten
        #print("bestChoice: " + str(bestQ[1]))
        self.bid = bestQ[1]
        return self.bid
        """

    def getQ(self, state, actions):
        vector_features = self.featureExtractor(state, actions)
        output =  self.model.predict(vector_features)
        return output

    def getStepSize(self):
        return 1.0 / self.numiters

    def incorporateFeedback(self, newState, reward):
        lastState, lastAction = self.playHistory[-1]
        self.personalFeedback(lastState, lastAction, reward, newState)


    def personalFeedback(self, state, action, reward, newState):
        self.numiters += 1
        vector_features = self.featureExtractor(state, action)
        target = reward + self.discount * self.getQ(newState, action)
        self.model.update(vector_features, target)


    def playCard(self, state, actions, pile=None):
        if random.random() < self.exploreProb:
            chosen = random.choice(actions)
            self.hand.remove(chosen)
            return chosen

        # tuple hax
        score, chosen = max([(self.getQ(state,action), action) for action in actions])
        self.hand.remove(chosen)
        self.playHistory.append((state, chosen))
        return chosen


'''

Class used to encapsulate generic evaluation functions that we might want
to test out for Q_opt.

E.X 

hidden = 100
lr = 1e-3 # usually a reasonable val
weights = nn.Sequential(
    nn.Linear(52, hidden ),
    nn.ReLU(),
    nn.Linear(hidden, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(weights.parameters(), lr=learning_rate)

def pred(weights, features):
    with torch.no_grad():
        score = weights(features)
        return weights

def upd(weights, features, target):
    current estimate = weights(features)
    loss = criterion(current_estimate, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

class QModel:

    def __init__(self, model, predict_lambda, update_lambda):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
    
    def predict(self, features):
        features = torch.tensor(features)
        return self.predict_lambda(self.model, features)
    
    def update(self, features, target):
        
        features = torch.tensor(features)
        self.update_lambda(self.model, features, target)



### Play test around
class ModelTest(ModelPlayer):

    def __init__(self, hand, name=""):

        
        learning_rate = 1e-3 # usually a reasonable val
        LEN_FEATURE_VECTOR =      52    +          52      +     14*4     +  52   +   4    +    4
        #                    playerCards    claimedCards    playerBids   pile    tricks       
        
        # Auto create deep linear NN from just changing hidden
        hidden = [100,  200] ##Just change this
        
        #######  Keep Here ############
        modules = [nn.Linear(LEN_FEATURE_VECTOR, hidden[0])]
        for i in range(len(hidden)-1 ):
            modules.append(nn.Linear(hidden[i], hidden[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden[-1], 1))
        ######   KEEP HERE #########
        
        weights = nn.Sequential(*modules)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(weights.parameters(), lr=learning_rate, weight_decay=1e-4)
        # setup gpu computing
        if cuda.is_available():
            weights = weights.cuda()
            criterion = criterion.cuda()
            


        def pred(weights, features):
            with torch.no_grad():
                if cuda.is_available():
                    features = features.cuda()
                score = weights(features)
            return score

        def upd(weights, features, target):
            t = torch.tensor(float(target))
            if cuda.is_available():
                features = features.cuda()
                t = t.cuda()
            
            current_estimate = weights(features)
            loss = criterion(current_estimate, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        testQModel = QModel(weights, pred, upd)
        super().__init__(1, testQModel, utils.genActions, game.Game.stateFeatureExtractor, .01, hand, name)
