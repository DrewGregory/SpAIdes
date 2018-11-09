from player import Player
import random


class ModelPlayer(Player):
    '''
    General TD-Learning class

    discount -- policy discount terms
    qmodel -- a learning model class, that supports prediction and updating
    featureExtractor -- an extractor for the state
    exploreProb -- eps greedy policy
    '''

    def __init__(self, discount, model, featureExtractor, exploreProb=0):
        super(self)
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploreProb = exploreProb
        self.numiters = 0
        self.model = model

    def getTD(self, state, actions):
        vector_features = self.featureExtractor(state, actions)
        return self.model.predict(vector_features)

    def getStepSize(self):
        return 1.0 / self.numiters

    def incorporateFeedback(self, state, action, reward, newState):

        currentEstimate = self.getTD(state, action)
        target = reward + self.discount * self.getTD(newState, action)
        diff = self.getStepSize() * (currentEstimate - target)
        self.model.update(diff)

    def declareBid(self, state):
        return 0

    def playCard(self, state, actions, pile=None):
        self.numiters += 1
        if random.random() < self.exploreProb:
            return random.choice(actions)

        # tuple hax
        return self.getTD(state, actions)



'''
E.X model = nn.Sequential(
    nn.Linear(),
    nn.ReLU(),
    nn.Linear() 
    )

    def p(features):
        with(torch.no_grad()):
            result = model(features)
        return result
    def u(features):
        result = model(features)
        loss = loss_criterion(result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    td = TDModel(model, p, u)
'''

class TDModel:

    def __init__(self, model, predict_lambda, update_lambda):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
    
    def predict(self, features):
        return self.predict_lambda(features)
    
    def update(self, diff):
        self.update_lambda(diff)
    