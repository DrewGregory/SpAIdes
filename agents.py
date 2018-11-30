# NN stuffs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch
from tensorboardX import SummaryWriter
import math

# game stuff
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
    '''

    def __init__(self, discount, model, actions, featureExtractor, hand, name=""):
        super().__init__(hand, name)
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.numiters = 0
        self.model = model # evaluation function class
        self.actions = actions
        self.bidderModel = BidderModel()
        self.numBids = 1
        self.bidsPerBid = [1] * 14

    def declareBid(self, state):
        '''
        Use model to determine value of each bid, then bid accordingly.
        '''
        features = self.featureExtractor(state, None)
        # Trim down features to player hand and player bids.
        importantFeatures = features[:52] + features[104:108]
        print(importantFeatures)
        bestBid = (0,0)
        for i in range(0, 14):
            importantFeatures[52] = i
            bestBid = max(bestBid, (self.bidderModel.predictor(self.bidderModel.weights, \
                torch.tensor(importantFeatures)) + math.sqrt((2 * math.log(self.numBids))/(self.bidsPerBid[i])), i))
        self.bid = bestBid[1]
        importantFeatures[52] = self.bid
        self.biddingFeatures = importantFeatures
        self.numBids += 1
        self.bidsPerBid[self.bid] += 1
        if self.numBids % 1000 == 0:
            state = {
                "model": self.bidderModel.weights.state_dict(),
                "optimizer": self.bidderModel.optimizer.state_dict(),
            }
            torch.save(state, "./bidder")
        return self.bid

    def calculateScore(self):
        score = super().calculateScore()
        self.bidderModel.updater(self.bidderModel.weights, torch.tensor(self.biddingFeatures), score)
        return score

    def getQ(self, state, action):
        vector_features = self.featureExtractor(state, action)
        output = self.model.predict(vector_features)
        return output

    def getStepSize(self):
        return 1.0 / self.numiters

    def incorporateFeedback(self, newState, reward):
        lastState, lastAction = self.playHistory[-1]
        self.personalFeedback(lastState, lastAction, reward, newState)

    def personalFeedback(self, state, action, reward, newState):
        self.numiters += 1
        vector_features = self.featureExtractor(state, action)
        # get best action for next state
        
        nextActions =  self.actions(*game.Game.genActionParams(newState))
        # print(nextActions)
        nextQs = [(self.getQ(newState, a) , a) for a in nextActions]
        nextBestQ = (max(nextQs))[0] if len(nextQs) > 0 else 0
        target = reward + self.discount * nextBestQ
        # print("TARGET: " + str(target) + " " + str(reward) + " " + str(nextBestQ))
        self.model.update(vector_features, target)

    def playCard(self, state, actions, pile=None):
        # naive UCB
        if random.random() < np.sqrt(float(1 / max(self.numiters, 1))):
            chosen = random.choice(actions)
        else:
            # tuple hax
            score, chosen = max([(self.getQ(state,action), action) for action in actions])
        self.hand.remove(chosen)
        self.playHistory.append((state, chosen))
        # print("MODEL PLAYED:", chosen)
        return chosen



class QModel:

    def __init__(self, model, predict_lambda, update_lambda, name=""):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
        self.num_iters = 0
        self.name = name

    
    def predict(self, features):
        features = torch.tensor(features)
        return self.predict_lambda(self.model, features)
    
    def update(self, features, target):
        self.num_iters += 1
        features = torch.tensor(features)
        loss = self.update_lambda(self.model, features, target)
        utils.TWriter.add_scalar('data/loss' + self.name, loss, self.num_iters)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(-1)

class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()
    def forward(self, x):
        return x.view(1, 1,-1)


def getLambdas(criterion, optimizer, oldweights, weights):
        def pred(weights, features):
            with torch.no_grad():
                if cuda.is_available():
                    features = features.cuda()
                score = oldweights(features)
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
            return loss
        return pred, upd

class BidderModel:

    def __init__(self):
        self.weights = self.getNNStructure()
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.weights.parameters(), lr=1e-3, weight_decay=0)
        self.predictor, self.updater = getLambdas(self.criterion, self.optimizer, self.weights, self.weights)
        self.model = QModel(self.weights, self.predictor, self.updater, name="BidderModel")

    def getNNStructure(self):
        BIDDING_FEATURE_LENGTH = 56 # Player Hand + Bids
        return nn.Sequential(
            nn.Linear(BIDDING_FEATURE_LENGTH, 40),
            Unflatten(),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            Flatten(),
            nn.Linear(4*40, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

### Play test around
class ModelTest(ModelPlayer):

    QMODEL_OLD = None
    QMODEL_TRAIN = None
    QMODEL_CRITERION = None


    def __init__(self, hand, name=""):

        
        learning_rate = 1e-3 # usually a reasonable val
        LEN_FEATURE_VECTOR =      52    +        52        +     4    +  52   +  4   +     4      + 52
        #                    playerCards    claimedCards    playerBids   pile  tricks  playerBags   actions      
        
        weights = self.getNNStructure(LEN_FEATURE_VECTOR)
        oldweights = self.getNNStructure(LEN_FEATURE_VECTOR)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(weights.parameters(), lr=learning_rate, weight_decay=0)

        if not (ModelTest.QMODEL_OLD == None and ModelTest.QMODEL_TRAIN == None):
            weights = ModelTest.QMODEL_TRAIN
            oldweights = ModelTest.QMODEL_TEST
            criterion = ModelTest.QMODEL_CRITERION
        else:
            # setup gpu computing
            if cuda.is_available():
                weights = weights.cuda()
                oldweights = oldweights.cuda()
                criterion = criterion.cuda()
                optimizer = optim.Adam(weights.parameters(), lr=learning_rate, weight_decay=0).cuda()
                print("cuda'd optimizer")
            
            self.load(oldweights, optimizer) #use when need to load old models
            self.load(weights, optimizer) #use when need to load old models
            ModelTest.QMODEL_TRAIN = weights
            ModelTest.QMODEL_TEST = oldweights
            ModelTest.QMODEL_CRITERION = criterion


        pred, upd = getLambdas(criterion, optimizer, oldweights, weights)
        self.optimizer = optimizer
        self.weights = weights
        self.oldweights = oldweights
        self.testQModel = QModel(weights, pred, upd)
        super().__init__(1, self.testQModel, utils.genActions, game.Game.stateFeatureExtractor,  hand, name)


    def save(self, path="./qmodel"):
        
        state = {
            "model": self.weights.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, model, optimizer, path="./qmodel"):
        deviceName = 'cpu'
        if cuda.is_available():
            deviceName = 'cuda'
        device = torch.device(deviceName)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    

    def getNNStructure(self, feature_len):
        return nn.Sequential(
            nn.Linear(feature_len, 40),
            Unflatten(),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            Flatten(),
            nn.Linear(4*40, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
