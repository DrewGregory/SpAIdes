# NN stuffs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch
from tensorboardX import SummaryWriter

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
    exploreProb -- eps greedy policy
    '''

    def __init__(self, discount, model, actions, featureExtractor, exploreProb, hand, name="",):
        super().__init__(hand, name)
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.exploreProb = exploreProb
        self.numiters = 0
        self.model = model # evaluation function class
        self.actions = actions


    def declareBid(self, state):
        '''
        As number of iterations increases, probability of exploration
        decreases inversely proportionally. Alternate bid chosen with
        probability proportional to perceived Q-value
        '''
        bestQ = (float("-inf"), 0) # 0 because weird errors with None
        qVals = []
        minQ = float('inf')
        for i in range(Card.NUM_PER_SUIT + 1):
            state[2][0] = i
            newQ = float(self.getQ(state, None))
            bestQ = max(bestQ, (newQ, i))
            minQ = min(minQ, newQ)
            qVals.append(newQ) # index ~ bid number
        # Don't need to revert our bid cuz it will be overwritten
        assert not bestQ[0] == None
        self.bid = bestQ[1]

        # sample 'informed' random choice
        # dynamic explore prob, sqrt to incr probability as numiters increases
        if random.random() < np.sqrt(float(1 / max(self.numiters, 1))):
            # print('*** SAMPLING RANDOM: ***')
            # NOTE possible problem: strategy dictates least valuable strategy never chosen
            denom = float(sum(qVals) - len(qVals) * minQ)
            qVals = [float((qVal - minQ) / denom) for qVal in qVals]
            # print('original bid: ' + str(self.bid))
            # Using 1 because of default None because of cryptic dtype bug
            self.bid = np.random.choice(range(Card.NUM_PER_SUIT + 1), 1, p = qVals).tolist()[0]
            # print('randomly sampled bid: ' + str(self.bid))

        return self.bid

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

    def __init__(self, model, predict_lambda, update_lambda):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
        self.num_iters = 0

    
    def predict(self, features):
        features = torch.tensor(features)
        return self.predict_lambda(self.model, features)
    
    def update(self, features, target):
        self.num_iters += 1
        features = torch.tensor(features)
        loss = self.update_lambda(self.model, features, target)
        utils.TWriter.add_scalar('data/loss', loss, self.num_iters)

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


        pred, upd = self.getLambdas(criterion, optimizer, oldweights, weights)
        self.optimizer = optimizer
        self.weights = weights
        self.oldweights = oldweights
        self.testQModel = QModel(weights, pred, upd)
        super().__init__(1, self.testQModel, utils.genActions, game.Game.stateFeatureExtractor, .001, hand, name)


    def save(self, path="./qmodel"):

        state = {
            "model": self.weights.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, model, optimizer, path="./qmodel"):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    def getLambdas(self, criterion, optimizer, oldweights, weights):
        def pred(weights, features):
            with torch.no_grad():
                if cuda.is_available():
                    features = features.cuda()
                score = oldweights(features)
            return score

        def upd(weights, features, target):
            return 0
            t = torch.tensor(float(target))
            if cuda.is_available():
                features = features.cuda()
                t = t.cuda()
            current_estimate = weights(features)
            #if target < 10:
            #    print(str(current_estimate) + " " + str(target))
            loss = criterion(current_estimate, t)
            #print("Loss: " + str(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
        return pred, upd

    def getNNStructure(self, feature_len):
        return nn.Sequential(
            nn.Linear(feature_len, 400),
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
            nn.Linear(4*400, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
