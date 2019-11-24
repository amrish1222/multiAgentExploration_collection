#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  14 09:45:29 2019

@author: bala
"""

import random
import numpy as np
import copy
from sklearn.metrics import mean_squared_error as skMSE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

class agentModel(nn.Module):
    def __init__(self,iSize, oSize):
        super().__init__()
        self.l1 = nn.Linear(in_features=iSize, out_features=64)
        self.l2 = nn.Linear(in_features=64, out_features=64)
        self.l3 = nn.Linear(in_features=64, out_features=oSize)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
        
class SimpleNNagent():
    def __init__(self):
#        self.trainX = []
#        self.trainY = []
#        self.replayMemory = []
#        self.epsilon = 1.0
#        self.minEpsilon = 0.01
#        self.epsilonDecay = 0.997
#        self.discount = 0.95
#        self.learningRate = 0.002
#        self.batchSize = 128
#        self.sLow = env.observation_space.low
#        self.sHigh = env.observation_space.high
#        self.nActions = env.action_space.n
#        self.buildModel(env.observation_space.shape[0], env.action_space.n)
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_{random.randint(0, 1000)}")
        
    def nState(self, state):
#        return np.divide(state-self.sLow,
#                         (self.sHigh-self.sLow))
        return state
        
    def buildModel(self,iSize, oSize):   
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device : {self.device}')
        self.model = agentModel(iSize,oSize).to(self.device)
        self.loss_fn = nn.MSELoss()
#        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learningRate)
        
    def trainModel(self):
        self.model.train()
        X = torch.from_numpy(self.trainX).to(self.device)
        Y = torch.from_numpy(self.trainY).to(self.device)
        for i in range(2):
            self.optimizer.zero_grad()
            predY = self.model(X.float())
            loss = self.loss_fn(Y,predY)
            loss.backward()
            self.optimizer.step()
        
    def EpsilonGreedyPolicy(self,state):
        if random.random() <= self.epsilon:
            # choose random
            action = random.randint(0,self.nActions-1)
        else:
            #ChooseMax
            #Handle multiple max
            self.model.eval()
            X = torch.from_numpy(np.reshape(self.nState(state),(-1,2))).to(self.device)
            self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
            action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def newGame(self):
        self.trainX = []
        self.trainY = []
        print("new game")
    
    def getTrainAction(self,state):
        action = self.EpsilonGreedyPolicy(state)
        return action    
    
    def getAction(self,state):
        self.model.eval()
        X = torch.from_numpy(np.reshape(self.nState(state),(-1,2))).to(self.device)
        self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
        action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def buildReplayMemory(self, currState, nextState, reward, done, action):
#        if len(self.replayMemory)> self.batchSize:
#            self.replayMemory.pop()
        self.replayMemory.append([currState, nextState, reward, done, action])
    
    def buildMiniBatchTrainData(self):
        c = []
        n = []
        r = []
        d = []
        a = []
        if len(self.replayMemory)>self.batchSize:
            minibatch = random.sample(self.replayMemory, self.batchSize)
        else:
            minibatch = self.replayMemory
        for ndx,[currState, nextState, reward, done, action] in enumerate(minibatch):
            c.append(currState)
            n.append(nextState)
            r.append(reward)
            d.append(done)
            a.append([ndx, action])
        c = np.asanyarray(c)
        n = np.asanyarray(n)
        r = np.asanyarray(r)
        d = np.asanyarray(d)
        a = np.asanyarray(a)
        a = a.T
        self.model.eval()
        X = torch.from_numpy(np.reshape(self.nState(n),(-1,2))).to(self.device)
        qVal_n = self.model(X.float()).cpu().detach().numpy()
        qMax_n = np.max(qVal_n, axis  = 1)
        X = torch.from_numpy(np.reshape(self.nState(c),(-1,2))).to(self.device)
        qVal_c = self.model(X.float()).cpu().detach().numpy()
        Y = copy.deepcopy(qVal_c)
        y = np.zeros(r.shape)
        ndx = np.where(d == True)
        y[ndx] = r[ndx]
        ndx = np.where(d == False)
        y[ndx] = r[ndx] + self.discount * qMax_n[ndx]
        Y[a[0],a[1]] = y
        self.trainX = c
        self.trainY = Y
        return skMSE(Y,qVal_c)
        
    def saveModel(self, filePath):
        torch.save(self.model, filePath)
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
        
    def modelPredict(self, testSample):
        self.model.eval()
        X = torch.from_numpy(np.reshape(testSample,(-1,2))).to(self.device)
        predY = self.model(X.float()).cpu().detach().numpy()
        return predY
    
    def summaryWriter_showNetwork(self):
        self.sw.add_graph(self.model, torch.tensor([1.0, 1.0]).to(self.device))
    
    def summaryWriter_addMetrics(self, episode, loss, reward, maxDist):
        self.sw.add_scalar('Loss', loss, episode)
        self.sw.add_scalar('Reward', reward, episode)
        self.sw.add_scalar('MaxDistance', maxDist, episode)
        
        self.sw.add_histogram('l1.bias', self.model.l1.bias, episode)
        self.sw.add_histogram('l1.weight', self.model.l1.weight, episode)
#        self.sw.add_histogram('l1.weight.grad', self.model.l1.weight.grad, episode)
    
    def summaryWriter_close(self):
        self.sw.close()