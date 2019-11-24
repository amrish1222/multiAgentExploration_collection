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


class agentModelCNN1(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.stateSpaceSz, \
        self.w, \
        self.h, \
        self.drPos, \
        self.mrVel, \
        self.mrPos, \
        self.dCharge = env.getStateSpace()
        
        # cnn
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 5)
        self.mp1 = nn.MaxPool2d(2)
        self.cnn2 = nn.conv2d(in_channels = 5, out_channels = 16, kernel_size = 1)
        
        # fc
        self.fcInputs = self.mrPos + self.mrVel + self.drPos + self.dCharge
        self.l1 = nn.Linear(in_features = self.fcInputs, out_features = self.fcInputs)
        
        # concat
        self.fc1 = nn.Linear(in_features = 16+self.fcInputs, out_features = 16+self.fcInputs)
        self.fc2 = nn.Linear(in_features = 16+self.fcInputs, out_features = len(env.getActionSpace()))
    
    def forward(self, x):
        # todo how to handle batch training / batch inputs
        x1,x2 = x
        
        #cnn
        x1 = F.relu(self.mp1(self.cnn1(x1)))
        x1 = F.relu(self.cnn2(x1))
    
        #fc
        x2 = F.relu(self.l1(x2))
        
        #concat
        x = torch.cat((x1,x2), dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def stitch(self,state):
        n_mrPos, \
        n_mrVel, \
        n_localArea, \
        n_dronePos, \
        n_droneVel, \
        n_droneCharge, \
        n_dock, \
        n_reward, \
        n_done = state
        d1 = np.hstack((n_mrPos.reshape(-1),
                         n_mrVel.reshape(-1),
                         n_dronePos.reshape(-1),
                         np.asarray(n_droneCharge).reshape(-1)))
        d2 = n_localArea.reshape(-1)
        return (d1, d2)
    
class agentModelFC(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.stateSpaceSz, \
        self.w, \
        self.h, \
        self.drPos, \
        self.mrVel, \
        self.mrPos, \
        self.dCharge = env.getStateSpace()
        
        self.l1 = nn.Linear(in_features = self.stateSpaceSz, out_features = int(self.stateSpaceSz/2))
        self.l2 = nn.Linear(in_features = int(self.stateSpaceSz/2), out_features = int(self.stateSpaceSz/8))
        self.l3 = nn.Linear(in_features = int(self.stateSpaceSz/8), out_features = 64)
        self.l4 = nn.Linear(in_features = 64, out_features = len(env.getActionSpace()))
    
    def stitch(self,state):
        n_mrPos, \
        n_mrVel, \
        n_localArea, \
        n_dronePos, \
        n_droneVel, \
        n_droneCharge, \
        n_dock, \
        n_reward, \
        n_done = state
        
        return np.hstack((n_mrPos.reshape(-1),
                          n_mrVel.reshape(-1),
                          n_localArea.reshape(-1),
                          n_dronePos.reshape(-1),
                          np.asarray(n_droneCharge).reshape(-1)))
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
        
class SimpleNNagent():
    def __init__(self,env):
        self.trainX = []
        self.trainY = []
        self.replayMemory = []
        self.maxReplayMemory = 5000
        self.epsilon = 1.0
        self.minEpsilon = 0.01
        self.epsilonDecay = 0.997
        self.discount = 0.95
        self.learningRate = 0.002
        self.batchSize = 128
        self.envActions = env.getActionSpace()
        self.nActions = len(self.envActions)
        self.buildModel(env)
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_{random.randint(0, 1000)}")
        
    def buildModel(self,env):   
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device : {self.device}')
        self.model = agentModelFC(env).to(self.device)
        self.loss_fn = nn.MSELoss()
#        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learningRate)
        
    def trainModel(self):
        self.model.train()
        X = torch.from_numpy(self.trainX).to(self.device)
        Y = torch.from_numpy(self.trainY).to(self.device)
        for i in range(2): # number epoh
            self.optimizer.zero_grad()
            predY = self.model(X.float())
            loss = self.loss_fn(Y,predY)
            loss.backward()
            self.optimizer.step()
        
    def EpsilonGreedyPolicy(self,state):
        if random.random() <= self.epsilon:
            # choose random
            action = self.envActions[random.randint(0,self.nActions-1)]
        else:
            #ChooseMax
            #Handle multiple max
            self.model.eval()
            X = torch.from_numpy(np.reshape(self.model.stitch(state),(1,-1))).to(self.device)
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
        X = torch.from_numpy(np.reshape(self.model.stitch(state),(1,-1))).to(self.device)
        self.qValues = self.model(X.float()).cpu().detach().numpy()[0]
        action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def buildReplayMemory(self, currState, nextState, action):
        if len(self.replayMemory)> self.maxReplayMemory:
            self.replayMemory.pop()
        self.replayMemory.append([currState, nextState, action])
    
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
        bSize = len(minibatch)
        for ndx,[currState, nextState, action] in enumerate(minibatch):
            c.append(self.model.stitch(currState))
            n.append(self.model.stitch(nextState))
            r.append(nextState[-2])
            d.append(nextState[-1])
            a.append([ndx, action])
        c = np.asanyarray(c)
        n = np.asanyarray(n)
        r = np.asanyarray(r)
        d = np.asanyarray(d)
        a = np.asanyarray(a)
        a = a.T
        self.model.eval()
        X = torch.from_numpy(np.reshape(n,(bSize,-1))).to(self.device)
        qVal_n = self.model(X.float()).cpu().detach().numpy()
        qMax_n = np.max(qVal_n, axis  = 1)
        X = torch.from_numpy(np.reshape(c,(bSize,-1))).to(self.device)
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
    
    def summaryWriter_showNetwork(self):
        self.sw.add_graph(self.model, torch.tensor([1.0]*self.model.stateSpaceSz).to(self.device))
    
    def summaryWriter_addMetrics(self, episode, loss, reward):
        self.sw.add_scalar('Loss', loss, episode)
        self.sw.add_scalar('Reward', reward, episode)
        
        self.sw.add_histogram('l1.bias', self.model.l1.bias, episode)
        self.sw.add_histogram('l1.weight', self.model.l1.weight, episode)
#        self.sw.add_histogram('l1.weight.grad', self.model.l1.weight.grad, episode)
    
    def summaryWriter_close(self):
        self.sw.close()