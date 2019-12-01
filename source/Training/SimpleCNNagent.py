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
    def __init__(self,env, device):
        super().__init__()
        self.stateSpaceSz, \
        self.w, \
        self.h, \
        self.drPos, \
        self.mrVel, \
        self.mrPos, \
        self.dCharge = env.getStateSpace()
        
        self.device = device
        
        # cnn
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 5)
        self.mp1 = nn.MaxPool2d(2)
        self.cnn2 = nn.Conv2d(in_channels = 5, out_channels = 1, kernel_size = 5)
        
        # fc
        self.fcInputs = self.mrPos + self.mrVel + self.drPos + self.dCharge
        self.l1 = nn.Linear(in_features = self.fcInputs, out_features = self.fcInputs)
        
        # concat
        self.fc1 = nn.Linear(in_features = (14*14)+self.fcInputs, out_features = 256)
        self.fc2 = nn.Linear(in_features = 256, out_features = len(env.getActionSpace()))
    
    def forward(self, x1, x2):
#        x1,x2 = x
        
        #cnn
        x1 = F.relu(self.mp1(self.cnn1(x1)))
        x1 = F.relu(self.cnn2(x1))
    
        #fc
        x2 = F.relu(self.l1(x2))
        
        #concat
        x = torch.cat((x1.flatten(start_dim = 1),x2), dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def stitch_batch(self,stitched_states):
        # cnn tensor = [num_samples, num_channels, num_width, num_height]
        cnn_x = np.zeros((len(stitched_states),
                          stitched_states[0][0].shape[1],
                          stitched_states[0][0].shape[2],
                          stitched_states[0][0].shape[3]))
        fc_x = []
        for ndx, (cnn_i, fc_i) in enumerate(stitched_states):
            cnn_x[ndx, :, : , :] = cnn_i
            fc_x.append(fc_i)
        cnn_x = torch.from_numpy(cnn_x).to(self.device)
        cnn_x = cnn_x.float()
        fc_x = torch.from_numpy(np.reshape(np.asarray(fc_x),
                                           (len(stitched_states),-1))).to(self.device)
        fc_x = fc_x.float()
        return (cnn_x, fc_x)
    
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
        fc_i = np.hstack((np.asarray(n_mrPos).reshape(-1),
                          np.asarray(n_mrVel).reshape(-1),
                          np.asarray(n_dronePos).reshape(-1),
                          np.asarray(n_droneCharge).reshape(-1)))
        n_localArea = np.asarray(n_localArea)
        cnn_i = n_localArea.reshape((1, 1, n_localArea.shape[0], n_localArea.shape[1]))
        return (cnn_i, fc_i)
    
class SimpleCNNagent():
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
        self.sw = SummaryWriter(log_dir=f"tf_log/demo_CNN{random.randint(0, 1000)}")
        
    def buildModel(self,env):   
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device : {self.device}')
        self.model = agentModelCNN1(env, self.device).to(self.device)
        self.loss_fn = nn.MSELoss()
#        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learningRate)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learningRate)
        
    def trainModel(self):
        self.model.train()
        X = self.trainX
        Y = torch.from_numpy(self.trainY).to(self.device)
        for i in range(2): # number epoh
            self.optimizer.zero_grad()
            predY = self.model(*X)
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
            X = self.model.stitch_batch([self.model.stitch(state)])
            self.qValues = self.model(*X).cpu().detach().numpy()[0]
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
        X = self.model.stitch_batch([self.model.stitch(state)])
        self.qValues = self.model(*X).cpu().detach().numpy()[0]
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
        for ndx,[currState, nextState, action] in enumerate(minibatch):
            c.append(self.model.stitch(currState))
            n.append(self.model.stitch(nextState))
            r.append(nextState[-2])
            d.append(nextState[-1])
            a.append([ndx, action])
        c = self.model.stitch_batch(c)
        n = self.model.stitch_batch(n)
        r = np.asanyarray(r)
        d = np.asanyarray(d)
        a = np.asanyarray(a)
        a = a.T
        self.model.eval()
        X = n
        qVal_n = self.model(*X).cpu().detach().numpy()
        qMax_n = np.max(qVal_n, axis  = 1)
        X = c
        qVal_c = self.model(*X).cpu().detach().numpy()
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
        torch.save(self.model, f"{filePath}/{self.model.__class__.__name__}.pt")
    
    def loadModel(self, filePath):
        self.model = torch.load(filePath)
    
    def summaryWriter_showNetwork(self, curr_state) :
        X = self.model.stitch_batch([self.model.stitch(curr_state)])
        self.sw.add_graph(self.model, X)
    
    def summaryWriter_addMetrics(self, episode, loss, reward, lenEpisode):
        self.sw.add_scalar('Loss', loss, episode)
        self.sw.add_scalar('Reward', reward, episode)
        self.sw.add_scalar('Episode Length', lenEpisode, episode)
        
        self.sw.add_histogram('l1.bias', self.model.l1.bias, episode)
        self.sw.add_histogram('l1.weight', self.model.l1.weight, episode)
        self.sw.add_histogram('l1.weight.grad', self.model.l1.weight.grad, episode)
    
    def summaryWriter_close(self):
        self.sw.close()