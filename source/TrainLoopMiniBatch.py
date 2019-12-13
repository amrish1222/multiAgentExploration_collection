#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:56:01 2019

@author: bala
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import time
import pickle as pkl
import numpy as np

from tqdm import tqdm

import Training.SimpleNNagent as sNN
import Training.SimpleCNNagent as cNN
import Training.DoubleCNNagent as dcNN
import Training.DoubleCNNagent_priority as dcNN_p
import Training.DoubleCNNagent_priority_Noisy as dcNN_pn
from env import Env
from drone import Drone
from mobile_robot import MobileRobot
from mobileAgent import mobileRandomAgent
from constants import *

np.set_printoptions(precision=3, suppress=True)
NUM_DR = 1
NUM_MR = 1
env = Env(NUM_DR, NUM_MR)

# Parameters
NUM_EPISODES = 10000
LEN_EPISODE = 1000
reward_history = [] # remove
loss_history = [] # remove

dispFlag = True

#dAgent = sNN.SimpleNNagent(env, loggingLevel = 1)
#dAgent = cNN.SimpleCNNagent(env, loggingLevel = 3)
#dAgent = dcNN.DoubleCNNagent(env, loggingLevel = 3)
#dAgent = dcNN_p.DoubleCNNagent_Priority(env, loggingLevel = 3)
dAgent = dcNN_pn.DoubleCNNagent_Priority_Noisy(env, loggingLevel = 3)

curr_state = env.reset() # mrPos, mrVel, localArea, dronePos, droneVel, droneCharge, dock, done
c_mrPos, \
c_mrVel, \
c_localArea, \
c_dronePos, \
c_droneVel, \
c_droneCharge, \
c_dock, \
c_reward, \
c_done = curr_state
for ndx in range(NUM_DR):
    c_s = [c_mrPos[0],
           c_mrVel[0],
           c_localArea[0],
           c_dronePos[ndx],
           c_droneVel[ndx],
           c_droneCharge[ndx],
           c_dock[ndx],
           c_reward[ndx],
           c_done[ndx]]
dAgent.summaryWriter_showNetwork(c_s)
    
aDocks = []
for i in range(NUM_DR):
    aDocks.append(False)
mActions = []
for i in range(NUM_MR):
    mActions.append(0)

# Run for NUM_EPISODES
for episode in tqdm(range(NUM_EPISODES)):
    dAgent.newGame()
    episode_reward = 0
    episode_loss = 0
    curr_state = env.reset() # mrPos, mrVel, localArea, dronePos, droneVel, droneCharge, dock, done
#    print(f"episode, epsilon : {episode, dAgent.epsilon}")
    
    for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
        if episode % 25 == 0 and dispFlag or episode > 1:
            if RENDER_PYGAME:
                env.render()
        
        if step == 200 -1:
            print("200 steps run")
        
        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
        c_mrPos, \
        c_mrVel, \
        c_localArea, \
        c_dronePos, \
        c_droneVel, \
        c_droneCharge, \
        c_dock, \
        c_reward, \
        c_done = curr_state
        dAction = []
        for ndx in range(NUM_DR):
            dAction.append(dAgent.getTrainAction((c_mrPos[0],
                                                  c_mrVel[0],
                                                  c_localArea[0],
                                                  c_dronePos[ndx],
                                                  c_droneVel[ndx],
                                                  c_droneCharge[ndx],
                                                  c_dock[ndx],
                                                  c_reward[ndx],
                                                  c_done[ndx])))

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       LEN_EPISODE steps are done
        next_state = env.step(mActions, dAction, aDocks)

        # This is where your NN/GP code should go
        # Create target vector
        # Train the network/GP
        n_mrPos, \
        n_mrVel, \
        n_localArea, \
        n_dronePos, \
        n_droneVel, \
        n_droneCharge, \
        n_dock, \
        n_reward, \
        n_done = next_state
        for ndx in range(NUM_DR):
            c_s = [c_mrPos[0],
                   c_mrVel[0],
                   c_localArea[0],
                   c_dronePos[ndx],
                   c_droneVel[ndx],
                   c_droneCharge[ndx],
                   c_dock[ndx],
                   c_reward[ndx],
                   c_done[ndx]]
            n_s = [n_mrPos[0],
                   n_mrVel[0],
                   n_localArea[0],
                   n_dronePos[ndx],
                   n_droneVel[ndx],
                   n_droneCharge[ndx],
                   n_dock[ndx],
                   n_reward[ndx],
                   n_done[ndx]]
            dAgent.buildReplayMemory(c_s, n_s, dAction[ndx])
        loss = dAgent.buildMiniBatchTrainData()
        dAgent.trainModel()

        # Record history
        reward = sum(n_reward)
        episode_reward += reward
        episode_loss += loss

        # Current state for next step
        curr_state = next_state
        
#        if not np.any(n_localArea[0] == 50) or env.checkClose() or any(n_done):
        if np.count_nonzero(n_localArea[0] == 255) == 7*7 or env.checkClose() or any(n_done):
            # Epsilon Decay
            if dAgent.epsilon >= dAgent.minEpsilon:
                dAgent.epsilon *= dAgent.epsilonDecay
            
            # Record history
            reward_history.append(episode_reward)
            loss_history.append(episode_loss)
            dAgent.summaryWriter_addMetrics(episode, episode_loss, episode_reward, step)
            # You may want to plot periodically instead of after every episode
            # Otherwise, things will slow
            if episode % 25 == 0:
# =============================================================================
#                 if dispFlag:
#                     fig = plt.figure(1)
#                     plt.clf()
# #                    plt.xlim([0,NUM_EPISODES])
#                     plt.plot(reward_history,'ro')
#                     plt.xlabel('Episode')
#                     plt.ylabel('Reward')
#                     plt.title('Reward Per Episode')
#                     plt.pause(0.01)
#                     fig.canvas.draw()
#                     
#                     fig = plt.figure(2)
#                     plt.clf()
# #                    plt.xlim([0,NUM_EPISODES])
#                     plt.plot(loss_history,'bo')
#                     plt.xlabel('Episode')
#                     plt.ylabel('Loss')
#                     plt.title('Loss per episode')
#                     plt.pause(0.01)
#                     fig.canvas.draw()
# =============================================================================
                dAgent.saveModel("checkpoints")
            break
    
dAgent.saveModel("checkpoints")
dAgent.summaryWriter_close()
