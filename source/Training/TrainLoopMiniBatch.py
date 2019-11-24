#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:56:01 2019

@author: bala
"""

import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
import time
import pickle as pkl
import numpy as np

import SimpleNNagent_torch as sNN
import PolicyVisualization as pv
import GetUserInput as gui

env = gym.make('MountainCar-v0')
env.seed(42);

# Print some info about the environment
print("MiniBatch Training Loop")
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space)

# Parameters
NUM_STEPS = 200
NUM_EPISODES = 5000
LEN_EPISODE = 200
reward_history = []
loss_history = []
max_dist = []
maxDist = -0.4
dispFlag = True

agent = sNN.SimpleNNagent_torch(env)
agent.summaryWriter_showNetwork()

# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
    agent.newGame()
    episode_reward = 0
    episode_loss = 0
    episode_maxDist = -0.4
    curr_state = env.reset()
    print(f"episode : {episode}")
    
    for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
        if episode % 25 == 0 and dispFlag:
            env.render()
        
        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
        action = agent.getTrainAction(curr_state)
        

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       200 steps are done
        next_state, reward, done, _ = env.step(action)
        
        reward = agent.getReward(curr_state,
                        next_state, 
                        action, 
                        reward, 
                        maxDist,
                        step,
                        done)

        # This is where your NN/GP code should go
        # Create target vector
        # Train the network/GP
        agent.buildReplayMemory(curr_state, next_state, reward, done, action)
        loss = agent.buildMiniBatchTrainData()
        agent.trainModel()
        if agent.epsilon >= agent.minEpsilon:
            agent.epsilon *= agent.epsilonDecay

        # Record history
        episode_reward += reward
        episode_loss += loss
        if next_state[0] > episode_maxDist:
            episode_maxDist = next_state[0]
        
        if next_state[0] > maxDist:
            maxDist = next_state[0]

        # Current state for next step
        curr_state = next_state
        
        if done:
            # Record history
            reward_history.append(episode_reward)
            loss_history.append(episode_loss)
            max_dist.append(episode_maxDist)
            agent.summaryWriter_addMetrics(episode, episode_loss, episode_reward, episode_maxDist)
            # You may want to plot periodically instead of after every episode
            # Otherwise, things will slow
            if episode % 25 == 0:
                ip = gui.getUsernput(1)
                if ip == 'y':
                    dispFlag = True
                elif ip == 'n':
                    dispFlag = False
                if dispFlag:
                    fig = plt.figure(1)
                    plt.clf()
#                    plt.xlim([0,NUM_EPISODES])
                    plt.plot(reward_history,'ro')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Reward Per Episode')
                    plt.pause(0.01)
                    fig.canvas.draw()
                    
                    fig = plt.figure(2)
                    plt.clf()
#                    plt.xlim([0,NUM_EPISODES])
                    plt.plot(loss_history,'bo')
                    plt.xlabel('Episode')
                    plt.ylabel('Loss')
                    plt.title('Loss per episode')
                    plt.pause(0.01)
                    fig.canvas.draw()
                    
                    fig = plt.figure(3)
                    plt.clf()
#                    plt.xlim([0,NUM_EPISODES])
                    plt.plot(max_dist,'yo')
                    plt.xlabel('Episode')
                    plt.ylabel('Max Distance')
                    plt.title('Max distance Per Episode')
                    plt.pause(0.01)
                    fig.canvas.draw()
                
                agent.saveModel("model_torch.h5")
                
            break
    if episode % 100 == 0 and dispFlag:
        pass
        pv.ploicyViz_torch(agent)  
#    break
    
agent.saveModel("model_torch.h5")
pkl.dump([max_dist, loss_history, reward_history], open( "history.pkl", "wb" ))
#[max_dist, loss_history, reward_history] = pkl.load( open( "history.pkl", "rb" ))
pv.ploicyViz_torch(agent)
agent.summaryWriter_close()
