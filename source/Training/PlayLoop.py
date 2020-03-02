#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:50:10 2019

@author: bala
"""


import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17

import SimpleNNagent_torch as sNN

env = gym.make('MountainCar-v0')
env.seed(42);

# Print some info about the environment
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space)

# Parameters
NUM_STEPS = 200
NUM_EPISODES = 100
LEN_EPISODE = 200
reward_history = []
loss_history = []
max_dist = []
final_position = []
success = 0
noSteps = []

agent = sNN.SimpleNNagent_torch(env)
agent.loadModel("model_torch_discrete.h5")
agent.summaryWriter_showNetwork()

# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
    agent.newGame()
    episode_reward = 0
    episode_loss = 0
    episode_maxDist = -0.4
    curr_state = env.reset()
    
    for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
        env.render()
        
        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
        action = agent.getAction(curr_state)

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       200 steps are done
        next_state, reward, done, _ = env.step(action)
        
        # This is where your NN/GP code should go
        # Create target vector
        # Train the network/GP

        # Record history
        episode_reward += reward
        if next_state[0] > episode_maxDist:
            episode_maxDist = next_state[0]

        # Current state for next step
        curr_state = next_state
        
        
        if (done and step == LEN_EPISODE-1) or (curr_state[0] >=0.5):
            # Record history
            if curr_state[0] >= 0.5:
                success += 1
                noSteps.append(step)
            reward_history.append(episode_reward)
            loss_history.append(episode_loss)
            max_dist.append(episode_maxDist)
            final_position.append(curr_state[0])
            agent.summaryWriter_addMetrics(episode, episode_loss, episode_reward, episode_maxDist)
            # You may want to plot periodically instead of after every episode
            # Otherwise, things will slow
            if episode % 25 == 0:
                fig = plt.figure(1)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(reward_history,'ro')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode')
                plt.pause(0.01)
                fig.canvas.draw()
                
                fig = plt.figure(2)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(noSteps,'bo')
                plt.xlabel('Episode')
                plt.ylabel('Number of Steps')
                plt.title('Number of steps Taken Per Episode')
                plt.pause(0.01)
                fig.canvas.draw()
                
                fig = plt.figure(3)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(max_dist,'yo')
                plt.xlabel('Episode')
                plt.ylabel('Max Distance')
                plt.title('Max distance Per Episode')
                plt.pause(0.01)
                fig.canvas.draw()
                
            break
agent.summaryWriter_close()
print("----------------------  Metrics  ----------------------")
print(f"Number of Episodes = {NUM_EPISODES}")
print(f"Success Rate = {success} %")
print(f"Average number of steps taken  = {sum(noSteps)/len(noSteps)}")