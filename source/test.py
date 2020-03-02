# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Main Run file

import numpy as np
import pygame
import time 
import matplotlib.pyplot as plt

from env import Env
from drone import Drone
from mobile_robot import MobileRobot
from mobileAgent import mobileRandomAgent
from constants import *

np.set_printoptions(precision=3, suppress=True)

env = Env(1, 1)

mAgent = mobileRandomAgent()
dAgent = [mobileRandomAgent(), mobileRandomAgent()]

win_close_f = False

actionSpace = env.getActionSpace()
stateSpace = env.getStateSpace()

for episodes in range(NUM_EPISODES):
    print(f"episode {episodes}")
    env.reset()
    while True:
        droneActions = []
        docks = []
        for i in range(2):
            droneActions.append(dAgent[i].getAction())
#            droneActions.append(1)
            docks.append(False)
        mrActions = []
        for i in range(1):
            #mrActions.append(mAgent.getAction())
            mrActions.append(0)
        
    #    dronePos, droneVel, droneCharge, dock, done= env.stepDrones(droneActions, docks)
    #    mrPos, mrVel,localArea = env.stepMobileRobs(mrActions)
        mrPos, mrVel, localArea, dronePos, droneVel, droneCharge, dock, reward, done = env.step(mrActions, droneActions, docks)
        #print(dronePos, mrPos, droneCharge)
        env.render()
        
        # check if any unexploredArea in local Area for static MobileRobot test
        if not np.any(localArea[0] == 50):
            break
    
        if env.checkClose():
            win_close_f = True
            break
        
        if any(done):
            break
    if win_close_f:
        break