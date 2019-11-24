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
np.set_printoptions(precision=3, suppress=True)

env = Env(1, 1)

mAgent = mobileRandomAgent()
dAgent = [mobileRandomAgent(), mobileRandomAgent()]

while True:
    droneActions = []
    docks = []
    for i in range(2):
#        droneActions.append(dAgent[i].getAction())
        droneActions.append(1)
        docks.append(False)
    mrActions = []
    for i in range(1):
#        mrActions.append(mAgent.getAction())
        mrActions.append(2)
    
#    dronePos, droneVel, droneCharge, dock, done= env.stepDrones(droneActions, docks)
#    mrPos, mrVel,localArea = env.stepMobileRobs(mrActions)
    mrPos, mrVel, localArea, dronePos, droneVel, droneCharge, dock, done = env.step(mrActions, droneActions, docks)
    env.render()
#    plt.imshow(localArea[0])
#    plt.pause(0.001)
    time.sleep(0.01)
    if env.checkClose() or done:
        break