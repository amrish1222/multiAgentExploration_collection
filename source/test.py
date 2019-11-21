# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Main Run file

from drone import Drone
from mobile_robot import MobileRobot
from env import Env
import numpy as np
import pygame
import time 
from mobileAgent import mobileRandomAgent

env = Env(2, 1)

mAgent = mobileRandomAgent()
dAgent = [mobileRandomAgent(), mobileRandomAgent()]

while True:
    droneActions = []
    docks = []
    for i in range(2):
        droneActions.append(dAgent[i].getAction())
        docks.append(False)
    mrActions = []
    for i in range(1):
        mrActions.append(mAgent.getAction())
    
    mrPos, mrVel = env.stepMobileRobs(mrActions)
    dronePos, droneVel, droneCharge, dock, done= env.stepDrones(droneActions, docks)
    env.render()
    print(dronePos[0], droneCharge)
    
    if env.checkClose() or done:
        break