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

env = Env(2, 1)
droneActions = []
docks = []
while 1:
	for i in range(2):
		droneActions.append(np.random.randint(0,4))
		docks.append(False)
	mrActions = []
	for i in range(1):
		mrActions.append(np.random.randint(0,4))

	env.stepMobileRobs(mrActions)
	env.stepDrones(droneActions, docks)
	env.render()
	break