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

env = Env(2, 1)

while True:
	droneActions = []
	docks = []
	for i in range(2):
		droneActions.append(np.random.randint(1,5))
		docks.append(False)
	mrActions = []
	for i in range(1):
		mrActions.append(np.random.randint(1,5))

	print("droneActions:", droneActions)
	print("mrActions:", mrActions)
	env.stepMobileRobs(mrActions)
	env.stepDrones(droneActions, docks)
	env.render()
	#time.sleep(0.1)

	if env.checkClose():
		break