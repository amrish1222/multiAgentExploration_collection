# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

from drone import Drone
from mobile_robot import MobileRobot
from render import Render
import numpy as np
import pygame
import random
# screen_width=1000
# screen_height=1000
# BLACK=(0,0,0)
# GREEN=(0,255,0)

class Env:
  def __init__(self, numDrones, numMobileRobs):
    self.drones = self.initDrones(numDrones)
    self.mobilerobots = self.initMobileRobs(numMobileRobs)
    self.numCollectionPts = 20
    self.areaLength = 20 # in meters
    self.timeStep = 0.1
    
    #CONSTANTS
    self.screen_width=1000
    self.screen_height=1000
  
    #MAIN LOOP
      
  def initDrones(self, n):
    drones = []
    for i in range(0,n):
        drones.append(Drone())
    return drones

  def initMobileRobs(self, n):
    mRobs = []
    for i in range(0,n):
        mRobs.append(MobileRobot())
    return mRobs

  def genCollectionPts(self, n,rover_surface):
    resource_list=[]
    for i in range(0,n):
      resource_list.append((random.randint(0, 1000-rover_surface.get_size()[0]),random.randint(0, 1000-rover_surface.get_size()[1])))
    return resource_list
  
  def stepDrones(self, actions, docks):
    # have to decide on the action space
    # waypoints or velocity
    for drone, action, dock in zip(self.drones, actions, docks):
    	vel = np.array([0,0])
    	if action == 0:
    		pass
    	elif action == 1:
    		vel[1] = 1
    	elif action == 2:
    		vel[0] = -1
    	elif action == 3:
    		vel[1] = -1
    	elif action == 4:
    		vel[0] = 1
    	drone.setParams(vel,dock)
    	drone.updateState(self.mobilerobots[0].getState()[0], self.timeStep)

  def stepMobileRobs(self, actions):
    for mr, action in zip(self.mobilerobots, actions):
    	vel = np.array([0,0])
    	if action == 0:
    		pass
    	elif action == 1:
    		vel[1] = 1
    	elif action == 2:
    		vel[0] = -1
    	elif action == 3:
    		vel[1] = -1
    	elif action == 4:
    		vel[0] = 1
    	mr.setParams(vel)
    	mr.updateState(self.timeStep)

  def render(self):
    Render(len(self.drones),len(self.mobilerobots),self.drones,self.mobilerobots,self.numCollectionPts)


