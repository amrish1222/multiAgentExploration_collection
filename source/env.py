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
from constants import *

class Env:
    def __init__(self, numDrones, numMobileRobs):
        self.drones = self.initDrones(numDrones)
        self.mobilerobots = self.initMobileRobs(numMobileRobs)
        self.numCollectionPts = 20
        self.areaLength = 20 # in meters
        self.timeStep = timeStep
        self.collectionPts = self.genCollectionPts(self.numCollectionPts)

        #CONSTANTS
        self.screen_width=screenWidth
        self.screen_height=screenHeight
      
        #MAIN LOOP
        self.display=Render(len(self.drones),len(self.mobilerobots),self.drones,self.mobilerobots,self.collectionPts)
      
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
    
    def m_to_pix(self,x):
        return (self.screen_width/arenaWidth)*x[0],(self.screen_height/arenaHeight)*x[1]

    def genCollectionPts(self,n):
        resource_list=[]
        for i in range(0,n):
          resource_list.append((random.randint(0, arenaWidth),random.randint(0, arenaHeight)))
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

    def checkClose(self):
        return self.display.check()

    def render(self):
        self.display.render(self.drones,self.mobilerobots)
            


