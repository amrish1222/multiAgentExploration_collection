# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adhesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

from drone import Drone
from mobile_robot import MobileRobot

import numpy as np

class Env:
    def __init__(self, numDrones, numMobileRobs):
        self.drones = self.initDrones(numDrones)
        self.mobilerobots = self.initMobileRobs(numMobileRobs)
        self.numCollectionPts = 20
        self.areaLength = 20 # in meters
        self.collectionPts = self.genCollectionPts(self.numCollectionPts)
        self.dronesState = self.getDronesStates
        self.mobileRobsState = self.getMobileRobsStates
        
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
    
    def genCollectionPts(self, n):
        pts = np.random.randint(n,2)
        pts = self.areaLength * pts
        return pts
    
    def getDronesStates(self):
        states = []
        for drone in self.drones:
            states.append(drone.getState)
        return states
            
    def getMobileRobsStates(self):
        states = []
        for mr in self.mobilerobots:
            states.append(mr.getState)
        return states
            
    def stepDrones(self, actions, isdocks):
        # have to decide on the action space
        # waypoints or velocity
        # 0 - no motion
        # 1 - up
        # 2 - left
        # 3 - down
        # 4 - right
        for action, drone, isdock in zip(actions, self.drones, isdocks):
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
            drone.setParams(vel,isdock)
            
    
    def stepMobileRobs(self, actions):
        # 0 - no motion
        # 1 - up
        # 2 - left
        # 3 - down
        # 4 - right
        for action, mobileRobot in zip(actions, self.mobilerobots):
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
            mobileRobot.setParams(vel)
    
    def render(self):
        pass
    
    
    
        