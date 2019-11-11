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
            
    def stepDrones(self):
        # have to decide on the action space
        # waypoints or velocity
        pass
    
    def stepMobileRobs(self):
        pass
    
    def render(self):
        pass
    
    
    
        