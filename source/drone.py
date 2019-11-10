# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adhesh
# author Bala Murali
# Copyright
# brief Drone class containing state, methods and history of the drone

import numpy as np


class Drone:
    def __init__(self):
        self.curPos = [0,0]
        self. curVel = [0,0]
        self.size = 0.4 # Diameter of the drone
        self.tourTaken = [] # list of positions that the drone has taken
        self.isDocked = False # whether the drone is docked to the mobile robot
        self.maxCharge = 50 # charge capacity in seconds
        self.currentCharge = 50 # current Charge in seconds
        self.parentPos = [0,0] # parent mobile robot position
        self.dockingThreshold = 0.1 # docking range from center of drone
        
    def setParams(self, vel, dock):
        self.curVel = vel
        if np.linalg.norm(self.curPos-self.parentPos) < self.dockingThreshold:
            self.isDocked = True
        else:
            print("Can not dock: out of dock range")
            
    def updateState(self, parentPos):
        pass
    
    def updatePos(self):
        pass
    
    def updateCharge(self):
        pass
    
    def updateTour(self):
        pass
    
    def getState(self):
        time2dock = self.maxCharge - self.currentCharge
        return self.curPos, self.tourTaken, self.isDocked, time2dock
        
        
