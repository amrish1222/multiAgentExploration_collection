# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adhesh
# author Bala Murali
# Copyright
# brief Drone class containing state, methods and history of the drone

import numpy as np

class Drone:
    def __init__(self):
        self.curPos = np.array([0,0])
        self. curVel = np.array([0,0])
        self.size = 0.4 # Diameter of the drone
        self.tourTaken = [] # list of positions that the drone has taken
        self.isDocked = False # whether the drone is docked to the mobile robot
        self.maxCharge = 50 # charge capacity in seconds
        self.currentCharge = 50 # current Charge in seconds
        self.parentPos = [0,0] # parent mobile robot position
        self.dockingThreshold = 0.1 # docking range from center of drone
        self.chargeTimeFactor = 2 # charge = time * chargeTimeFactor
        self.maxVelocity = 0.5 # m/s
        
    def setParams(self, vel, dock):
        self.curVel = vel * self.maxVelocity
        distFromParent = np.linalg.norm(self.curPos - self.parentPos)
        if dock and distFromParent < self.dockingThreshold:
            self.isDocked = True
        else:
            print("Can not dock: out of dock range")
            
    def updateState(self, parentPos, timeStep):
        self.updatePos(parentPos,timeStep)
        self.updateCharge(timeStep)
        self.updateTour()
    
    def updatePos(self, parentPos, timeStep):
        self.parentPos = parentPos
        if self.isDocked:
            newPosition = parentPos
        else:
            newPosition = self.curPos + self.curVel * timeStep
        self.curPos = newPosition
    
    
    def updateCharge(self, timeStep):
        self.currentCharge += timeStep
        if self.currentCharge >= self.maxCharge and self.isDocked:
            self.isDocked = False
    
    def updateTour(self):
        if len(self.tourTaken) > 0:
            if not self.tourTaken[-1] == self.curPos:
              self.tourTaken.append(self.curPos)  
    
    def getState(self):
        time2release = self.maxCharge - self.currentCharge * self.chargeTimeFactor
        return self.curPos, self.curVel, self.tourTaken, self.isDocked, time2release
        
        
