# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adhesh
# author Bala Murali
# Copyright
# brief Mobile robot class containing state, methods and history of the drone

import numpy as np

class MobileRobot:
    def __init__(self):
        self.curPos = np.array([500,500])
        self. curVel = np.array([0,0])
        self.size = 2 # mobile robot square side
        self.tourTaken = [] # list of positions that the MR has taken
        self.resourcePt = [] # detected resource collection pts
        self.maxVelocity = 0.05 # m/s
        
    def setParams(self, vel, dock):
        self.curVel = vel * self.maxVelocity
    
    def updatePos(self, parentPos, timeStep):
        if self.isDocked:
            newPosition = parentPos
        else:
            newPosition = self.curPos + self.curVel * timeStep
        self.curPos = newPosition
    
    def updateTour(self):
        if len(self.tourTaken) > 0:
            if not self.tourTaken[-1] == self.curPos:
              self.tourTaken.append(self.curPos)
              
    def getState(self):
        return self.curPos, self.curVel, self.tourTaken