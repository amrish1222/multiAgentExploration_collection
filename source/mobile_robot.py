# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Mobile robot class containing state, methods and history of the drone

import numpy as np
from constants import *

class MobileRobot:
    def __init__(self):
        self.curPos = np.array([10,10])
        self. curVel = np.array([0,0])
        self.size = 2 # mobile robot square side
        self.tourTaken = [] # list of positions that the MR has taken
        self.resourcePt = [] # detected resource collection pts
        self.maxVelocity = maxMobileRobVelocity # m/s
        
    def setParams(self, vel):
        self.curVel = vel * self.maxVelocity
    
    def updateState(self, timeStep):
        self.updatePos(timeStep)
        self.updateTour()

    def updatePos(self, timeStep):
        newPosition = self.curPos + self.curVel * timeStep
        self.curPos = np.round(newPosition,3)
    
    def updateTour(self):
        if len(self.tourTaken) > 0:
            if not self.tourTaken[-1] == self.curPos:
              self.tourTaken.append(self.curPos)
              
    def getState(self):
        return self.curPos, self.curVel, self.tourTaken