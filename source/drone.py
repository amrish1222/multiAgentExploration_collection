# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Drone class containing state, methods and history of the drone

import numpy as np
from constants import *
np.set_printoptions(precision=3, suppress=True)
class Drone:
    def __init__(self):
        self.curPos = np.array([arenaWidth/2,arenaHeight/2])
        self.curVel = np.array([0,0])
        self.size = 0.4 # Diameter of the drone
        self.tourTaken = [] # list of positions that the drone has taken
        self.isDocked = False # whether the drone is docked to the mobile robot
        self.maxCharge = MAX_CHARGE # charge capacity in seconds
        self.currentCharge = MAX_CHARGE # current Charge in seconds
        self.parentPos = [0,0] # parent mobile robot position
        self.dockingThreshold = dockingThreshold # docking range from center of drone
        self.chargeTimeFactor = 2 # charge = time * chargeTimeFactor
        self.instantCharge = True
        self.maxVelocity = maxDroneVelocity # m/s
        
    def setParams(self, vel, dock):
        self.curVel = vel * self.maxVelocity
        distFromParent = np.linalg.norm(self.curPos - self.parentPos)
        if dock:
            if distFromParent < self.dockingThreshold:
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
        self.curPos = np.round(newPosition,3)
    
    
    def updateCharge(self, timeStep):
        self.currentCharge -= timeStep
        
        ## to use dock!
        if self.currentCharge >= self.maxCharge and self.isDocked:
            self.isDocked = False
        if self.isDocked:
            self.currentCharge += timeStep * self.chargeTimeFactor
        
        distFromParent = np.linalg.norm(self.curPos - self.parentPos)
        if self.instantCharge and (distFromParent < self.dockingThreshold):
            self.currentCharge = self.maxCharge
    
    def updateTour(self):
        if len(self.tourTaken) > 0:
            #print((self.tourTaken[-1] == self.curPos).all())
            if not (self.tourTaken[-1] == self.curPos).all():
                self.tourTaken.append(self.curPos)
                #print(len(self.tourTaken))
        else:
            self.tourTaken.append(self.curPos)

    
    def getState(self):
        if self.isDocked:
            time2release = max(0,self.maxCharge - self.currentCharge * self.chargeTimeFactor)
        else:
            time2release = 0
        return self.curPos, self.curVel, self.tourTaken, round(self.currentCharge, 3), self.isDocked, time2release
        
        
