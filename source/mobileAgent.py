#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:16:45 2019

@author: bala
"""

import numpy as np
from constants import *
np.set_printoptions(precision=3, suppress=True)

class mobileRandomAgent:
    def __init__(self,defined=False):
        self.prevAngle = 0
        self.momentum = 1
        self.currAngle = 0
        self.rand = 0
        self.defined=defined
    
    def getAction(self,mrPos):
        self.rand = np.random.uniform(-180,180)
        currPos=mrPos
        if self.defined:
            newAngle = self.prevAngle + (1-self.momentum) * self.rand
        else:
            newAngle = self.prevAngle + (1-self.momentum)
        if newAngle < 0:
            self.currAngle = 360 + newAngle
        else:
            self.currAngle = newAngle
        if currPos[0]==0:
            self.currAngle=180
        if currPos[1]==0:
            self.currAngle=270
        if currPos[0]==arenaWidth:
            self.currAngle=0
        if currPos[1]==arenaHeight:
            self.currAngle=90

            
        self.currAngle = self.currAngle % 360
        action = int(self.currAngle / 90)
        print(action)
        #print(action)
        self.prevAngle = self.currAngle

        return action + 1

# class mobileAgent:
#     def __init__(self):
#         self.prevAngle = 0
#         self.momentum = 1
#         self.currAngle = 0
#         self.rand = 0

#     def getAction(self):
#         newAngle = self.prevAngle + (1-self.momentum)
#         if newAngle < 0:
#             self.currAngle = 360 + newAngle
#         else:
#             self.currAngle = newAngle
#         self.currAngle = self.currAngle % 360
        
#         action = int(self.currAngle / 90)
#         self.prevAngle = self.currAngle
#         return action + 1




if __name__ == "__main__":
    
    act =[]
    m=mobileRandomAgent()
    
    for i in range(1000):
        act.append(m.getAction(mrPos))


    print(f"Momentum: {m.momentum} --> \n \
        Action 0: {act.count(0)}  \n\
        Action 1: {act.count(1)} \n\
        Action 2: {act.count(2)} \n\
        Action 3: {act.count(3)} \n\
        Action 4: {act.count(4)} \n")
        
        # m = mobileRandomAgent()    
        # m.momentum = mom
        # for i in range(300):
        #     act.append(m.getAction())
        # print(f"Momentum: {mom} --> \n \
        #      Action 0: {act.count(0)}  \n\
        #       Action 1: {act.count(1)} \n\
        #       Action 2: {act.count(2)} \n\
        #       Action 3: {act.count(3)} \n\
        #       Action 4: {act.count(4)} \n")
        # 