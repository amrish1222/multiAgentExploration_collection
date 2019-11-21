#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:16:45 2019

@author: bala
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)

class mobileRandomAgent:
    def __init__(self):
        self.prevAngle = 0
        self.momentum = 0.7
        self.currAngle = 0
        self.rand = 0
    
    def getAction(self):
        self.rand = np.random.uniform(-180,180)
        newAngle = self.prevAngle + (1-self.momentum) * self.rand
        if newAngle < 0:
            self.currAngle = 360 + newAngle
        else:
            self.currAngle = newAngle
        self.currAngle = self.currAngle % 360
        action = int(self.currAngle / 90)
        self.prevAngle = self.currAngle
        return action + 1


if __name__ == "__main__":
    for mom in [0.1, 0.5, 0.9]:
        act =[]
        m = mobileRandomAgent()    
        m.momentum = mom
        for i in range(300):
            act.append(m.getAction())
        print(f"Momentum: {mom} --> \n \
             Action 0: {act.count(0)}  \n\
              Action 1: {act.count(1)} \n\
              Action 2: {act.count(2)} \n\
              Action 3: {act.count(3)} \n\
              Action 4: {act.count(4)} \n")
        