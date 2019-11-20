#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:16:45 2019

@author: bala
"""

import numpy as np

class mobileRandomAgent:
    def __init__(self):
        self.prevAngle = 0
        self.momentum = 0.9
        self.currAngle = 0
        self.rand = 0
    
    def getAction(self):
        self.rand = np.random.uniform(-180,180)
        newAngle = self.prevAngle + (1-self.momentum) * self.rand
        if newAngle < 0:
            self.currAngle = 360 - newAngle
        else:
            self.currAngle = newAngle
        self.currAngle = self.currAngle % 360
        action = int(self.currAngle / 90)
        self.prevAngle = self.currAngle
        return action


if __name__ == "__main__":
    for mom in [0.9, 0.5, 0.9]:
        act =[]
        m = mobileRandomAgent()    
        m.momentum = mom
        for i in range(300):
            act.append(m.getAction())
        print(f"{mom} :: {act.count(0)} :: {act.count(1)} :: {act.count(2)} :: {act.count(3)}")
        