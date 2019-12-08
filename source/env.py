# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

import numpy as np
import pygame
import random
import copy

from constants import *
from drone import Drone
from mobile_robot import MobileRobot
from render import Render

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self, numDrones, numMobileRobs):
        self.drones = self.initDrones(numDrones)
        self.mobilerobots = self.initMobileRobs(numMobileRobs)
        self.numCollectionPts = 20
        self.areaLength = 20 # in meters
        self.timeStep = timeStep
        self.collectionPts = self.genCollectionPts(self.numCollectionPts)

        #CONSTANTS
        self.screen_width=screenWidth
        self.screen_height=screenHeight
      
        # Area coverage
        self.totalArea = self.initTotalArea()
        self.totalAreaWithDrone = np.copy(self.totalArea)
        
        #MAIN LOOP
        self.display=Render(len(self.drones),
                            len(self.mobilerobots),
                            self.drones,
                            self.mobilerobots,
                            self.collectionPts)
        
    def initTotalArea(self):
        # beyond = 0
        # unexplored = 50
        # explored = 255
        # drone pos = 100
        tarea = np.zeros((G_RANGE_X,G_RANGE_Y))
        tarea[G_PADDING:G_RANGE_X-G_PADDING ,
              G_PADDING:G_RANGE_Y-G_PADDING] = 50
        states = self.drones[0].getState()
        x,y = states[0]
        x = int(x//GRID_SZ)
        y = int(y//GRID_SZ)
        tarea[x+G_PADDING, y+G_PADDING] = 255
        return tarea
        
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
    
    def m_to_pix(self,x):
        return (self.screen_width/arenaWidth)*x[0],(self.screen_height/arenaHeight)*x[1]
    
    def m_to_grid(self,pt):
        x,y = pt
        x = int(x//GRID_SZ)
        y = int(y//GRID_SZ)
        return np.asarray([x, y])

    def genCollectionPts(self,n):
        resource_list=[]
        for i in range(0,n):
          resource_list.append((random.randint(1, arenaWidth-1),random.randint(1, arenaHeight-1)))
        return resource_list
      
    def reset(self):
        self.drones = self.initDrones(len(self.drones))
        self.mobilerobots = self.initMobileRobs(len(self.mobilerobots))
        self.collectionPts = self.genCollectionPts(self.numCollectionPts)
        self.totalArea = self.initTotalArea()
        self.totalAreaWithDrone = np.copy(self.totalArea)
        self.display.reset(self.drones, self.mobilerobots, self.collectionPts)
        return self.step([0]*len(self.mobilerobots),
                         [0]*len(self.drones),
                         [False]*len(self.drones))
        
    def getActionSpace(self):
        return [0,1,2,3,4]
    
    def getStateSpace(self):
        localArea = self.getLocalArea(self.mobilerobots[0])
        w, h = localArea.shape
        # descritize area
        stateSpaceSz = w * h
        # drone Pos
        stateSpaceSz += 2
        # Vel Rover
        stateSpaceSz += 2
        # rover pos
        stateSpaceSz += 2
        # charge
        stateSpaceSz += 1
        return stateSpaceSz, w, h, 2, 2, 2, 1
    
    def stepDrones(self, actions, docks):
        # have to decide on the action space
        # waypoints or velocity
        posOut = []
        curChargeDistOut = []
        velOut = []
        isDockOut = []
        done = []
        for drone, action, dock in zip(self.drones, actions, docks):
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
            drone.setParams(vel,dock)
            drone.updateState(self.mobilerobots[0].getState()[0], self.timeStep)
            curState = drone.getState()
            posOut.append(curState[0])
            velOut.append(curState[1])
            curChargeDistOut.append(curState[3])
            isDockOut.append(curState[4])
            done.append(curState[3] <= 0)
        return posOut, velOut, curChargeDistOut, isDockOut, done
            
    def stepMobileRobs(self, actions):
        posOut = []
        velOut = []
        for mr, action in zip(self.mobilerobots, actions):
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
            mr.setParams(vel)
            mr.updateState(self.timeStep)
            curState = mr.getState()
            posOut.append(curState[0])
            velOut.append(curState[1])
        return posOut, velOut
    
    def step(self, mrActions, droneActions, docks):
        mrPos, mrVel = self.stepMobileRobs(mrActions)
        dronePos, droneVel, droneCharge, dock, done = self.stepDrones(droneActions, docks)
        reward = self.getReward()
        self.updateArea()
        localArea = [self.getLocalArea(mr) for mr in self.mobilerobots]
        return [self.m_to_grid(i) for i in mrPos], \
                mrVel, \
                localArea, \
                [self.m_to_grid(i) for i in dronePos], \
                droneVel, \
                droneCharge, \
                dock, \
                reward, \
                done
                
    def checkClose(self):
        return self.display.check()

    def render(self):
        self.display.render(self.drones,self.mobilerobots, self.totalAreaWithDrone)
    
    def getLocalArea(self,mr):
        x, y = mr.getState()[0]
        x = int(x//GRID_SZ)
        y = int(y//GRID_SZ)
        s = int((G_LOCAL-1)/2)
        return self.totalAreaWithDrone[x+G_PADDING-s : x+G_PADDING+s+1,
                                       y+G_PADDING-s : y+G_PADDING+s+1]
            
    def updateArea(self):
        for drone in self.drones:
            x,y =drone.getState()[0]
            x = int(x//GRID_SZ)
            y = int(y//GRID_SZ)
            self.totalArea[x+G_PADDING, y+G_PADDING] = 255
            self.totalAreaWithDrone = np.copy(self.totalArea)
            self.totalAreaWithDrone[x+G_PADDING, y+G_PADDING] = 100 
        
    def getReward(self):
        reward = []
        for drone in self.drones:
            states = drone.getState()
            x,y = states[0]
            x = int(x//GRID_SZ)
            y = int(y//GRID_SZ)
            rem_charge = int(states[3]//GRID_SZ)
            l1_dist2par = int(states[-1]//GRID_SZ)
            if self.totalArea[x+G_PADDING, y+G_PADDING] == 50:
                # unexplored region => new area 
                reward.append(10)
            elif self.totalArea[x+G_PADDING, y+G_PADDING] == 255:
                # explored region => old area
                reward.append(-10)
            else:
                reward.append(0)
            if rem_charge <= 0:
                # penalize for die
                reward[-1] += -1000
               
            if (rem_charge - l1_dist2par*1.2) >= 0:
                # if inside charge radius 
                reward[-1] +=  1
            else:
                # if outside charge radius
                reward[-1] += -50
            if l1_dist2par <= 1 and rem_charge <=5:
                reward[-1] += 1000
                
        return reward
                


