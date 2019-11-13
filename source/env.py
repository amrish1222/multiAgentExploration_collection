# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adhesh
# author Bala Murali
# Copyright
# brief Environment class for the simulation

from drone import Drone
from mobile_robot import MobileRobot

import numpy as np
import pygame

# screen_width=1000
# screen_height=1000
# BLACK=(0,0,0)
# GREEN=(0,255,0)

class Env:
    def __init__(self, numDrones, numMobileRobs):
        self.drones = self.initDrones(numDrones)
        self.mobilerobots = self.initMobileRobs(numMobileRobs)
        self.numCollectionPts = 20
        self.areaLength = 20 # in meters
        self.collectionPts = self.genCollectionPts(self.numCollectionPts)
        
        #CONSTANTS
        self.screen_width=1000
        self.screen_height=1000
        # self.BLACK=(0,0,0)
        
        #INIT
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        self.caption()
        self.backg=self.background()
        
        #IMAGES
        self.drone_surface=self.drone_icon(numDrones)
        self.rover_surface=rover_icon(numMobileRobs)

        #MAIN LOOP
        self.render(screen,backg,drone_surface,rover_surface,drones,mobilerobots,numDrones)
        
        
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
        pts = np.random.rand(n,2)
        pts = self.areaLength * pts
        return pts
    
    def stepDrones(self):
        # have to decide on the action space
        # waypoints or velocity
        pass
    
    def stepMobileRobs(self):
        pass
    
    def caption(self):
        pygame.display.set_caption('Multi-Agent Explorer')
        icon=pygame.image.load('Images/icon.png')
        pygame.display.set_icon(icon)

    def background(self):
        background=pygame.image.load('Images/terrain.jpeg')
        background=pygame.transform.scale(background, (1400, 1000))
        return background

    def drone_icon(self,n):
        drone_icon=pygame.image.load('Images/drone.ico')
        drone_icon=pygame.transform.scale(drone_icon, (10, 10))
        num_of_drones=n
        drone_list=[]
        for i in range(0,num_of_drones):
            drone_list.append(pygame.image.load('Images/drone.ico'))
        return drone_list
    
    def rover_icon(self,n):
        robot_icon=pygame.image.load('Images/rover2.ico')
        return robot_icon

    def drone_blit(self,screen,drone_surface,x,y):
        self.screen.blit(drone_surface[i],(x,y))

    def rover_blit(self,screen,rover_surface,x,y):
        self.screen.blit(rover_surface,(x,y))

    def render(self,screen,background,drone_surface,rover_surface,drones,mobilerobots,n):
        running = True
        BLACK=(0,0,0)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill(BLACK)
            self.screen.blit(self.background,(0,0))
            self.rover_blit(self.screen,self.rover_surface,self.mobilerobots.getState[0][0],self.mobilerobots.getState[0][1])

            for i in range(0,n):
                self.drone_blit(self.screen,self.drone_surface[i],self.drones[i].getState[0][0], self.drones[i].getState[0][1])
            pygame.display.update()
    
        