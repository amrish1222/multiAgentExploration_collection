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
import random
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
    
    
    #CONSTANTS
    self.screen_width=1000
    self.screen_height=1000
    
    #INIT
    pygame.init()
    self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
    self.caption()
    self.backg=self.background()
    
    #IMAGES
    self.drone_surface=self.drone_icon(numDrones)
    self.rover_surface=self.rover_icon()
    self.collectionPts = self.genCollectionPts(self.numCollectionPts,self.rover_surface)

    #MAIN LOOP
    self.render(self.screen,self.backg,self.drone_surface,self.rover_surface,self.drones,self.mobilerobots,numDrones,self.numCollectionPts,self.collectionPts)
      
      
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

  def genCollectionPts(self, n,rover_surface):
    resource_list=[]
    for i in range(0,n):
      resource_list.append((random.randint(0, 1000-rover_surface.get_size()[0]),random.randint(0, 1000-rover_surface.get_size()[1])))
    return resource_list
  
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
    drone_icon=pygame.transform.scale(drone_icon, (20, 20))
    num_of_drones=n
    drone_list=[]
    for i in range(0,num_of_drones):
        drone_list.append(pygame.image.load('Images/drone.ico'))
    return drone_list
  
  def rover_icon(self):
    robot_icon=pygame.image.load('Images/rover2.ico')
    return robot_icon

  def drone_blit(self,screen,drone_surface,x,y):
    self.screen.blit(drone_surface,(x,y))

  def rover_blit(self,screen,rover_surface,x,y):
    self.screen.blit(rover_surface,(x,y))

  def resources_blit(self,resource_list):
    AQUA=(0,255,255)
    pygame.draw.circle(self.screen, AQUA, (resource_list[0],resource_list[1]), 5, 5) 

  def render(self,screen,background,drone_surface,rover_surface,drones,mobilerobots,num_of_drones,num_of_resources,resource_list):
    running = True
    BLACK=(0,0,0)
    
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      #INIT
      self.screen.fill(BLACK)
      self.screen.blit(background,(0,0))

      #RESOURCES
      for i in range(0,num_of_resources):
        self.resources_blit(resource_list[i])

      #ROVER
      self.rover_blit(self.screen,rover_surface,self.mobilerobots[0].getState()[0][0],self.mobilerobots[0].getState()[0][1])

      #DRONE
      for i in range(0,num_of_drones):
        self.drone_blit(self.screen,drone_surface[i],self.drones[i].getState()[0][0], self.drones[i].getState()[0][1])
      
      #UPDATE
      pygame.display.update()