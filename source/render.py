# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Render class for diplay

import pygame 
import numpy as np
import random
from constants import *
from drone import Drone
from mobile_robot import MobileRobot
np.set_printoptions(precision=3, suppress=True)

class Render:
    def __init__(self,numDrones,numMobileRobots,drone,mobile_robot,coll_pts,gridAreaWithDrone):
        self.drones=drone
        self.mobilerobots=mobile_robot
        self.collectionPts = coll_pts
        self.screen_width=screenWidth
        self.screen_height = screenHeight
        self.gridAreaWithDrone = gridAreaWithDrone
        #INIT
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        self.alpha_surface=pygame.Surface((self.screen_width,self.screen_height), pygame.SRCALPHA)
        self.caption()
        self.backg=self.background()
        
        #IMAGES
        self.drone_surface=self.drone_icon(numDrones)
        self.rover_surface=self.rover_icon()
        
        #FLAGS
        self.showGrid_f = False

    def m_to_pix(self,x):
        return int((self.screen_width/arenaWidth)*x[0]),int((self.screen_height/arenaHeight)*x[1])

    def caption(self):
        pygame.display.set_caption('Multi-Agent Explorer')
        icon=pygame.image.load('Images/icon.png')
        pygame.display.set_icon(icon)

    def background(self):
        background=pygame.image.load('Images/terrain.jpeg')
        background=pygame.transform.scale(background, (self.screen_width, self.screen_height))
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

    def drone_blit(self,drone_surface,x,y):
        self.screen.blit(drone_surface,(x,y))

    def rover_blit(self,x,y):
        self.screen.blit(self.rover_surface,(x,y))
    
    def resources_blit(self,pt):
        pygame.draw.circle(self.screen, AQUA, self.m_to_pix(pt), 5, 5) 

    def path_blit(self,path):
        for i in path:
			#self.m_to_pix(i)
            pygame.draw.circle(self.alpha_surface, GREEN_ALPHA, self.m_to_pix(i), 5, 5) 		

    def mob_path_blit(self,path):
        for i in path:
			#self.m_to_pix(i)
            pygame.draw.circle(self.alpha_surface, BLUE_ALPHA, self.m_to_pix(i), 5, 5) 		
    
    def gray(self, im):
        im = 255 * (im / im.max())
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
        return ret
    
    def area_blit(self, areaWithDrone):
        area_surf = self.gray(areaWithDrone[G_PADDING:-G_PADDING, G_PADDING:-G_PADDING])
        area_surf = pygame.surfarray.make_surface(area_surf)
        area_surf = pygame.transform.scale(area_surf, (self.screen_width, self.screen_height))
        self.screen.blit(area_surf, (0,0))
	
    def render(self,drones,mobilerobots, areaWithDrone):
        events = pygame.event.get()
        
        drone_surface=self.drone_surface
        rover_surface=self.rover_surface
        
        #INIT
        self.screen.fill(BLACK)
        		
        
        self.screen.blit(self.backg,(0,0))
        self.screen.blit(self.alpha_surface,(0,0))
        
        #RESOURCES
        for pt in self.collectionPts:
            self.resources_blit(pt)

        x_mob=self.m_to_pix(self.mobilerobots[0].getState()[0])[0]
        y_mob=self.m_to_pix(self.mobilerobots[0].getState()[0])[1]
        
        # if x_mob<0:
        # 	x_mob=0
        # elif x_mob>0:
        # 	x_mob=self.screen_width-self.rover_surface.get_size()[0]
        
        #ROVER
        self.rover_blit(x_mob,y_mob)
        self.mob_path_blit(self.mobilerobots[0].getState()[2])
        
        #DRONE
        for i in range(0,len(drones)):
            self.drone_blit(drone_surface[i],self.m_to_pix(self.drones[i].getState()[0])[0],self.m_to_pix(self.drones[i].getState()[0])[1])
            #print(f'drone{i}: {self.drones[i].getState()[2]}')
            
            self.path_blit(self.drones[i].getState()[2])

        #UPDATE
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.showGrid_f = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_d:
                    self.showGrid_f = False
        if self.showGrid_f:
            self.area_blit(areaWithDrone)
            
        pygame.display.update()
	
    def check(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                return True
        return False