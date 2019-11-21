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
	def __init__(self,numDrones,numMobileRobots,drone,mobile_robot,coll_pts):
		self.drones=drone
		self.mobilerobots=mobile_robot
		self.collectionPts = coll_pts
		self.screen_width=screenWidth
		self.screen_height=screenHeight
		#INIT
		pygame.init()
		self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
		self.alpha_surface=pygame.Surface((self.screen_width,self.screen_height), pygame.SRCALPHA)
		self.caption()
		self.backg=self.background()
    
		#IMAGES
		self.drone_surface=self.drone_icon(numDrones)
		self.rover_surface=self.rover_icon()

		#MAIN LOOP
		#self.render(self.drone_surface,self.rover_surface,self.drones,self.mobilerobots,numDrones)

	# def genCollectionPts(self):
	#   resource_list=[]
	#   for i in range(0,self.numCollectionPts):
	#     resource_list.append((random.randint(0, 1000-self.rover_surface.get_size()[0]),random.randint(0, 1000-self.rover_surface.get_size()[1])))
	#   return resource_list

	def m_to_pix(self,x):
		return int((self.screen_width/arenaWidth)*x[0]),int((self.screen_height/arenaHeight)*x[1])

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

	def render(self,drones,mobilerobots):
		running = True
		drone_surface=self.drone_surface
		rover_surface=self.rover_surface

		#INIT
		self.screen.fill(BLACK)
		

		self.screen.blit(self.backg,(0,0))
		self.screen.blit(self.alpha_surface,(0,0))

		#RESOURCES
		for pt in self.collectionPts:
			self.resources_blit(pt)
		#print(self.collectionPts)

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
		pygame.display.update()

	def check(self):
		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				return True
		return False
