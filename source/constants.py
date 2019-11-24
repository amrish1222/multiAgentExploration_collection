# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Constants

screenHeight = 1000
screenWidth = 1000
arenaWidth = 50
arenaHeight= 50

numDrone = 2
numMobileRobots = 1

BLACK=(0,0,0)
AQUA=(0,255,255)
GREEN_ALPHA=(0,255,0,100)
BLUE_ALPHA=(0,0,255,100)
timeStep=0.1

maxMobileRobVelocity=0.25


## Drone Constants

maxDroneVelocity= 0.5
dockingThreshold = maxDroneVelocity * timeStep * 2
MAX_CHARGE = 25

## Area
GRID_SZ = maxMobileRobVelocity * timeStep
G_MAIN = arenaWidth//GRID_SZ
LOCAL_ENV_SZ = maxDroneVelocity * MAX_CHARGE * 2
G_LOCAL = LOCAL_ENV_SZ//GRID_SZ
PADDING = LOCAL_ENV_SZ/2
G_PADDING = PADDING//GRID_SZ

G_RANGE_X = G_MAIN + 2* G_PADDING
G_RANGE_Y = G_MAIN + 2* G_PADDING
