# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Constants

NUM_EPISODES = 10

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
GRID_SZ = maxDroneVelocity * timeStep * 1.2
G_MAIN = int(arenaWidth//GRID_SZ)
LOCAL_ENV_SZ = maxDroneVelocity * MAX_CHARGE * 2
G_LOCAL = int(LOCAL_ENV_SZ//GRID_SZ)
if G_LOCAL % 2 == 0:
    G_LOCAL+=1
PADDING = LOCAL_ENV_SZ/2
G_PADDING = int(PADDING//GRID_SZ)

G_RANGE_X = int(G_MAIN + 2* G_PADDING)
G_RANGE_Y = int(G_MAIN + 2* G_PADDING)
