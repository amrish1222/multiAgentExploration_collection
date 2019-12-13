	# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# Constants

RENDER_PYGAME = True

NUM_EPISODES = 100

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
timeStep=1

maxMobileRobVelocity=0.25


## Drone Constants

maxDroneVelocity= 0.5
dockingThreshold = maxDroneVelocity * timeStep * 1.0
MAX_CHARGE = 25


#DRONE_NEW_AREA_REWARD = 5
#DRONE_OLD_AREA_REWARD = -1
#DRONE_DISCHARGED_REWARD = -100
#
#DRONE_CURR_CHARGE_RWD = False
#EXCESS_RETURN_CHARGE_RWD = False
#DO_RETURN_POSSIBLE_RWD = True
#RETURN_POSSIBLE_RWD  = -50

## Area
GRID_SZ = maxDroneVelocity * timeStep * 1.0
G_MAIN = int(arenaWidth//GRID_SZ)
LOCAL_ENV_SZ = maxDroneVelocity * MAX_CHARGE * 2
G_LOCAL = int(LOCAL_ENV_SZ//GRID_SZ)
if G_LOCAL % 2 == 0:
    G_LOCAL+=1
PADDING = LOCAL_ENV_SZ/2
G_PADDING = int(PADDING//GRID_SZ)

G_RANGE_X = int(G_MAIN + 2* G_PADDING)
G_RANGE_Y = int(G_MAIN + 2* G_PADDING)
