import numpy as np
import matplotlib.pyplot as plt
from random import random


#grid size, time steps, number particles
N_grid= 50
N_t = 500
N_p = 25

#grid spacing
dx = 10./(N_grid-1)

#time spacing
dt = 0.1


#generate particles
#radius
x = np.zeros((N_t, N_p))
rj = 1
rk = 1

def random_choice_noreplace(m,n, axis=-1):
    # m, n are the number of rows, cols of output
    return np.random.rand(m,n).argsort(axis=axis)

ran = random_choice_noreplace(1,N_g)


for p in range(0,N_p):                    #particle
    for i in range(0,N_grid):             #x
        for j in range(0,N_grid):         #y
            x[0,p, i,j] = ran[i]

                





#potential

epsilon = 0.1

def pot(r):
    if r > rj + rk:
        return 0
    else:
        return .5 * epsilon * (1- r/(rj+rk))**2
for i

    for j

kraft - nable pot
m=1

#Verlet algorithm
for t in range(1,N_t-1):                    #time
    print(t)                                #particle
    for p in range(0,N_p):                  #x
        for i in range(1,N_grid-1):         #y
            for j in range(1,N_grid-1):
                #solve diff eq
                x[t+1,p, i,j] = 2*x[t,p,i,j] - x[t-1,p,i,j] + dt**2/m * nabla (pot(x[t,p, i,j]))

                #Boundary xonditions
                x[t+1,p, N_grid,j] = x[t+1,p, 0,j]
                x[t+1,p, i,N_grid] = x[t+1,p, i,0]

                x[t+1,p, 0,j] = x[t+1,p, N_grid,j]
                x[t+1,p, i,0] = x[t+1,p, i,N_grid]


#plotting??

#xx,yy = np.meshgrid(x,y)





























    


