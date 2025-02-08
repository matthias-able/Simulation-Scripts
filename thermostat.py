'''
Project: Molecular Dynamics Simulation of Particle Interactions
Developed a Python-based molecular dynamics simulation to model the motion and interactions of particles in a 2D box. Key features include:

    Particle Initialization: Randomly initialized particle positions and velocities within a defined boundary.

    Force Calculation: Implemented a pairwise force model to simulate particle interactions, including periodic boundary conditions to handle edge effects.

    Time Integration: Used a velocity Verlet algorithm to update particle positions and velocities over time, ensuring accurate dynamics.

    Visualization: Generated plots to visualize particle trajectories and final positions using matplotlib.

'''
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt


boxside = 10
R = .5
epsilon = 100
 
m = np.ones(30) 
#m = np.array([1, 1])
v = np.random.rand(len(m), 2) - 1/2
v*=10
#v = np.array([[0.0, 0.0], [0.0, 0.0]])
x = np.random.rand(len(m), 2) * boxside
#x = np.array([[1.0,50.0], [99, 50.0]]) 
boxcounter = np.zeros((len(m), 2))

print(np.sum(v, 0))


# definitions for Andersen themostat
"""
After a random time (e.g., chosen from a Poisson distribution P (τ ) ∝ exp(−ντ )) the velocity of
a randomly selected particles is set to a new value drawn from a Maxwell-Boltzmann distribution for each component j of the velocity.
"""
def poisson (tau):
	nu = 0.1
	return np.e**(-nu*tau)
	
def max_bolz_dist (v):
	m = 1
	kb = 1e-19
	T = 1
	return np.sqrt(m / (2*np.pi*kb*T)) * np.e**(-mv/(2*kb*T))


dt = 0.001
T = 1


def Fs(x):
    F = np.zeros((len(x), 2))
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            rvec = x[j] - x[i]
            overflow = np.where(rvec > boxside/2, 1, 0)
            overflow += np.where(rvec < -boxside/2, -1, 0)
            rvec -= overflow * boxside
            rabs = np.sqrt(np.sum(rvec**2))
            if rabs < 2*R:
                Fabs = (epsilon / (2*R)) * (1 - rabs / (2*R))
                Fvec = Fabs * rvec / rabs
                F[i] -= Fvec
                F[j] += Fvec
    return F

print("Start:\nx =\n{}\nv =\n{}\n".format(x, v))

a1 = []
a2 = []

for t in np.arange(0, T, dt):
    print(t)
    F = Fs(x)
    x += (v + F * dt / (2 *dt * m[:,np.newaxis])) * dt
    overflow = np.where(x > boxside, 1, 0)
    overflow += np.where(x < 0, -1, 0)
    boxcounter += overflow
    x -= overflow * boxside
    v += (Fs(x) + F) * dt / (2 * m[:,np.newaxis])
    a1.append(np.copy(x[:,0]))
    a2.append(np.copy(x[:,1]))
    
fig1=plt.plot(a1[0], a2[0], ".", linestyle="None")
fig2=plt.plot(a1[:], a2[:], ".", linestyle="None", markersize=0.5)
fig3=plt.plot(a1[-1], a2[-1], ".", linestyle="None")
#plt.savefig("testplot.jpg", dpi=300)
plt.show()
print("End:\nx =\n{}\nv =\n{}".format(x, v))


print(np.sum(v, 0))

    
