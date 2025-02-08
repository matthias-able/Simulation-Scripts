'''
This code simulates the dynamics of an epidemic using the SIR (Susceptible-Infected-Recovered) model. It models the spread of an infectious disease in a population and tracks how the number of susceptible, infected, and recovered individuals change over time.
Key Functionality:

    SIR Model:
        The SIR model is a compartmental model in epidemiology that divides the population into three categories:
            S: susceptible individuals who can contract the disease.
            I: infected individuals who can spread the disease.
            R: recovered individuals who are no longer susceptible and do not contribute to the spread of the disease.

    Model Parameters:
        gamma (recovery rate): The fraction of infected individuals who recover each week. A larger gamma means quicker recovery.
        beta (contact rate * infection probability): Represents the rate at which an infected person contacts a susceptible individual and transmits the disease.
        time: The number of weeks (steps) the simulation will run.

    Initial Conditions:
        The initial number of infected (I[0]) is set to a small fraction of the population (N/1000), and the recovered individuals (R[0]) are set to 55% of the population.
        The number of susceptible individuals is calculated as the remaining population after subtracting the initially infected and recovered individuals.

    Differential Equations:
        The changes in susceptible (dS) and infected (dI) individuals are governed by two differential equations:
            dS = - β * I * S / N: The rate of change of susceptible individuals is proportional to the number of infected and susceptible individuals.
            dI = β * I * S / N - γ * I: The rate of change of infected individuals is determined by new infections and recoveries.

    Numerical Solution (Riemann Integral):
        The changes in the compartments (S, I, and R) are calculated using Euler's method, which approximates the solution to the differential equations by discretizing time (dt = 1 week).
        The state of the system is updated iteratively for each time step, and the result is stored in arrays S, I, and R.

    Visualization:
        The results of the simulation are plotted using matplotlib to visualize the time evolution of the S, I, and R populations.
        The plot is labeled with appropriate titles and axes, and it is saved as an image (SIR.png).
'''

import matplotlib.pyplot as plt
import numpy as np

gamma = .1 # recovery time (2 weeks)
beta = 1   # contact rate * infection probability, beta/gamma = R wert
time = 100 # *dt = weeks(recovery time)

S, I, R = np.zeros(time), np.zeros(time), np.zeros(time)
N = 80000000

I[0] = N/1000
R[0] = N*.55
dt = 1
S[0] = N - I[0] - R[0]


def dS (I, S):
    return - beta * I * S / N

def dI (I, S):
    return beta * I * S / N - gamma * I



#Riemannintegral
for t in range(1, S.size):
    S[t] = S[t-1] + dS(I[t-1], S[t-1]) *dt
    I[t] = I[t-1] + dI(I[t-1], S[t-1]) *dt


R = N - S - I



fig1, axes = plt.subplots()
axes.grid()
x = np.arange(0,S.size, 1)
axes.set_title('RIS model')

axes.plot(x, S , label = "S: susceptible" )
axes.plot(x,I , label = "I: infected" )
axes.plot(x, R , label = "R: either death or recovery" )

axes.set_xlabel('Time')
axes.set_ylabel('SIR population')
plt.legend()
#plt.show()
fig1.savefig('SIR.png',bbox_inches='tight', dpi=300)
