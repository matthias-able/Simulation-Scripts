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
