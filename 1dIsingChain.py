import numpy as np
import matplotlib.pyplot as plt

T = 100
J = 1
N = 50
steps = 1

kb = 1


def metropolis(C, M, T):
    indices = np.random.randint(0, N)
    delta = 0
    for i in range(-1,1,1):
        x = indices[0] + i
        if x == N:
            x = 0
        delta += C[x]
    delta *= J * (2) * C[indices[0]]


    beta = 1 / (kb * T)
    f = np.exp(-beta*delta)
    if f >= np.random.rand():
        C[indices[0]] *= -1
        M += 2 * C[indices[0]]
    return C, M


def mag(T):
    #C = np.random.choice((-1,1), (N,N))
    C = np.ones(N)
    M = np.sum(C)
    for t in range(N**2):
        C, M = metropolis(C, M, T)

    Ms = []
    for t in range(steps):
        Ms.append(M)
        C, M = metropolis(C, M, T)
    
    return np.abs(np.sum(np.array(Ms)))/(N**2) / steps, np.sum(np.array(Ms)**2)/N**4 / steps
    print("<abs(Magnetization)>: {}, <Magnetization**2>: {}".format(np.sum(np.abs(M/(N**2))) / steps, (np.sum(M**2)/N**2)**2 / steps)) 
    
mag(100)

x = []
y= []
for T in np.linspace(0.05, 20, 100):
    x.append(T)
    y.append(mag(T))

fig, ax = plt.subplots()
ax.plot(x, y)
#ac.plot(range(1,200), mag(range(1,200)))
plt.savefig("plot.png", dpi=300)
plt.show()
