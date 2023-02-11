# This code calculates how far into the walls of the finite well the wave
# function is able to tunnel as a function of the potential energy outside the
# well.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

m = 1
hbar = 1
init = 0, 1
x = np.linspace(-2,2,1000)
E = 1

# Sets the potential energy of the system at the given value of x.
def V(x):
    if abs(x) <= 1:
        return 0 # Potential energy = 0 inside the well, -1<x<1
    else:
        return V0 # Potential energy = v0 outside the well

# Sets up the Schrodinger Equation as a second order ODE.
def SE(psi, x, E):
    psi0 = psi[0]
    psi1 = psi[1]
    psi2 = -1.0*((2*m)/hbar**2)*(E-V(x))*psi0
    return psi1, psi2

# Calculates the solution to the ODE at a given energy. Allowed energies result
# in wave functions.
def wave_function(E):
    global psi
    for i in x:
        psi = odeint(SE, init, x, args=(E,))
        psi = normalization()*psi
    return psi[-1,0]

# Normalizes the wave function so the total probability density = 1.
def normalization():
    psi_squared = psi[:,0]**2
    probability = np.trapz(psi_squared, x)
    N = 1/probability
    return N

# Determines which energies are allowed by the boundary conditions as physically
# representative solutions to the ODE.
def allowed_energies(E):
    zeros = []
    n = 1
    # Determines which energies diverge towards 0 at the bounds.
    for i in range(E):
        if np.sign(wave_function(i)) != np.sign(wave_function(i+1)):
            z = brentq(wave_function, i, i+1)
            zeros.append(z)
            n = n+1
    return zeros

# Calculates the position 'x' at which the wave function decays to a very small
# chosen value of psi(x) to represent how far into the walls of the finite well
# the wave function is able to tunnel.
def depth(psi):
    func = np.zeros([500,2])
    func[:,1] = x[500:]
    func[:,0] = psi
    m = 0
    for i in func[:,0]:
        if i <= 0.00001:
            return func[m,1]
            break
        m = m+1
# Plots the distance past the walls of the finite well the wave function is
# able to tunnel as a function of the potential energy outside the well. For
# each potential energy,only the first energy level wave function is considered.
xaxis = []
yaxis = []
V0 = 20
while V0 <= 40:
    n = 1
    xaxis.append(V0)
    for i in allowed_energies(E):
        print('V0=%d: E%d=%.3f'%(V0,n,i))
        distance = depth(psi[500:,0])-1
        yaxis.append(distance)
        print('Distance past finite wall = %.3f'%distance)
        n = n+1
    V0 = V0+5

plt.plot(xaxis,yaxis)
plt.xlabel('$V_0$')
plt.ylabel('Distance past finite wall')
plt.show()
