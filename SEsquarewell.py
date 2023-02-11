# This code solves the 1D Schrodinger Equation for a particle in a finite square
# well. The potential energy outside of the well is set at V0. A range of
# energies are tested to determine which energies are allowed by the boundary
# conditions.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

m = 1
hbar = 1
init = 0, 1
x = np.linspace(-2,2,1000)
E = 15
V0 = 20

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

# Normalizes the wave function so that the total probability density = 1.
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
    # Determines which energies diverge towards 0 at the bounds and plots the
    # wave functions for the allowed energies.
    for i in range(E):
        if np.sign(wave_function(i)) != np.sign(wave_function(i+1)):
            z = brentq(wave_function, i, i+1)
            zeros.append(z)
            plt.plot(x, psi[:,0], label='$E_%d$ = %.3f'%(n,z))
            n = n+1
    return zeros

print('The eigenvalues of the finite potential well, where V0 = 20 outside of the well, are:')
n = 1
for i in allowed_energies(E):
    print('E%d = %.3f'%(n,i))
    n = n+1

plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.show()
