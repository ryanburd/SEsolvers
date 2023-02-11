# This code creates a superposition of the first two states of the infinite
# potential well Schrodinger Equation.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

m = 1
hbar = 1
init = 0, 1
num = 1000
x = np.linspace(-1,1,num)
E = 5

# Sets up the Schrodinger Equation as a second order ODE. The potential energy
# is 0 inside the well and infinite outside the well, which cannot physically
# occur. So, the ODE is only solved for the length of the well with V(x) = 0.
def SE(psi, x, E):
    psi0 = psi[0]
    psi1 = psi[1]
    psi2 = -1.0*((2*m)/hbar**2)*(E)*psi0
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
    global zeros
    zeros = []
    # Determines which energies diverge towards 0 at the bounds.
    for i in range(E):
        if np.sign(wave_function(i)) != np.sign(wave_function(i+1)):
            z = brentq(wave_function, i, i+1)
            zeros.append(z)
    return zeros

# Creates a superposition of the first two wave functions by adding them
# together. The resulting wave function is plotted.
def superposition(energies):
    global superpsi
    superpsi = np.zeros([num,2])
    n = 1
    for i in energies:
        wave_function(i)
        superpsi = np.sum([superpsi,psi], axis=0)
        n = n+1
    plt.plot(x,superpsi[:,0], label='wave function')

# Creates the probability density of the superpositioned wave function and
# plots it.
def super_prob(superpsi):
    super_squared = abs(superpsi[:,0]**2)
    plt.plot(x,super_squared, label='probability')

superposition(allowed_energies(E)[:2])
super_prob(superpsi)

plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('$\psi_1(x)+\psi_2(x)$')
plt.show()