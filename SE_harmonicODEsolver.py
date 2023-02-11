# This code solves the Schrodinger Equation with an an harmonic oscillator
# potential using an ODE solver.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Define variables.
n_max = 9
bound = 5
x = np.linspace(-1*bound,bound,1000)

# Define constants.
m = 1
hbar = 1
k = 1
k_prime = 0.25
init = 0, 1

# Generate Schrodinger Equation.
def SE(psi, x, E):
    psi0 = psi[0]
    psi1 = psi[1]
    psi2 = -1*((2*m)/hbar**2)*(E-V(x))*psi0
    return psi1, psi2

# Solve Schrodinger Equation to obtain normalized wave function.
def wave_function(E):
    sol = odeint(SE, init, x, args=(E,))
    psi = sol[:,0]
    psi_squared = psi**2
    norm_psi = normalize(psi_squared)*psi
    return norm_psi
    
# Normalize wave functions.
def normalize(psi_squared):
    probability = np.trapz(psi_squared, x)
    N = 1/probability
    return N

# Set potential energy at 'x'.
def V(x):
    potential = (0.5*k*x**2)-((1/6)*k_prime*x**3)
    return potential

# Find the value of psi at the positive bound of x (needed by brentq).
def psi_at_bound(E):
    norm_psi = wave_function(E)
    psi_at_bound = norm_psi[-1]
    return psi_at_bound

# Determines if the wave function is bounded by the potential well as x goes to
# infinity. If E <= V(x), the function is bounded. If E > V(x), the function is
# unbounded.
def bounded(E, n):
    if E <= V(bound):
        print('psi%d(x) is bounded as x goes to positive infinity.'%n)
    elif E > V(bound):
        print('psi%d(x) becomes unbounded as x goes to positive infinity.'%n)

# Calculate eigenvalues, i.e. allowed energies.
def eigenvalues(n_max):
    n = 0
    a, b = 0, 1
    while n <= n_max:
        E = brentq(psi_at_bound,a,b)
        bounded(E, n)
        plot(E, n)
        a, b = a+1, b+1
        n = n+1

# Plot the eigenvectors, i.e. wave functions.
def plot(E, n):
    plt.plot(x, wave_function(E), label='$\psi_%d(x): E_%d = %.3f$'%(n,n,E))
    plt.xlabel('x')
    plt.ylabel('$\psi_%d(x)$'%n)
    plt.legend(loc='upper right')
    plt.show()

eigenvalues(n_max)
