# This code compares the solutions to the Schrodinger Equation with a harmonic
# vs an anharmonic potential by plotting the eigenvectors and a plot of the
# difference in the energy at each quantum number n.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Define variables.
n_max = 10
x = np.linspace(-5,5,1001)

# Define constants.
m = 1
hbar = 1
k = 1
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

# Set (an)harmonic potential energy at 'x'.
def V(x):
    potential = (0.5*k*x**2)-((1/6)*k_prime*x**3)
    return potential

# Find the value of psi at the positive bound of x (needed by brentq).
def psi_at_bound(E):
    norm_psi = wave_function(E)
    psi_at_bound = norm_psi[-1]
    return psi_at_bound

# Calculate eigenvalues, i.e. allowed energies.
def eigenvalues(n_max):
    energies = []
    n = 0
    a, b = 0, 1
    while n <= n_max:
        E = brentq(psi_at_bound,a,b)
        energies.append(E)
        plot(E, n)
        a, b = a+1, b+1
        n = n+1
    return energies

# Plot the eigenvectors, i.e. wave functions.
def plot(E, n):
    plt.plot(x, wave_function(E), label='$\psi_%d(x): E_%d = %.3f$'%(n,n,E))
    plt.xlabel('x')
    plt.ylabel('$\psi_%d(x)$'%n)
    plt.legend(loc='upper right')
    plt.show()

k_prime = 0
harm_eig = eigenvalues(n_max)
k_prime = 0.25
anharm_eig = eigenvalues(n_max)
difference = []
for i in range(n_max+1):
    d = harm_eig[i] - anharm_eig[i]
    difference.append(d)

plt.plot(range(n_max+1), difference)
plt.xlabel('quantum number n')
plt.ylabel('E_harmonic - E_anharmonic')
plt.show()