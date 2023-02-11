# This code determines whether are not certain wave functions of a finite
# square well are bound or unbound to the well.

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
from scipy.misc import derivative
import matplotlib.pyplot as plt

m = 1
hbar = 1
init = 0, 1
NumPts = 1000
x = np.linspace(-2,2,NumPts)
E = 25
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
    global n
    global z
    n = 1
    # Determines which energies diverge towards 0 at the bounds. A few energy
    # levels above and below the potential energy of the walls of the well are
    # tested for whether or not they are bound to the well.
    for i in range(E):
        if np.sign(wave_function(i)) != np.sign(wave_function(i+1)):
            z = brentq(wave_function, i, i+1)
            zeros.append(z)
            if z >=14:
                plt.plot(x, psi[:,0], label='$E_%d$ = %.2f'%(n,z))
                decay_test(psi[750:,1])
            n = n+1
    return zeros

# Determines whether or not the wave function is exponentially decaying before
# after entering the square well. Decaying functions are bound to the well,
# nondecaying functions are unbound.
def decay_test(slopes):
    for i in slopes:
        if np.sign(i) != np.sign(i+1):
            print('The wave function is unbounded at E%d = %.2f'%(n,z))
            break
    else:
        print('The wave function is bounded at E%d = %.2f'%(n,z))

# Determines whether the solution to the Schrodinger Equation when E=V0 is bound
# or unbound to the well.
def E_equals_V0(slopes):
    for i in slopes:
        if np.sign(i) != np.sign(i+1):
            print('The solution is unbounded at E = V0 = %d'%V0)
            break
    else:
        print('The solution is bounded at E = V0 = %d'%V0)

allowed_energies(E)
wave_function(V0)
E_equals_V0(psi[750:,1])

plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.show()
