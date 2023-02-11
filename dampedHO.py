# This script solves the ODE of a damped harmonic oscillator:
# x''(t)+2*(gamma)*x'(t)+(omega^2)*x(t)=0, where 'gamma' is the
# dampening constant, and 'omega' is the oscillation frequency.
# The harmonic oscillator is overdamped when gamma > omega,
# underdamped when gamma < omega, critically damped when
# gamma = omega, and undamped when gamma = 0. The initial
# conditions x(0) and x'(0) are defined as 'init', and the time
# interval is defined as an array 't', ranging from t(initial) to
# t(final) with a given number of times to calculate the ODE at
# over the time interval. The script prints the values of 'x'
# at each value of 't' over the time interval for each value of gamma,
# and plots 'x' vs 't' for the time interval at each value of gamma.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def driver(x, t, gamma, omega):
    x0 = x[0]
    x1 = x[1]
    x2 = (-1.0*2*gamma*x1)+(-1.0*(omega**2)*x0)
    return x1, x2

# Oscillation frequency 'omega'
omega = 1.0

# Initial conditions on x, x' at t=0
init = 1.0, 0.0

# range of t
t = np.linspace(0,25,200)

# Various values of gamma
gamma = np.linspace(0.0, 2.0, 11)

# solver
for g in gamma:
    sol = odeint(driver, init, t, args=(g,omega))
    if g == 0.0:
        plt.plot(t, sol[:,0], color='0.90')
    elif g != 0.0 and g < omega:
        plt.plot(t, sol[:,0], color='b')
    elif g == omega:
        plt.plot(t, sol[:,0], color='r')
    elif g > omega:
        plt.plot(t, sol[:,0], color='g')

blue = mlines.Line2D([], [], color='b', label='underdamped')
red = mlines.Line2D([], [], color='r', label='critically damped')
green = mlines.Line2D([], [], color='g', label='overdamped')
gray = mlines.Line2D([], [], color='0.90', label='undamped')
plt.legend(handles=[blue, red, green, gray], loc='upper right')
plt.xlabel('t')
plt.ylabel('x')
plt.show()
