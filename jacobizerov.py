#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:17:45 2016

@author: shaandesai
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# Determine initial conditions
mu = 0.01
sample = 11
tr = 500
uptime = 1000
dt = 0.01

# jacobi constant
def J(x,y,u,v):
    return (x-mu)**2 + y**2 + 2*(1-mu)/(x**2 + y**2)**(1.0/2) + 2*mu/((x-1)**2 + y**2)**(1.0/2) - u**2 - v**2
# define dz/dt components for odeint solver
def pend(z,t):
    x,y,u,v = z
    dzdt = [u,v,v + 0.5*(2*(x-mu) + ((mu-1)*2*x)/(x**2+y**2)**(3.0/2) + (2*mu*(1-x))/((x-1)**2 + y**2)**(3.0/2)),-u + 0.5*(2*y + ((mu-1)*2*y)/(x**2+y**2)**(3.0/2) + (-2*mu*y)/((x-1)**2 + y**2)**(3.0/2)) ]
    return dzdt

# gives bounds of u in y = 0 plane (see latex file)
def ytst(x):
    return np.sqrt((x-mu)**2 + 2*(1-mu)/np.abs(x) + 2*mu/np.abs(x-1) -C)

# function used in determining set of initial conditions
def f(x):
    return (x-mu)**2 + 2*(1-mu)/np.abs(x) + 2*mu/np.abs(x-1)

# given x and u, return v
def initv(x,u):
    return np.sqrt((x-mu)**2 + 2*(1-mu)/np.abs(x) + 2*mu/np.abs(x-1) - u**2 - C)

# create an initial set of x and u then test them below
x = np.linspace(-1.2,1.2,sample)
y = np.linspace(-1.5,1.5,sample)

# find the set of x and u that meet our condition for a real 'v' term
plt.figure(1)
X,Y = np.meshgrid(x,y)
Z = J(X,Y,0,0)
plt.contour(X,Y,Z)