# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

plots red dots for a range of x values and u zero
plots blue dots for one iteration forward from the red initial conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt


# Determine initial conditions
mu = 0.01
C = 3.2
sample = 100
#number of crossings
tr = 1
uptime = 1000
dt = 0.01
h = 10**(-2)
# define dz/dt components for odeint solver
def pend(z,t):
    x,y,u,v = z
    dzdt = [u,v,v + 0.5*(2*(x-mu) + ((mu-1)*2*x)/(x**2+y**2)**(3.0/2) + (2*mu*(1-x))/((x-1)**2 + y**2)**(3.0/2)),-u + 0.5*(2*y + ((mu-1)*2*y)/(x**2+y**2)**(3.0/2) + (-2*mu*y)/((x-1)**2 + y**2)**(3.0/2)) ]
    return dzdt

# function used in determining set of initial conditions
def f(x):
    return (x-mu)**2 + 2*(1-mu)/np.abs(x) + 2*mu/np.abs(x-1)

# given x and u, return v
def initv(x,u):
    return np.sqrt((x-mu)**2 + 2*(1-mu)/np.abs(x) + 2*mu/np.abs(x-1) - u**2 - C)

# create an initial set of x and u then test them below
x = np.linspace(-1.1,1.2,sample)
#u = np.linspace(-0.5,0.5,sample)
# find the set of x and u that meet our condition for a real 'v' term
xstore = []

# for each x determine set of u's that ensure v is real
for i in range(sample):
    if f(x[i]) - C >= 0:
        xstore.append(x[i])
 
xstore = np.array(xstore)

res = np.zeros(((len(xstore),1,4)))
times = np.array([0,dt])
# given each x,u and v determine trajectories
for i in range(len(xstore)):          
    y0 = np.array([xstore[i],0, 0,initv(xstore[i], 0)])
    crossings = 0
    countr = 0
    while crossings < tr:
        if countr == 1000:
            break
        else:
            sol = sp.integrate.odeint(pend, y0, times)[1,:]
            if sol[1] >= 0 and y0[1] < 0:
                crossings = crossings + 1
                if crossings == tr:
                    res[i,0,:] = y0
                countr = 0
        countr = countr + 1
        y0 = sol

trajxu = np.zeros(((len(xstore),tr,2)))
# brents method on critical points for each trajectory
for j in range(len(xstore)):
        t0 = 0
        x = res[j,0,0]
        y = res[j,0,1]
        u = res[j,0,2]
        v = res[j,0,3]            
        fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
        retval = opt.brentq(fbrent,0,dt)
        trajxu[j,0,0] = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,0]
        trajxu[j,0,1] = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,2]
        
plt.figure(1)
plt.scatter(xstore,np.zeros(len(xstore)),s=1,color = 'red')
plt.scatter(trajxu[:,:,0],trajxu[:,:,1],s=2,color = 'blue')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel('x')
plt.ylabel('u')
