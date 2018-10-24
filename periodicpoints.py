# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt


# Determine initial conditions
mu = 0.01
C = 3.1
sample = 4
#number of crossings
tr = 300
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
x = np.linspace(-1,1.5,sample)
u = np.linspace(-1,1.2,sample)
# find the set of x and u that meet our condition for a real 'v' term
xstore = []
xprime = []

# for each x determine set of u's that ensure v is real
for i in range(sample):
    a = x[i]
    for q in range(sample):
        if f(a) - u[q]**2- C >= 0:
            xstore.append(a)
            xprime.append(u[q])

xstore = np.array(xstore)
xprime = np.array(xprime)


res = np.zeros(((len(xstore),tr,4)))
times = np.array([0,dt])
# given each x,u and v determine trajectories
for i in range(len(xstore)):          
    y0 = np.array([xstore[i],0, xprime[i],initv(xstore[i], xprime[i])])
    crossings = 0
    countr = 0
    while crossings < tr:
        if countr == 1000:
            break
        else:
            sol = sp.integrate.odeint(pend, y0, times)[1,:]
            if sol[1] >= 0 and y0[1] < 0:
                res[i,crossings,:] = y0
                #res[i,crossings,:] = sol
                crossings = crossings + 1 
                countr = 0
        countr = countr + 1
        y0 = sol
    
trajxu = np.zeros(((len(xstore),tr,2)))
# newtons method on critical points for each trajectory
for j in range(len(xstore)):
    for q in range(tr):
            t0 = 0
            x = res[j,q,0]
            y = res[j,q,1]
            u = res[j,q,2]
            v = res[j,q,3]
            fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
            retval = opt.brentq(fbrent,0,dt)
            trajxu[j,q,0] = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,0]
            trajxu[j,q,1] = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,2]
            
plt.figure(1)
plt.scatter(trajxu[:,:,0],trajxu[:,:,1],s=0.01)
plt.xlim([-0.4,0.6])
plt.ylim([-20,20])

