# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# Determine initial conditions
mu = 0.01
C = 5
sample = 10
tr = 100
uptime = 10000

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
x = np.linspace(-2,2,sample)
u = np.linspace(-2,2,sample)

# find the set of x and u that meet our condition for a real 'v' term
xstore = []
xprime = []

# for each x determine set of u's that ensure v is real
for i in range(sample):
    a = x[i]
    for q in range(sample):
        b = u[q]
        if f(a) - b**2 - C >= 0:
            xstore.append(a)
            xprime.append(b)

xstore = np.array(xstore)
xprime = np.array(xprime)
res = []

# given each x,u and v determine trajectories
for i in range(len(xstore)):          
    y0 = np.array([xstore[i],0,xprime[i],initv(xstore[i],xprime[i])])
    times = np.linspace(0,10000,uptime)
    sol = sp.integrate.odeint(pend, y0, times)
    res.append(sol)
res = np.array(res)

# create container for each trajectory (assume max # of intersections is tr)
index = np.zeros((len(res),tr))

i = 0
# for each trajectory, look at times when we go from -y to +y and store the index
for j in range(len(res)):        
    i = 0
    for k in range(uptime-1):
        if res[j,k,1] < 0 and res[j,k+1,1]>=0  and res[j,k,3] > 0:
                index[j,i] = k
                i = i +1

# create trajxu array to store all newton roots of x and u
trajxu = np.zeros(((len(res),tr,2)))
trajyv = np.zeros(((len(res),tr,2)))

# newtons method on critical points for each trajectory
for j in range(len(res)):
    for q in range(tr):
        if index[j,q] != 0:
            t0 = 0
            x = res[j,index[j,q],0]
            y = res[j,index[j,q],1]
            u = res[j,index[j,q],2]
            v = res[j,index[j,q],3]
            while np.abs(y) > 10**(-10):
                t1 = t0 - y/v
                x,y,u,v = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([t0,t1]))[1,:]
                t0 = t1
            trajxu[j,q,0] = x
            trajxu[j,q,1] = u
            trajyv[j,q,0] = y
            trajyv[j,q,1] = v


plt.figure(1)
eta = np.linspace(-2,2,5000)
#plt.scatter(xstore,xprime,s = 0.01)
plt.plot(eta,ytst(eta))
plt.plot(eta,-ytst(eta))
plt.scatter(trajxu[5,:,0],trajxu[5,:,1],s = 0.01) 
plt.xlim([-0.5,0.5])
plt.ylim([-20,20])
