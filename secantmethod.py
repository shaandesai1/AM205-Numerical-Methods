# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

# Determine initial conditions
mu = 0.01
C = 4
sample = 50
tr = 200
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
x = np.linspace(-1,1.1,sample)

# find the set of x and u that meet our condition for a real 'v' term
xstore = []


# for each x determine set of u's that ensure v is real
for i in range(sample):
    a = x[i]
    if f(a) - C >= 0:
        xstore.append(a)

xstore = np.array(xstore)

res = np.zeros(((len(xstore),tr,4)))
times = np.array([0,dt])
# given each x,u and v determine trajectories
for i in range(len(xstore)):          
    y0 = np.array([xstore[i],0,0,initv(xstore[i],0)])
    crossings = 0
    countr = 0
    while crossings < tr:
        if countr == 1000:
            break
        else:
            sol = sp.integrate.odeint(pend, y0, times)[1,:]
            if sol[1] >= 0 and y0[1] < 0:
                res[i,crossings,:] = y0
                #res[i,crossings,:,1] = sol
                crossings = crossings + 1 
                countr = 0
        countr = countr + 1
        y0 = sol
    
trajxu = np.zeros(((len(xstore),tr,2)))
# newtons method on critical points for each trajectory
for j in range(len(xstore)):
    for q in range(tr):
            t0 = 0
            xa = res[j,q,0]
            ya = res[j,q,1]
            ua = res[j,q,2]
            va = res[j,q,3]
            b = dt
            a = 0
            while b - a >10**(-15) :
                c = 0.5*(b+a)
                integ = sp.integrate.odeint(pend,np.array([xa,ya,ua,va]),np.array([0,c]))[1,:]
                if integ[1] < 0:
                    a = c
                    xa = integ[0]
                    ya = integ[1]
                    ua = integ[2]
                    va = integ[3] 
                else:
                    b = c
            trajxu[j,q,0] = xa
            trajxu[j,q,1] = ua
                
plt.figure(1)
plt.scatter(trajxu[:,:,0],trajxu[:,:,1],s=0.05,color='blue')
plt.xlim([-2,2])
plt.ylim([-20,20])
plt.ylabel('u')
plt.xlabel('x')
plt.title('J = 4')
v = np.linspace(-3,3,5000)
plt.plot(v,ytst(v),'g',label='limit')
plt.plot(v,-ytst(v),'g')
t1 = time.clock() - tzero