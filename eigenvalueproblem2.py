# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

solves for the stable equilibrium points

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize as opt

import scipy.linalg as spl
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

def crossings(xstart):
    xings = 0
    yn = np.array([xstart,0, 0,initv(xstart, 0)])
    while xings < tr:
         sol = sp.integrate.odeint(pend, yn, times)[1,:]
         if sol[1] >= 0 and yn[1] < 0:
             xings = xings + 1
             if xings == tr:                 
                 x = yn[0]
                 y = yn[1]
                 u = yn[2]
                 v = yn[3]
                 fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
                 retval = opt.brentq(fbrent,0,dt)
                 xpoint = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,:]
         yn = sol
    return xpoint
    
tr = 1

mu = 0.01
dt = 0.01
dx = 0.1

xup = 1.1
xdown = -1.1
sample = 40

Carr = np.linspace(3.0,3.5,10)
times = np.array([0,dt])
storage = np.zeros((((100,1,2,10))))

for j in range(len(Carr)):
    num = 0
    xstore = []
    x = np.linspace(xdown,xup,sample)  
    C = Carr[j]
# for each x determine permissibility
    for i in range(sample):
        if f(x[i]) - C > 0:
            xstore.append(x[i])
    xstore = np.array(xstore)
# iterate once to fin1d first crossing with first x value
    xstart = xstore[0]  
    ref = crossings(xstart)
    xstart = xstore[1]
    index = 2
# find the next points crossing and test it with the previous
    while index < len(xstore):
        xpoint = crossings(xstart)
# check for a sign change and if there is one save the points before and after
        if np.sign(xpoint[2]) != np.sign(ref[2]):
            storage[j,:,0,num] = xstore[index-2]
            storage[j,:,1,num] = xstore[index-1]
            num = num + 1
        xstart = xstore[index]
        index = index + 1
        ref = xpoint

# brentq root finding method on the two values saved above 

roots = np.zeros((100,10))

for i in range(len(Carr)):
    for q in range(0,10):
        fbrent1 = lambda r: crossings(r)[2]
        retval = opt.brentq(fbrent1,storage[i,0,1,q],storage[i,0,0,q])
        roots[i,q] = retval
            
eigensols = np.zeros(((len(Carr),10,2)),dtype = 'complex')

# Matrix calculation: solves eigenvalues of jacobi from fixed point perturbations: see write up

h = 10**(-8)
for i in range(len(Carr)):
    for q in range(10):
        x1 = roots[i,q]
        if x1 != 0:
            # perturb x1
            xstart = x1 + h
            vert0 = crossings(xstart)            
            xstart = x1
            ustart = h 
            yn = np.array([xstart,0,ustart,initv(xstart,ustart)])
            xings = 0            
            while xings < tr:
                sol = sp.integrate.odeint(pend, yn, times)[1,:]
                if sol[1] >= 0 and yn[1] < 0:
                    xings = xings + 1 
                    if xings == tr:                    
                        countr = 0
                        t0 = 0
                        x = yn[0]
                        y = yn[1]
                        u = yn[2]
                        v = yn[3]
                        fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
                        retval = opt.brentq(fbrent,0,dt)
                        vert1 = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,:]
                yn = sol
            # central node
            xstart = x1
            cent = crossings(xstart)            
            # define new matrix of results
            A = (1/h)*np.array(([vert0[0]-cent[0],vert1[0]-cent[0]],[vert0[2] - cent[2],vert1[2]-cent[2]]))
            eigensols[j,q,:] = spl.eig(A)[0]

vector = np.zeros((100,10))

for i in range(len(Carr)):
    for q in range(10):
        if np.round(np.abs(eigensols[i,q,0]),1) == 1.0 and np.round(np.abs(eigensols[i,q,1]),1) == 1.0:
            vector[i,q] = roots[i,q]