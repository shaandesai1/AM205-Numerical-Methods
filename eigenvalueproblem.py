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
# Determine initial conditions
mu = 0.01
#number of crossings
tr = 2
dt = 0.01
dx = 0.1
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



# set of C values we want to test
Carr = np.array([3.2])
times = np.array([0,dt])
storage = np.zeros((((100,4,2,10))))
xinit = np.zeros((100,10))

for j in range(len(Carr)):
    xstore = []
    x = np.linspace(-1,1,100)  
    C = Carr[j]
# for each x determine permissibility
    for i in range(100):
        if f(x[i]) - C > 0:
            xstore.append(x[i])
    xstore = np.array(xstore)
# iterate once to fin1d first crossing with first x value
    xstart = xstore[0]  
    crossings = 0
    countr = 0
    num = 0
    yn = np.array([xstart,0, 0,initv(xstart, 0)])
    while crossings < tr:
        sol = sp.integrate.odeint(pend, yn, times)[1,:]
        if sol[1] >= 0 and yn[1] < 0:
            crossings = crossings + 1 
            countr = 0
        yn = sol

    ref = yn
    xstart = xstore[1]
    index = 2
# find the next points crossing and test it with the previous
    while index < len(xstore):
        crossings = 0
        yn = np.array([xstart,0, 0,initv(xstart, 0)])
        while crossings < tr:
             sol = sp.integrate.odeint(pend, yn, times)[1,:]
             if sol[1] >= 0 and yn[1] < 0:
                 crossings = crossings + 1
                 if crossings == tr:                 
                     t0 = 0
                     x = yn[0]
                     y = yn[1]
                     u = yn[2]
                    
                     v = yn[3]
                     fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
                     retval = opt.brentq(fbrent,0,dt)
                     xpoint = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,:]
             yn = sol
# check for a sign change and if there is one save the points before and after
        if np.sign(xpoint[2]) != np.sign(ref[2]):
            storage[j,:,0,num] = ref
            storage[j,:,1,num] = xpoint
            xinit[j,num] = xstart
            num = num + 1
        xstart = xstore[index]
        index = index + 1
        ref = xpoint

# linear root finding method on the two values saved above 

roots = np.zeros((100,10))

for i in range(len(Carr)):
    for q in range(10):
        u1 = storage[i,2,0,q]
        u2 = storage[i,2,1,q]
        x1 = storage[i,0,0,q]
        x2 = storage[i,0,1,q]
        if x2 -x1 == 0 or u2 - u1 == 0:
            roots[i,q] = 0
        else:
            roots[i,q] = ((u2 - u1)*x1/(x2 - x1) - u1)*(x2-x1)/(u2-u1)
            
eigensols = np.zeros(((len(Carr),10,2)),dtype = 'complex')

# Matrix calculation: solves eigenvalues of jacobi from fixed point perturbations: see write up

h = 10**(-8)
for i in range(len(Carr)):
    for q in range(10):
        x1 = roots[i,q]
        if x1 != 0:
            # perturb x1
            xstart = x1 + h
            crossings = 0
            yn = np.array([xstart,0, 0,initv(xstart, 0)])
            while crossings < tr:
                sol = sp.integrate.odeint(pend, yn, times)[1,:]
                if sol[1] >= 0 and yn[1] < 0:
                    crossings = crossings + 1 
                    countr = 0
                    if crossings == tr:                    
                        t0 = 0
                        x = yn[0]
                        y = yn[1]
                        u = yn[2]
                        v = yn[3]
                        fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
                        retval = opt.brentq(fbrent,0,dt)
                        vert0 = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,:]
                yn = sol
            xstart = x1
            crossings = 0
            ustart = h
            yn = np.array([xstart,0,ustart,initv(xstart,ustart)])
            while crossings < tr:
                sol = sp.integrate.odeint(pend, yn, times)[1,:]
                if sol[1] >= 0 and yn[1] < 0:
                    crossings = crossings + 1 
                    if crossings == tr:                    
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
            crossings = 0
            ustart = 0
            yn = np.array([xstart,0,ustart,initv(xstart,ustart)])
            while crossings < tr:
                sol = sp.integrate.odeint(pend, yn, times)[1,:]
                if sol[1] >= 0 and yn[1] < 0:
                    crossings = crossings + 1 
                    if crossings == tr:                    
                        countr = 0
                        t0 = 0
                        x = yn[0]
                        y = yn[1]
                        u = yn[2]
                        v = yn[3]
                        fbrent = lambda t: sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,t]))[1,1]
                        retval = opt.brentq(fbrent,0,dt)
                        cent = sp.integrate.odeint(pend,np.array([x,y,u,v]),np.array([0,retval]))[1,:]
                yn = sol
            # define new matrix of results
            A = (1/h)*np.array(([vert0[0]-cent[0],vert1[0]-cent[0]],[vert0[2] - cent[2],vert1[2]-cent[2]]))
            eigensols[j,q,:] = spl.eig(A)[0]
        