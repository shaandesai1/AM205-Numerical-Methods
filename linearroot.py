# -*- coding: utf-8 -*-
"""
Shaan Desai and Graham Lustiber
AM205 Final Project Code

solves for the stable equilibrium points using linear interpolation

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
    count = 0
    xpoint = np.array([0,0,0,0])
    while xings < tr:
        if count == 1000:
            break
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
                count = 0
        yn = sol
        count+=1
        
    return xpoint
    
tr = 1

mu = 0.01
dt = 0.01
dx = 0.1
nroots = 20

xup = 1.2
xdown = -1.2
sample = 50
Carr = np.linspace(2.8,4.0,80)

times = np.array([0,dt])
storage = np.zeros((((100,4,2,nroots))))

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
            storage[j,:,0,num] = ref
            storage[j,:,1,num] = xpoint
            num = num + 1
        xstart = xstore[index]
        index = index + 1
        ref = xpoint

# linear root finding method on the two values saved above 

roots = np.zeros((len(Carr),nroots))

for i in range(len(Carr)):
    for q in range(nroots):
        u1 = storage[i,2,0,q]
        u2 = storage[i,2,1,q]
        x1 = storage[i,0,0,q]
        x2 = storage[i,0,1,q]
        if x2 -x1 == 0 or u2 - u1 == 0:
            roots[i,q] = 0
        else:
            roots[i,q] = ((u2 - u1)*x1/(x2 - x1) - u1)*(x2-x1)/(u2-u1)
            
eigensols = np.zeros(((len(Carr),nroots,2)),dtype = 'complex')

# Matrix calculation: solves eigenvalues of jacobi from fixed point perturbations: see write up
xvals = np.zeros((len(Carr),nroots))
h = 10**(-8)
for i in range(len(Carr)):
    rv = 0    
    for q in range(nroots):
        C = Carr[i]
        x1 = roots[i,q]
        if x1 != 0.0:
            # perturb x1
            xvals[i,rv] = x1
            rv+=1            
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
            eigensols[i,q,:] = spl.eig(A)[0]


#vector = np.zeros((len(Carr),3))
new_xs = []
new_Carr = []

for i in range(len(Carr)):
    for q in range(nroots):
        if np.round(np.abs(eigensols[i,q,0]),1) == 1.0 and np.round(np.abs(eigensols[i,q,1]),1) == 1.0:
            if not (-0.1 < xvals[i][q] < 0.1):
                new_xs.append(xvals[i][q])
                new_Carr.append(Carr[i])
                
            #vector[i,q] = xvals[i,q]


plt.figure(1)
plt.scatter(new_xs, new_Carr,s=1)
plt.xlabel('x')
plt.ylabel('J')
plt.title('J vs x')
