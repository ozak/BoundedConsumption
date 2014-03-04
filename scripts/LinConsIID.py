#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# coding: utf-8
# Computes optimal consumption when income is iid process and agent is liquidity constrained
# Author: �mer �zak
# Date: September 2013
from __future__ import division
from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound
from dynsys import *        # From listing 6.4
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm
import time, sys, os
from itertools import product

# Seed the random number generator
np.random.seed(100)

# Output directory
dir=os.getenv("HOME")+'/Dropbox/LatexMe/Consumption/data/LogN/'
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
fileopt=dir+'optcons.npz'
fileout=dir+'lincons'

# Let's replicate the Howitt Ozak parameter's
theta, beta= 3, 0.9     # Preference Parameters
R=1                     # Gross Interest rate
sigman=0.18               # Std of log-income

# Assume income process is log-normal with mean 1 and std 0.1
y = exp(sigman*randn(1000))                   # Draws of shock 

# auxiliary parameters and functions
theta1=1-theta
rho=beta

# Parameters of the linear consumption function
a=np.arange(0.01,2,0.01)       # Intercept
b=np.arange(0.01,1,0.01)       # MPC: Marginal propensity to consume out of wealth

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
dx=0.01          # Gridcell size for pdf
grid = linspace(dx, gridmax**1e-1, gridsize)**10
wgrid=np.arange(0.1,gridmax,dx)

# Parameters for the optimization procedures
count=0
maxiter=1000
tol=1e-12
print 'tol=%1.10f' % tol

# Define CRRA Utility function 
def U(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return ln(c)
    else:
        return (c**theta1-1)/theta1 

# Wealth transition equation (s is savings in period t)
def f(s, y): return R*s+y     

# T operator
def T(sigma, w):
    "Implements the operator L T_sigma."
    vals = []
    for W in grid:
        Tw_y = U(max(W - sigma(W),0),theta1) + rho * mean(w(f(sigma(W),y)))
        vals.append(Tw_y)
    return LinInterp(grid, vals)

# Value function of following policy sigma
def get_value(sigma, v):    
    """Computes an approximation to v_sigma, the value
    of following policy sigma. Function v is a guess.
    """
    tol_v = 1e-5         # Error tolerance 
    #counter=0
    while 1:#counter<maxiter:
        #counter+=1
        new_v = T(sigma, v)
        err_v = 2*beta/((1-beta)**2)*max(abs(new_v(grid) - v(grid)))
        if err_v < tol_v:
            return new_v
        v = new_v 
    '''if counter>maxiter:
        print 'Maximum iterations exceeded'
        return err'''           

# Consumption function as a function of intercept, MCP, and wealth
def c(w):
    '''Computes min(w,a0+a1*w)'''
    w=np.array(w)
    return np.where(w<=a0+w*a1,w,a0+w*a1)

# Saving function
def sigma0(w):
    return w-c(w)
# Other functions
rou = lambda x: np.round(x*100)/100
phinv=lognorm.pdf               # PDF of the income process
dF = lambda y, x: 1
t=10000        # Number of periods to simulate in order to generate the stationary pdf

# Matrices for outputs
EV=np.zeros(shape=(len(a),len(b)))
Pstat=np.zeros(shape=(len(wgrid),len(a),len(b)))
V=np.zeros(shape=(len(wgrid),len(a),len(b)))

# Find Value Function for each linear consumption function
start=time.time()
for i,j in product(range(len(a)),range(len(b))):
    start2=time.time()
    a0=a[i]
    a1=b[j]
    v0=LinInterp(grid,U(c(grid),theta1))
    v1=get_value(sigma0,v0)
    g = lambda x: R*sigma0(x)    # Interest on savings
    F = lambda x, w : g(x)+w        # Wealth transition function
    Finv = lambda w, x: w-g(x)      # income level for given future wealth and current interest+savings
    Wealth_srs=SRS(F=F,phi=np.random.lognormal,X=np.random.permutation(grid)[0],mu=0,sigma=sigman)
    Pstat[:,i,j]=Wealth_srs.stationaryDist(n=t, FInv=Finv, dF=dF, phinv=phinv, dx=dx,xmin=0.1)
    V[:,i,j]=v0(wgrid)
    EV[i,j]=np.dot(Pstat[:,i,j],V[:,i,j])
    print(a0,a1)
    print('it took %2.2f seconds' %(time.time()-start2))
CE=(1+theta1*(1-beta)*EV)**(1/theta1)
np.savez_compressed(fileout,EVlin=EV,CElin=CE,Pstat=Pstat,Vlin=V)
print('it took %2.2f seconds to do the whole job' %(time.time()-start))

    

 

    
    