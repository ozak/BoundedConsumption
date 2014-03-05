#!/usr/bin/env python
# coding: utf-8
'''
======================================================
Author:  Ömer Özak, 2013--2014 (ozak at smu.edu)
Website: http://omerozak.com
GitHub:  https://github.com/ozak/BoundedConsumption
======================================================
# This code generates the Fully Rational Consumption and Value Functions for the paper:
# 1. Özak, Ömer, "Optimal consumption under uncertainty, liquidity constraints, and bounded rationality", Journal of Economic Dynamics and Control, 2014, Vol. 39: 237-254 (http://dx.doi.org/10.1016/j.jedc.2013.12.007)
# In particular, it computes optimal consumption when agent is liquidity constrained, income follows 3-point iid process (with possible inverse U-shaped trend), CRRA coefficient is 3, discount factor is 0.9 (as in Allen and Carroll) and agent lives T periods
# These are baseline results and are the common specification in both these papers and in the Allen and Carroll paper.
# The program is not fully optimized. Instead it is written in order to maximize readibility, understanding, and replicability.
# Should work on most Python distributions. Tested on Enthought Canopy 1.3, Python.org 2.7.6 + Numpy 1. + Scipy 1.10
# Feel free to use the code and play with parameters
# Author: Ömer Özak
# email: ozak (at) smu.edu
# Date: April 2013
'''
from __future__ import division
from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound
from dynsys import *        # From listing 6.4
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm
import time,sys,os

# Setup to make permanent income inverse Ushaped or constant
constant=1

# Seed the random number generator
np.random.seed(100)

# Output directory
dir='../data/T/HOAC/'
if os.path.exists(dir[0:len(dir)-7])==False:
    os.mkdir(dir[0:len(dir)-7])
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
if constant==0:
    dir='../data/T/HOAC/constant/'
    if os.path.exists(dir)==False:
        os.mkdir(dir)
file=dir+'optconsT'

# Let's replicate the Howitt Ozak (2014) parameter's
theta, beta= 3, 0.9                 # Preference Parameters
p=np.array([0.2, 0.6,0.2])          # Probability if income value i
y1=np.array([0.7,1,1.3])            # Income values
R=1                                 # Gross Interest rate

# Lifespan is TT+1, i.e. agent dies in period TT+1
TT=60

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
grid = linspace(0.01, gridmax**1e-1, gridsize)**10

# auxiliary parameters and functions
theta1=1-theta
rho=beta

def phi(): return y[sample(p)]    # Sample out of the income process

# Generate random sample of size 1000 from income process
yt=np.array(y1[[sample(p) for i in range(1000)]])           # Transitory income shock
yp = 1-constant*(np.arange(0,1,1/(TT+2))-.5)**2
# Parameters for the optimization procedures
maxiter=1000
tol=1e-12
print 'tol=%1.10f' % tol

# Define CRRA Utility function 
def U(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return np.ln(c)
    else:
        return (c**theta1-1)/theta1 

# Wealth transition equation (s is savings in period t)
def f(s, y): return R*s+y     

# Maximize function h on interval [a,b]
def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))

# Return Maximizer
def maximizer(h, a, b):
    return float(fminbound(lambda x: -h(x), a, b))

# The following two functions are used to find the optimal consumption and value functions using value function iteration
# Bellman Operator
def bellman(w,t):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
    Given a Value w function and period t, which determines the income to be received
    it returns the value at t-1
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,yp[t]*yt)))
        vals.append(maximum(h, 0, W))
    return LinInterp(grid, vals)

# Optimal policy
def policy(w,t):
    """The approximate optimal policy operator w-greedy.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that captures the optimal policy.
    Given the value function in perio t, it return the optimal saving in that period
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,yp[t]*yt)))
        vals.append(maximizer(h, 0, W))
    return LinInterp(grid, vals)

##################################################################################################
##################################################################################################
# Now let's start the computations
##################################################################################################
##################################################################################################

start=time.time()

##################################################################################################
# Finding the approximate value function using value function iteration
##################################################################################################

# Optinal policies and indirect utilities in the final period
count=TT
u0=LinInterp(grid,U(grid,theta1))
c0=LinInterp(grid,grid)
s0=LinInterp(grid,np.zeros_like(grid))

# Holders for our results
# u holds the optimal expected utility at each age utility
u=np.array([],dtype=type(u0))
# copt holds the optimal consumption function at each age
copt=np.array([],dtype=type(u0))
# sopt holds the optimal savings function at each age
sopt=np.array([],dtype=type(u0))

# u0 is the utility of consuming everything (c0) adn saving nothing (s0), which is optimal at age TT
u=np.append(u,u0)
copt=np.append(copt,c0)
sopt=np.append(sopt,s0)

# Iterate backwards
start=time.time()
while count>-1:
    u0=bellman(u0,count)
    s0=policy(u0,count)
    c0=LinInterp(grid,grid-s0(grid))
    u=np.append(u,u0)
    copt=np.append(copt,c0)
    sopt=np.append(sopt,s0)
    count-=1
print('it took %2.2f seconds to compute' % (time.time()-start))

# Now let's draw the approximate optimal consumption, saving, utility for each age
plt.figure(1)
#plt.ylim([0,2])
#plt.xlim([0,2])
#plt.plot(grid,grid)
for i in range(len(copt)):
    plt.plot(grid,copt[i](grid))
plt.savefig(dir+'OptCons.eps')
plt.draw()

plt.figure(2)
#plt.ylim([0,2])
#plt.xlim([0,2])
plt.plot(grid,grid)
for i in range(len(sopt)):
    plt.plot(grid,sopt[i](grid)+yp[i])
plt.savefig(dir+'OptSav.eps')
plt.draw()

wgrid=np.arange(0.1,gridmax,0.1)
plt.figure(3)
#plt.ylim([0,2])
#plt.xlim([0,2])
plt.plot(grid,grid)
for i in range(len(u)):
    plt.plot(wgrid,u[i](wgrid))
plt.savefig(dir+'OptValue.eps')
plt.draw()

copt2=np.array([copt[i](grid) for i in range(len(copt))])
sopt2=np.array([sopt[i](grid) for i in range(len(sopt))])
u2=np.array([u[i](grid) for i in range(len(u))])

# Save results to be used by adaptive algorithm
np.savez_compressed(file,grid=grid,sopt=sopt,copt=copt,u=u,sopt2=sopt2,copt2=copt2,u2=u2)
plt.show()
