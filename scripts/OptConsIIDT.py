#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# coding: utf-8
# Computes optimal consumption when income is iid process and agent is liquidity constrained
# Author: Ömer Özak
# Date: April 2013
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
dir=os.getenv("HOME")+'/Dropbox/LatexMe/Consumption/data/T/LogN/'
if os.path.exists(dir[0:len(dir)-7])==False:
    os.mkdir(dir[0:len(dir)-7])
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
if constant==0:
    dir=os.getenv("HOME")+'/Dropbox/LatexMe/Consumption/data/T/LogN/constant/'
    if os.path.exists(dir)==False:
        os.mkdir(dir)
file=dir+'optconsT'

# Let's replicate the Howitt Ozak parameter's
theta, beta= 3, 0.9     # Preference Parameters
#p=np.array([0.2, 0.6,0.2])        # Probability if income value i
#y=np.array([0.7,1,1.3])             # Income values
R=1                     # Gross Interest rate
sigman=0.18               # Std of log-income

# Lifespan is TT+1, i.e. agent dies in period T+1
TT=60

# auxiliary parameters and functions
theta1=1-theta
rho=beta

# Assume income process is U shaped + log-normal shock with mean 1 and std 0.1
yt = exp(sigman*randn(1000))                   # Draws of shock
#yp = 1-constant*(np.arange(0,1,1/(TT+2))-.5)**2
yp = .6-constant*1.5*(np.arange(0,1,1/(TT+2))-.5)**2

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

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
grid = linspace(0.01, gridmax**1e-1, gridsize)**10

# Maximize function h on interval [a,b]
def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))

# Return Maximizer
def maximizer(h, a, b):
    return float(fminbound(lambda x: -h(x), a, b))

# Bellman Operator
def bellman(w,t):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
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
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,yp[t]*yt)))
        vals.append(maximizer(h, 0, W))
    return LinInterp(grid, vals)

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

# Finding the approximate value function using value function iteration
count=TT
u0=LinInterp(grid,U(grid,theta1))
c0=LinInterp(grid,grid)
s0=LinInterp(grid,np.zeros_like(grid))
# u holds the optimal expected utility at each age utility
u=np.array([],dtype=type(u0))
# copt holds the optimal consumption function at each age
copt=np.array([],dtype=type(u0))
# sopt holds the optimal savings function at each age
sopt=np.array([],dtype=type(u0))
# u0 is the utility of consuming everything, which is optimal at age TT+1
u=np.append(u,u0)
copt=np.append(copt,c0)
sopt=np.append(sopt,s0)

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

# Now let's draw the approximate optimal consumption, saving, utility
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

########
np.savez_compressed(file,grid=grid,sopt=sopt,copt=copt,u=u,sopt2=sopt2,copt2=copt2,u2=u2)
plt.show()
'''
'''
