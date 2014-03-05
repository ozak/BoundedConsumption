#!/usr/bin/env python
# coding: utf-8
'''
======================================================
Author:  Ömer Özak, 2013--2014 (ozak at smu.edu)
Website: http://omerozak.com
GitHub:  https://github.com/ozak/BoundedConsumption
======================================================
# This code generates the Fully Rational Consumption and Value Functions for the papers:
# 1. Howitt, Peter and Özak, Ömer, "Adaptive Consumption Behavior" Journal of Economic Dynamics and Control, 2014, Vol. 39: 37-61 (http://dx.doi.org/10.1016/j.jedc.2013.11.003)
# 2. Özak, Ömer, "Optimal consumption under uncertainty, liquidity constraints, and bounded rationality", Journal of Economic Dynamics and Control, 2014, Vol. 39: 237-254 (http://dx.doi.org/10.1016/j.jedc.2013.12.007)
# In particular, it computes optimal consumption when agent is liquidity constrained, income follows 3-point iid process, CRRA coefficient is 3, and discount factor is 0.9 (as in Allen and Carroll) 
# These are baseline results and are the common specification in both these papers and in the Allen and Carroll paper.
# The program is not fully optimized. Instead it is written in order to maximize readibility, understanding, and replicability.
# It includes two ways of contructing the optimal policies and value functions.
# Should work on most Python distributions. Tested on Enthought Canopy 1.3, Python.org 2.7.6 + Numpy 1. + Scipy 1.10
# Feel free to use the code and play with parameters
# Author: Ömer Özak
# email: ozak (at) smu.edu
# Date: April 2013
'''
from __future__ import division
from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm
import time, sys, os
from dynsys import *        # From listing 6.4
import dynsysf 

# Seed the random number generator
np.random.seed(100)

# Output directory
dir='../data/HOAC/'
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
file=dir+'optcons'

# Let's replicate the Howitt Ozak (2014) parameter's
theta, beta= 3, 0.9                 # Preference Parameters
p=np.array([0.2, 0.6,0.2])          # Probability if income value i
y1=np.array([0.7,1,1.3])            # Income values
R=1                                 # Gross Interest rate

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
dx=0.01          # Gridcell size for pdf
grid = linspace(dx, gridmax**1e-1, gridsize)**10
wgrid=np.arange(0.1,gridmax,dx)

# auxiliary parameters and functions
theta1=1-theta
rho=beta
def phi(): return y[sample(p)]    # Sample out of the income process

# Generate random sample of size 1000 from income process
y=np.array(y1[[sample(p) for i in range(1000)]])

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

# Wealth transition equation (s is savings, R is interest factor, y is income, in period t)
def f(s, y): return R*s+y     

# Maximize function h on interval [a,b]
def maximum(h, a, b):
    return float(h(fminbound(lambda x: -h(x), a, b)))

# Return Maximizer
def maximizer(h, a, b):
    return float(fminbound(lambda x: -h(x), a, b))

# The following two functions are used to find the optimal consumption and value functions using value function iteration
# Bellman Operator
def bellman(w):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
    w is a function defined on the state space.
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,y)))
        vals.append(maximum(h, 0, W))
    return LinInterp(grid, vals)

# Optimal policy
def policy(w):
    """The approximate optimal policy operator w-greedy.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays). 
    Returns: An instance of LinInterp that captures the optimal policy.
    For each function w, policy(w) return the function that maximizes the RHS of the Bellman operator.
    Replace w for the Value function to get optimal policy.
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,y)))
        vals.append(maximizer(h, 0, W))
    return LinInterp(grid, vals)

# The following two functions are used to find the optimal consumption and value functions using policy function iteration
# T operator
def T(sigma, w):
    """Implements the operator L T_sigma.
    For each policy sigma and 'Value' function, it returns a new Value function
    Used by function get_value(sigma,v) to generate the value function under sigma.
    """
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
        
##################################################################################################
##################################################################################################
# Now let's start the computations
##################################################################################################
##################################################################################################

start=time.time()

##################################################################################################
# Finding the approximate value function using value function iteration
##################################################################################################

u0=LinInterp(grid,U(grid,theta1))   # Initial guess of value function (using linear interpolation on grid, u0 is a function!)
# Let's plot the intermediate output
plt.figure(1)
#plt.ylim([0,2])
#plt.xlim([0,2])
plt.plot(grid,u0(grid))
plt.draw()

while count<maxiter:
    u1=bellman(u0)
    err=2*beta/((1-beta)**2)*np.max(np.abs(np.array(u1(grid))-np.array(u0(grid))))
    if np.mod(count,25)==0:
        plt.figure(1)
        plt.ylim([-20,4])
        #plt.xlim([0,2])
        plt.plot(grid,u1(grid))
        plt.draw()
        print '%d %2.10f ' % (count,err)
    #print err
    if err<tol:
        print count
        break
    u0=u1
    count+=1
    #m0=maximum()
plt.plot(grid,u1(grid))
plt.draw()

# u1 is the optimal value function
optpolicy=policy(u0)                            # Optimal saving policy
optcons=LinInterp(grid,grid-optpolicy(grid))    # Optimal consumption function
print('error is %2.14f' % np.max(np.abs(np.array(u1(grid))-np.array(u0(grid)))))
print('it took %2.2f seconds to compute' % (time.time()-start))

# Now let's use the approximate optimal policy function to draw it
# Graph for Fully Rational Optimal Savings
plt.figure(2)
#plt.ylim([0,2])
#plt.xlim([0,2])
plt.plot(grid,grid)
plt.plot(grid,optpolicy(grid)+mean(y))
plt.savefig(dir+'OptSaving.eps')
plt.draw()

# Graph for Fully Rational Optimal Consumption
plt.figure(3)
#plt.ylim([0,2])
#plt.xlim([0,2])
#plt.plot(grid,grid)
plt.plot(grid,optcons(grid))
plt.savefig(dir+'OptCons.eps')
plt.draw()

##################################################################################################
# Now let us construct the approximate optimal policies and value functions
# using policy iteration
##################################################################################################

# Draw intermediate steps again
plt.figure(4)
start=time.time()
sigma0=LinInterp(grid,0.5*grid)             # Initial guess for optinal savings function
v0=LinInterp(grid,U(0.5*grid,theta1))       # Initial guess for value function of sigma
count=0
while count<maxiter:
    v1=get_value(sigma0,v0)                 # Generate v_sigma_0 value function
    sigma1=policy(v1)                       # Generate optimal policy for v_sigma_1
    plt.plot(grid,sigma1(grid))
    plt.draw()
    err=2*beta/((1-beta)**2)*np.max(np.abs(np.array(v1(grid))-np.array(v0(grid))))              # Stopping condition
    #err=2*beta/((1-beta)**2)*np.max(np.abs(np.array(sigma0(grid))-np.array(sigma1(grid))))     # Different Stopping condition
    if err<tol:
        print count
        break
    count+=1
    if np.mod(count,25)==0:
        print count
    v0=v1
    sigma0=sigma1
optcons2=LinInterp(grid,grid-sigma0(grid))
print('it took %2.2f seconds' %(time.time()-start))

plt.figure(5)
plt.plot(grid,sigma1(grid))
plt.draw()
plt.plot(grid,optpolicy(grid))
plt.savefig(dir+'OptSaving2.eps')
plt.draw()

plt.figure(9)
plt.plot(grid,optcons(grid))
plt.savefig(dir+'OptCons2.eps')
plt.draw()
plt.plot(grid,optcons2(grid))
plt.savefig(dir+'OptConsBoth.eps')
plt.draw()

plt.figure(6)
plt.ylim([-20,2])
plt.plot(grid,v1(grid))
plt.draw()
plt.plot(grid,u0(grid))
plt.savefig(dir+'OptValue.eps')
plt.draw()

##################################################################################################
# Now let's use the optimal policy from policy iteration to find the stationary distribution under that policy
##################################################################################################

t=10000                         # Number of periods to simulate in order to generate the stationary pdf
g = lambda x: R*sigma0(x)       # Interest on savings
F = lambda x, w : g(x)+w        # Wealth transition function
Finv = lambda w, x: w-g(x)      # income level for given future wealth and current interest+savings
dF = lambda y, x: 1             # Derivative of Finv wrt y

Wealth_srs=SRS(F=F,phi=phi,X=np.random.permutation(grid)[0])        # Stochastic Recursive System of Wealth
statprob2=Wealth_srs.stationaryDist(n=1000, FInv=Finv, dF=dF, phinv=dynsysf.phinv, dx=dx, xmin=0.1,xmax=gridmax)    # Stationary Probability
Ws=wgrid                        # Range of wealth in the grid

#Plot stationary PDF 
plt.figure(7)
plt.plot(Ws,statprob2)
plt.draw()
# Expected Wealth and Expected Value if initial wealth is distributed according to the stationary PDF
cons2=optcons2(Ws)                          # Consumption on Ws
EW2=sum(statprob2*Ws)                       # Expected wealth under stationary distribution
EV2=sum(statprob2*v0(Ws))                   # Expected Lifetime utility under optimal rule when initial wealth is distributed according to stationary probability
CE2=(1+theta1*(1-beta)*EV2)**(1/theta1)     # Certainty Equivalent 
print "Expected wealth=%1.4f, expected LT Utility=%1.4f, CE=%1.4f" %(EW2,EV2,CE2) 
 
##################################################################################################
# Now let's use the optimal policy from value function iteration to find the stationary distribution under that policy
##################################################################################################

t=10000                         # Number of periods to simulate in order to generate the stationary pdf
g = lambda x: R*optpolicy(x)    # Interest on savings
F = lambda x, w : g(x)+w        # Wealth transition function
Finv = lambda y, x: y-g(x)      # income level for given future wealth and current interest+savings
dF = lambda y, x: 1             # Derivative of Finv wrt y

Wealth_srs=SRS(F=F,phi=phi,X=np.random.permutation(grid)[0])    # Stochastic Recursive System of Wealth
statprob=Wealth_srs.stationaryDist(n=1000, FInv=Finv, dF=dF, phinv=dynsysf.phinv, dx=dx, xmin=0.1,xmax=gridmax) # Stationary Probability

#Plot stationary PDF 
plt.figure(8)
plt.plot(Ws,statprob)
plt.draw()

# Expected Wealth and Expected Value if initial wealth is distributed according to the stationary PDF
cons=optcons(Ws)                            # Consumption on Ws
EW=sum(statprob*Ws)                         # Expected wealth under stationary distribution
EV=sum(statprob*u0(Ws))                     # Expected Lifetime utility under optimal rule when initial wealth is distributed according to stationary probability
CE=(1+theta1*(1-beta)*EV)**(1/theta1)       # Certainty Equivalent
print "Expected wealth=%1.4f, expected LT Utility=%1.4f, CE=%1.4f" %(EW,EV,CE)

##################################################################################################
# Save results to be used by adaptive algorithm
np.savez_compressed(file,vopt=v0,vopt2=u0,EWopt=EW,EVopt=EV,CEopt=CE,EW2opt=EW2,EV2opt=EV2,CE2opt=CE2,Ws=Ws,statprob=statprob,statprob2=statprob2,optcons=cons,optcons2=cons2)
##################################################################################################
plt.show()
