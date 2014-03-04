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

# Seed the random number generator
np.random.seed(100)

# Output directory
dir=os.getenv("HOME")+'/Dropbox/LatexMe/Consumption/data/LogN/'
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
file=dir+'optcons'

# Let's replicate the Howitt Ozak parameter's
theta, beta= 3, 0.9     # Preference Parameters
#p=np.array([0.2, 0.6,0.2])        # Probability if income value i
#y=np.array([0.7,1,1.3])             # Income values
R=1                     # Gross Interest rate
sigman=0.18               # Std of log-income

# auxiliary parameters and functions
theta1=1-theta
rho=beta

# Assume income process is log-normal with mean 1 and std 0.1
y = exp(sigman*randn(1000))                   # Draws of shock 

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
def bellman(w):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
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
    """
    vals = []
    for W in grid:
        h = lambda k: U(max(W - k,0),theta1) + rho * mean(w(f(k,y)))
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
        

start=time.time()
# Finding the approximate value function using value function iteration
u0=LinInterp(grid,U(grid,theta1))
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

optpolicy=policy(u0)
optcons=LinInterp(grid,grid-optpolicy(grid))
print('error is %2.14f' % np.max(np.abs(np.array(u1(grid))-np.array(u0(grid)))))
print('it took %2.2f seconds to compute' % (time.time()-start))
# Now let's find the approximate optimal policy and draw it
plt.figure(2)
#plt.ylim([0,2])
#plt.xlim([0,2])
plt.plot(grid,grid)
plt.plot(grid,optpolicy(grid)+mean(y))
plt.savefig(dir+'OptSaving.eps')
plt.draw()

plt.figure(3)
#plt.ylim([0,2])
#plt.xlim([0,2])
#plt.plot(grid,grid)
plt.plot(grid,optcons(grid))
plt.savefig(dir+'OptCons.eps')
plt.draw()
'''
# Now let's use the optimal policy to find the stationary distribution under that policy
t=100000        # Number of periods to simulate in order to generate the stationary pdf
dx=0.01          # Gridcell size for pdf 
g = lambda x: R*optpolicy(x)    # Interest on savings
F = lambda x, w : g(x)+w        # Wealth transition function
Finv = lambda y, x: y-g(x)      # income level for given future wealth and current interest+savings
dF = lambda y, x: 1             # Derivative of Finv wrt y
phinv=lognorm.pdf               # PDF of the income process

# Wealth's Stochastic Recursive System
Wealth_srs=SRS(F=F,phi=np.random.lognormal,X=np.random.permutation(grid)[0],mu=0,sigma=sigman)
# Stationary PDF for wealth 
statprob=Wealth_srs.stationaryDist(n=t, FInv=Finv, dF=dF, phinv=phinv, dx=dx) # Outcome of MonteCarlo
xmin=0   # Grid initial w0
xmax=5   # Maximum wealth
Ws=np.arange(xmin,xmax,dx)      # Range of wealth in the grid
dists=LinInterp(Ws,statprob)     # Create a linear interpolation based on the grid 
#statpdf=LinInterp(grid,dists(grid))  # Now use original grid values in order to get the stationary pdf for them and initialize a linear interpolation based on them

#Plot stationary PDF 
plt.figure(4)
plt.plot(Ws,statprob)
plt.draw()

# What does the stationary distribution look like before 100000

statprob2=[Wealth_srs.stationary(y=y, n=500, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob2/sum(statprob2))
plt.draw()

statprob3=[Wealth_srs.stationary(y=y, n=100, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob3/sum(statprob3))
plt.draw()

statprob4=[Wealth_srs.stationary(y=y, n=50, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob4/sum(statprob4))
plt.draw()

statprob5=[Wealth_srs.stationary(y=y, n=10, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob5/sum(statprob5))
plt.draw()

# Expected Wealth and Expected Value if initial wealth is distributed according to the stationary PDF
cons=optcons(Ws)    # Consumption on Ws
EW=sum(statprob*Ws) # Expected wealth under stationary distribution
EV=sum(statprob*u0(Ws)) # Expected Lifetime utility under optimal rule when initial wealth is distributed according to stationary probability
CE=(1+theta1*(1-beta)*EV)**(1/theta1)
print "Expected wealth=%1.4f, expected LT Utility=%1.4f, CE=%1.4f" %(EW,EV,CE) 
 
plt.show()
'''
# Now let us construct the approximate optimal policies and value functions
# using policy iteration
plt.figure(4)
start=time.time()
sigma0=LinInterp(grid,0.5*grid)
v0=LinInterp(grid,U(0.5*grid,theta1))
count=0
while count<maxiter:
    v1=get_value(sigma0,v0)
    sigma1=policy(v1)
    plt.plot(grid,sigma1(grid))
    plt.draw()
    err=2*beta/((1-beta)**2)*np.max(np.abs(np.array(v1(grid))-np.array(v0(grid))))
    #err=2*beta/((1-beta)**2)*np.max(np.abs(np.array(sigma0(grid))-np.array(sigma1(grid))))
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

# Now let's use the optimal policy to find the stationary distribution under that policy
t=100000        # Number of periods to simulate in order to generate the stationary pdf
dx=0.01          # Gridcell size for pdf 
g = lambda x: R*sigma0(x)    # Interest on savings
F = lambda x, w : g(x)+w        # Wealth transition function
Finv = lambda y, x: y-g(x)      # income level for given future wealth and current interest+savings
dF = lambda y, x: 1             # Derivative of Finv wrt y
phinv=lognorm.pdf               # PDF of the income process

# Wealth's Stochastic Recursive System
Wealth_srs2=SRS(F=F,phi=np.random.lognormal,X=np.random.permutation(grid)[0],mu=0,sigma=sigman)
# Stationary PDF for wealth 
statprob2=Wealth_srs2.stationaryDist(n=t, FInv=Finv, dF=dF, phinv=phinv, dx=dx) # Outcome of MonteCarlo
xmin=0   # Grid initial w0
xmax=5   # Maximum wealth
Ws=np.arange(xmin,xmax,dx)      # Range of wealth in the grid
dists2=LinInterp(Ws,statprob2)     # Create a linear interpolation based on the grid 
#statpdf=LinInterp(grid,dists(grid))  # Now use original grid values in order to get the stationary pdf for them and initialize a linear interpolation based on them

#Plot stationary PDF 
plt.figure(7)
plt.plot(Ws,statprob2)
plt.draw()
# Expected Wealth and Expected Value if initial wealth is distributed according to the stationary PDF
cons2=optcons2(Ws)    # Consumption on Ws
EW2=sum(statprob2*Ws) # Expected wealth under stationary distribution
EV2=sum(statprob2*v0(Ws)) # Expected Lifetime utility under optimal rule when initial wealth is distributed according to stationary probability
CE2=(1+theta1*(1-beta)*EV2)**(1/theta1)
print "Expected wealth=%1.4f, expected LT Utility=%1.4f, CE=%1.4f" %(EW2,EV2,CE2) 
 
###########

# Now let's use the optimal policy to find the stationary distribution under that policy
t=100000        # Number of periods to simulate in order to generate the stationary pdf
dx=0.01          # Gridcell size for pdf 
g = lambda x: R*optpolicy(x)    # Interest on savings
F = lambda x, w : g(x)+w        # Wealth transition function
Finv = lambda y, x: y-g(x)      # income level for given future wealth and current interest+savings
dF = lambda y, x: 1             # Derivative of Finv wrt y
phinv=lognorm.pdf               # PDF of the income process

# Wealth's Stochastic Recursive System
Wealth_srs=SRS(F=F,phi=np.random.lognormal,X=np.random.permutation(grid)[0],mu=0,sigma=sigman)
# Stationary PDF for wealth 
statprob=Wealth_srs.stationaryDist(n=t, FInv=Finv, dF=dF, phinv=phinv, dx=dx) # Outcome of MonteCarlo
xmin=0   # Grid initial w0
xmax=5   # Maximum wealth
Ws=np.arange(xmin,xmax,dx)      # Range of wealth in the grid
dists=LinInterp(Ws,statprob)     # Create a linear interpolation based on the grid 
#statpdf=LinInterp(grid,dists(grid))  # Now use original grid values in order to get the stationary pdf for them and initialize a linear interpolation based on them

#Plot stationary PDF 
plt.figure(8)
plt.plot(Ws,statprob)
plt.draw()

# What does the stationary distribution look like before 100000

statprob2=[Wealth_srs.stationary(y=y, n=500, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob2/sum(statprob2))
plt.draw()

statprob3=[Wealth_srs.stationary(y=y, n=100, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob3/sum(statprob3))
plt.draw()

statprob4=[Wealth_srs.stationary(y=y, n=50, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob4/sum(statprob4))
plt.draw()

statprob5=[Wealth_srs.stationary(y=y, n=10, FInv=Finv, dF=dF, phinv=phinv) for y in np.arange(xmin,xmax,dx)]
plt.plot(Ws,statprob5/sum(statprob5))
plt.draw()

# Expected Wealth and Expected Value if initial wealth is distributed according to the stationary PDF
cons=optcons(Ws)    # Consumption on Ws
EW=sum(statprob*Ws) # Expected wealth under stationary distribution
EV=sum(statprob*u0(Ws)) # Expected Lifetime utility under optimal rule when initial wealth is distributed according to stationary probability
CE=(1+theta1*(1-beta)*EV)**(1/theta1)
print "Expected wealth=%1.4f, expected LT Utility=%1.4f, CE=%1.4f" %(EW,EV,CE) 
########
np.savez_compressed(file,vopt=v0,vopt2=u0,EWopt=EW,EVopt=EV,CEopt=CE,EW2opt=EW2,EV2opt=EV2,CE2opt=CE2,Ws=Ws,statprob=statprob,statprob2=statprob2,optcons=cons,optcons2=cons2)
plt.show()
'''
'''
