#!/Users/omer/Library/Enthought/Canopy_64bit/User/bin/Python
# coding: utf-8
# This code generates the dynamics of the learning algorithm if agents have finite lives for the paper:
# 1. Özak, Ömer, "Optimal consumption under uncertainty, liquidity constraints, and bounded rationality", Journal of Economic Dynamics and Control, 2014, Vol. 39: 237-254 (http://dx.doi.org/10.1016/j.jedc.2013.12.007)
# It can be used to generate similar results in
# Howitt, Peter and Özak, Ömer, "Adaptive Consumption Behavior" Journal of Economic Dynamics and Control, 2014, Vol. 39: 37-61 (http://dx.doi.org/10.1016/j.jedc.2013.11.003)
# Although it was not written for that purpose and needs some changes (see comments below)
# The program is not fully optimized. Instead it is written in order to maximize readibility, understanding, and replicability.
# Should work on most Python distributions. Tested on Enthought Canopy 1.3, Python.org 2.7.6 + Numpy 1. + Scipy 1.10
# Feel free to use the code and play with parameters
# Author: Ömer Özak
# email: ozak (at) smu.edu
# Date: April 2013
from __future__ import division
from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound
from dynsys import *        # From listing 6.4
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm
from scipy.interpolate import interp2d as lininterp2
from scipy.stats.mstats import mquantiles
import time,sys,os
from scipy.stats import ttest_ind

# Setup to make permanent income inverse Ushaped or constant
constant=1

# Choose type of learning algorithm
HO=0                            # Use original HO-algorithm matrix =1
alg='zero'                      # Type of algorithm: zero, backward
#alg='backward'                  # Type of algorithm: zero, backward
yp1='YP1/'
#yp1=''

# Parameters for varying the algorithm
Newt=0                      # Flag for using full Hessian
fr=1                        # Fractional step size
gain=0                      # Gain parameter
qzero = 1.0e-003            # Quasi-zero in steepest descent + shrinking

# Seed the random number generator
np.random.seed(100)

# Output directory
dir='../data/'
filelin=dir+'lincons.npz'
dir=dir+'T/'+yp1+'HOAC/'
if os.path.exists(dir[0:len(dir)-5-len(yp1)])==False:
    os.mkdir(dir[0:len(dir)-5-len(yp1)])
if os.path.exists(dir[0:len(dir)-5])==False:
    os.mkdir(dir[0:len(dir)-5])
if os.path.exists(dir)==False:
    os.mkdir(dir)
if constant==0:
    dir=dir+'constant/'
    if os.path.exists(dir)==False:
        os.mkdir(dir)
fileopt=dir+'optconsT.npz'
fileout=dir+'learnoptcons.npz'
dir=dir+alg+'/'+str(HO)+'/'
if os.path.exists(dir[0:len(dir)-2])==False:
    os.mkdir(dir[0:len(dir)-2])
if os.path.exists(dir)==False:
    os.mkdir(dir)
fileout=dir+'learnoptcons.npz'


# Sample size and time horizon
N=100000
T=60

# Let's replicate the Howitt Ozak (2014) parameter's
theta, beta= 3, 0.9                 # Preference Parameters
p=np.array([0.2, 0.6,0.2])          # Probability if income value i
y1=np.array([0.7,1,1.3])            # Income values
R=1                                 # Gross Interest rate

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
dx=0.01          # Gridcell size for pdf
wgrid=np.arange(0.1,gridmax,dx)
Ws=wgrid

# Lifespan is TT+1, i.e. agent dies in period T+1
TT=T

# auxiliary parameters and functions
theta1=1-theta
rho=beta

# Assume income process is U shaped + log-normal shock with mean 1 and std 0.18
yt=np.array(y1[[sample(p) for i in range(N*TT)]])           # Transitory income shock
yt=np.reshape(yt2,(N,TT))
if yp1=='':
    yp = .6-constant*1.5*(np.arange(0,1,1/(TT+2))-.5)**2
else:
    yp = 1-constant*(np.arange(0,1,1/(TT+2))-.5)**2

# Parameters of the linear consumption function
agrid=np.arange(0.01,1.5,0.01)       # Intercept
bgrid=np.arange(0.01,1,0.01)       # MPC: Marginal propensity to consume out of wealth

# Parameters for varying the algorithm
Newt=0                      # Flag for using full Hessian
fr=1                        # Fractional step size
gain=0                      # Gain parameter
HO=0                        # Use original HO-algorithm
qzero = 1.0e-003            # Quasi-zero in steepest descent + shrinking

# Parameters for the optimization procedures
count=0
maxiter=1000
tol=1e-12
print 'tol=%1.10f' % tol

# Define CRRA Utility function and its derivatives
def U(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return ln(c)
    else:
        return (c**theta1-1)/theta1 

def U1(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return 1/c
    else:
        return c**(theta1-1) 

def U2(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return -1/c**2
    else:
        return (theta1-1)*c**(theta1-2) 

def U3(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return 2/c**3
    else:
        return (theta1-1)**(theta1-2)*c**(theta1-3) 

# Wealth transition equation (s is savings in period t)
def f(s, y): return R*s+y     

# Consumption function as a function of intercept, MCP, and wealth
def c(a,w):
    '''Computes min(w,a0+a1*w)'''
    if type(w)!=np.ndarray:
        w=np.array(w)
    if type(a)!=np.ndarray:
        a=np.array(a)
    return np.min([w,a[0]+w*a[1]],axis=0)

# Notional unrestricted Consumption function as a function of intercept, MCP, and wealth
def chat(a,w):
    '''Computes min(w,a0+a1*w)'''
    if type(w)!=np.ndarray:
        w=np.array(w)
    if type(a)!=np.ndarray:
        a=np.array(a)
    return a[0]+w*a[1]

def kappa(t):
    return sqrt(t)/(t+1)

def q(c):
    return R*beta*U1(c,theta1)

def wmat(w):
    if type(w)!=np.ndarray:
        return np.array([[1,w],[w,w**2]])
    else:
        return np.array([[[1,wi],[wi,wi**2]] for wi in w])

def m(u1,u2,u3,q):
    return (u2**2+Newt*u3*(u1-q))

# Grid of values for wealth over which function will be approximated
gridmax, gridsize = 5, 300
grid = linspace(0.01, gridmax**1e-1, gridsize)**10

# Load data from optimal consumption function
optimal=np.load(fileopt)
sopt=optimal['sopt']
copt=optimal['copt']

##########################################
# Choose how to initialize agents
'''
# Initialize agents Random rules
a0=np.random.choice(agrid,size=N)
b0=np.random.choice(bgrid,size=N)
w0=np.random.choice(Ws,size=N)
'''
# Initialize agents (Identical rules)
a0=0*np.ones(N)
b0=.5*np.ones(N)
w0=Ws.mean()*np.ones(N)

# Initialize Matrices to keep outputs of simulation
C=np.zeros((N,T))   # Consumption
EV=np.zeros((N,T))  # Equivalent Variation
A=np.zeros((N,T))   # Intercept
B=np.zeros((N,T))   # MPC
W=np.zeros((N,T))   # Wealth

# Store initial values
W[:,0]=w0
A[:,0]=a0
B[:,0]=b0
C[:,0]=np.array([c([a,b],w) for a,b,w in zip(a0,b0,w0)])
d=np.array([chat([a,b],w) for a,b,w in zip(a0,b0,w0)])

# Define matrix M
if HO==1:
    M=np.zeros((N,2,2))
else:
    m=10#450
    M=m*np.array([[[1,0.975],[0.975,1]] for i in range(N)])
    M=np.array([np.linalg.inv(M[i]) for i in range(N)])

# Consumption in a period for all agents
for t in range(1,T):
    if alg=='zero':
        u1=U1(C[:,t-1],theta1)
        u2=U2(C[:,t-1],theta1)*np.where(C[:,t-1]<W[:,t-1],1,0)
        u3=U3(C[:,t-1],theta1)   
    elif alg=='backward':
        u1=U1(d,theta1)
        u2=U2(d,theta1)
        u3=U3(d,theta1)
    W[:,t]=f(W[:,t-1]-C[:,t-1],yp[t]*yt[:,t])
    C[:,t]=np.array([c([a,b],w) for a,b,w in zip(A[:,t-1],B[:,t-1],W[:,t])])
    d=np.array([chat([a,b],w) for a,b,w in zip(A[:,t-1],B[:,t-1],W[:,t])])
    Q=q(C[:,t])
    if HO==1:
        m0=m(u1,u2,u3,Q)
        m0=np.array([m0[i]*wmat(W[i,t-1]) for i in range(N)])
        M=(1-gain)*M+m0
        M=np.array([(log(np.linalg.cond(M[i]))<10)*M[i]+(log(np.linalg.cond(M[i]))>=10)*np.diag(np.diagonal(M[i])) for i in range(N)])
        Minv=np.array([np.linalg.inv(M[i]) for i in range(N)])
        A2=np.array( [ (u2[i]*(Q[i]-u1[i])) * np.dot(Minv[i], np.array([1,W[i,t-1]]) ) for i in range(N) ] )
        A2=np.array([A[:,t-1],B[:,t-1]]).transpose()+kappa(t)*A2
    else:
        A2=np.array( [ (u2[i]*(Q[i]-u1[i])) * np.dot(M[i], np.array([1,W[i,t-1]]) ) for i in range(N) ] )
        A2=np.array([A[:,t-1],B[:,t-1]]).transpose()+0.35*A2#0.35*A2#kappa(t)*A2
    # Choose what to do if new coefficients are outside of the acceptable space
    #''' Keep previous values as new values
    A2[:,0]=np.where(A2[:,0]<0,A[:,t-1],A2[:,0])
    A2[:,1]=np.where(A2[:,1]<0,B[:,t-1],A2[:,1])
    A2[:,0]=np.where(A2[:,0]>agrid.max(),A[:,t-1],A2[:,0])
    A2[:,1]=np.where(A2[:,1]>bgrid.max(),B[:,t-1],A2[:,1])
    #'''
    '''Take new values randomly from set as new values
    A2[:,0]=np.where(A2[:,0]<0,np.random.choice(agrid),A2[:,0])
    A2[:,1]=np.where(A2[:,1]<0,np.random.choice(bgrid),A2[:,1])
    A2[:,0]=np.where(A2[:,0]>agrid.max(),np.random.choice(agrid),A2[:,0])
    A2[:,1]=np.where(A2[:,1]>bgrid.max(),np.random.choice(bgrid),A2[:,1])
    #'''
    '''Take average as new values
    A2[:,0]=np.where(A2[:,0]<0,agrid.mean(),A2[:,0])
    A2[:,1]=np.where(A2[:,1]<0,bgrid.mean(),A2[:,1])
    A2[:,0]=np.where(A2[:,0]>agrid.max(),agrid.mean(),A2[:,0])
    A2[:,1]=np.where(A2[:,1]>bgrid.max(),bgrid.mean(),A2[:,1])
    '''
    A[:,t]=fr*A2[:,0]+(1-fr)*A[:,t-1]
    B[:,t]=fr*A2[:,1]+(1-fr)*B[:,t-1]

# Plot resulting paths for coefficients, wealth, consumption
plt.figure()    
plt.plot([[A[:,t].min(),A[:,t].max(),A[:,t].mean(),mquantiles(A[:,t],prob=[0.25]),mquantiles(A[:,t],prob=[0.75]),mquantiles(A[:,t],prob=[0.5])] for t in range(T)])
plt.xlabel(r'Period')
plt.ylabel(r'$\alpha^0_t$')
plt.savefig(dir+'Apath.eps')
plt.draw()
plt.figure()    
plt.plot([[B[:,t].min(),B[:,t].max(),B[:,t].mean(),mquantiles(B[:,t],prob=[0.25]),mquantiles(B[:,t],prob=[0.75]),mquantiles(B[:,t],prob=[0.5])] for t in range(T)])
plt.xlabel(r'Period')
plt.ylabel(r'$\alpha^1_t$')
plt.savefig(dir+'Bpath.eps')
plt.draw()
plt.figure()    
plt.plot([[C[:,t].min(),C[:,t].max(),C[:,t].mean(),mquantiles(C[:,t],prob=[0.25]),mquantiles(C[:,t],prob=[0.75]),mquantiles(C[:,t],prob=[0.5])] for t in range(T)])
plt.xlabel(r'Period')
plt.ylabel(r'$c_t$')
plt.savefig(dir+'Cpath.eps')
plt.draw()
plt.figure()    
plt.plot([[W[:,t].min(),W[:,t].max(),W[:,t].mean(),mquantiles(W[:,t],prob=[0.25]),mquantiles(W[:,t],prob=[0.75]),mquantiles(W[:,t],prob=[0.5])] for t in range(T)])
plt.xlabel(r'Period')
plt.ylabel(r'$w_t$')
plt.savefig(dir+'Wpath.eps')
plt.draw()

# Consumption function with average coefficients
plt.figure()    
plt.plot(Ws,c([A[:,T-1].mean(),B[:,T-1].mean()],Ws),label=r'$c^b(\overline{\alpha},w)$')
[plt.plot(Ws,c([A[:,t].mean(),B[:,t].mean()],Ws)) for t in np.arange(0,T,5)]
[plt.plot(Ws,copt.item(t)(Ws),linestyle='dashed') for t in range(T)]
plt.xlabel(r'$w_t$')
plt.ylabel(r'$c_t$')
plt.legend(loc=2)
plt.savefig(dir+'Caverage.eps')
plt.draw()

# Average Consumption function
plt.figure()    
plt.plot(Ws,np.array([c([A[i,T-1],B[i,T-1]],Ws) for i in range(N)]).mean(axis=0),label=r'$\overline{c^b(\alpha,w)}$')
[plt.plot(Ws,copt.item(t)(Ws),linestyle='dashed') for t in range(T)]
plt.xlabel(r'$w_t$')
plt.ylabel(r'$c_t$')
plt.legend(loc=2)
plt.savefig(dir+'Caverage2.eps')
plt.draw()

##############################################################################################################################
# Compute differences in life-time utility and certainty equivalents between fully rational and boundedly rational rules
##############################################################################################################################
# Life-time Utility received by agents using linear rule
ULin=np.array( [beta**t*U(C[:,t],theta1) for t in range(T)])
#Average lifetime utility across agents
ULin.sum(axis=0).mean()

# Life-time Utility received by agents under fully rational rule
Wopt=np.zeros((N,T))
Copt=np.zeros((N,T))
Wopt[:,0]=w0
Copt[:,0]=copt.item(T-1)(w0)
for t in range(1,T):
    Wopt[:,t]=f(Wopt[:,t-1]-Copt[:,t-1],yp[t]*yt[:,t])
    Copt[:,t]=copt.item(T-1-t)(Wopt[:,t])
Uopt=np.array( [beta**t*U(Copt[:,t],theta1) for t in range(T)])

# Difference in agents utility between using bounded or fully rational
Udif=ULin.sum(axis=0)-Uopt.sum(axis=0)
#tstat=Udif.mean()/Udif.std()/sqrt(N)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(Uopt.sum(axis=0).min(),Uopt.sum(axis=0).max(),0.01),np.arange(Uopt.sum(axis=0).min(),Uopt.sum(axis=0).max(),0.01),color='r',linewidth=2)
plt.scatter(ULin.sum(axis=0),Uopt.sum(axis=0))
plt.ylabel(r'$\sum_{t=0}^T \beta^tU(c^*(w_t))$')
plt.xlabel(r'$\sum_{t=0}^T \beta^tU(c^b_t(w_t))$')
plt.savefig(dir+'UoptUbounded.eps')
plt.draw()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(Uopt.sum(axis=0).min(),Uopt.sum(axis=0).max(),0.01),np.arange(Uopt.sum(axis=0).min(),Uopt.sum(axis=0).max(),0.01),color='r',linewidth=2)
plt.scatter(ULin.sum(axis=0),Uopt.sum(axis=0))
plt.xlim(ULin.sum(axis=0).mean()-.5*ULin.sum(axis=0).std(),Uopt.sum(axis=0).max())
plt.ylim(Uopt.sum(axis=0).min(),Uopt.sum(axis=0).max())
plt.ylabel(r'$\sum_{t=0}^T \beta^tU(c^*(w_t))$')
plt.xlabel(r'$\sum_{t=0}^T \beta^tU(c^b_t(w_t))$')
plt.savefig(dir+'UoptUbounded2.eps')
plt.draw()


CELin=(ULin.sum(axis=0)*(1-beta)/(1-beta**(T+1))*theta1+1)**(1/theta1)
CEOpt=(Uopt.sum(axis=0)*(1-beta)/(1-beta**(T+1))*theta1+1)**(1/theta1)

CELinmean=(ULin.sum(axis=0).mean()*(1-beta)/(1-beta**(T+1))*theta1+1)**(1/theta1)
CEOptmean=(Uopt.sum(axis=0).mean()*(1-beta)/(1-beta**(T+1))*theta1+1)**(1/theta1)

plt.figure(figsize=(10, 6))
plt.xlim(-4,Uopt.sum(axis=0).max())
plt.hist(ULin.sum(axis=0),bins=500,histtype='step',color='b',normed=True,weights=1/N*np.ones(N),label=r'$\sum_{t=0}^T \beta^tU(c^b_t(w_t))$',range=[-4,Uopt.sum(axis=0).max()])
plt.hist(Uopt.sum(axis=0),bins=500,histtype='step',color='g',normed=True,weights=1/N*np.ones(N),label=r'$\sum_{t=0}^T \beta^tU(c^*(w_t))$',range=[-4,Uopt.sum(axis=0).max()])
plt.axvline(ULin.sum(axis=0).mean(), color='b', linestyle='dashed', linewidth=2,label=r'$\frac{1}{N}\sum_{t=0}^T \beta^tU(c^b_t(w_t))$')
plt.axvline(Uopt.sum(axis=0).mean(), color='g', linestyle='dashed', linewidth=2,label=r'$\frac{1}{N}\sum_{t=0}^T \beta^tU(c^*(w_t))$')
plt.ylabel('Density')
plt.legend(loc=2)
plt.savefig(dir+'DistUoptUbounded.eps')
plt.draw()

plt.figure(figsize=(10, 6))
plt.xlim((-4,Udif.max()+.1))    
plt.hist(Udif,bins=500,histtype='step',color='b',normed=True,weights=1/N*np.ones(N),label=r'Utility Difference',range=[-4,Udif.max()+.1])
plt.axvline(Udif.mean(), color='b', linestyle='dashed', linewidth=2,label=r'Mean Utility Difference')
plt.axvline(np.median(Udif), color='k', linestyle='dotted', linewidth=2,label=r'Median Utility Difference')
plt.axvline(0, color='g', linestyle='dashed', linewidth=2)
plt.legend(loc=2)
plt.xlabel(r'$\sum_{t=0}^T \beta^t[U(c^b_t(w_t))-U(c^*(w_t))]$')
plt.ylabel('Density')
plt.savefig(dir+'DistDifUoptUbounded.eps')
plt.draw()

plt.figure(figsize=(10, 6))    
plt.xlim((CELin.mean()-2*CELin.std(),CEOpt.max()+.01))    
plt.hist(CELin,bins=500,histtype='step',color='b',normed=True,label=r'CE Bounded')
plt.hist(CEOpt,bins=500,histtype='step',color='g',normed=True,label=r'CE Optimal')
plt.axvline(CELinmean, color='b', linestyle='dashed', linewidth=2,label=r'Mean CE Bounded')
plt.axvline(np.median(CELin), color='k', linestyle='dotted', linewidth=2,label=r'Median CE Bounded')
plt.axvline(CEOptmean, color='g', linestyle='dashed', linewidth=2,label=r'Mean CE Optimal')
plt.ylabel('Density')
plt.legend(loc=2)
plt.savefig(dir+'DistCE.eps')
plt.draw()

plt.figure(figsize=(10, 6))    
plt.xlim(((CELin-CEOpt).mean()-(CELin-CEOpt).std(),(CELin-CEOpt).max()+.01))  
plt.hist(CELin-CEOpt,bins=2000,histtype='step',color='b',normed=True,label=r'CE Difference')
plt.axvline(CELinmean-CEOptmean, color='b', linestyle='dashed', linewidth=2,label=r'Mean CE Difference')
plt.axvline(np.median(CELin-CEOpt), color='r', linestyle='dotted', linewidth=2,label=r'Median CE Difference')
plt.ylabel('Density')
plt.xlabel(r'$CE^b-CE^*$')
plt.legend(loc=2)
plt.savefig(dir+'DistDifCE.eps')
plt.draw()

plt.show()
