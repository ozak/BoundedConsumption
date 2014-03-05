#!/usr/bin/env python
'''
======================================================
Author:  Ömer Özak, 2013--2014 (ozak at smu.edu)
Website: http://omerozak.com
GitHub:  https://github.com/ozak/BoundedConsumption
======================================================
'''
from __future__ import division
from random import uniform
from numpy import ones, identity, transpose
from numpy.linalg import solve
from numpy.linalg import matrix_power as Mpower
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from scipy.stats import norm, lognorm

'''
DS is the dynamic system class
'''
class DS:
    def __init__(self , h=None, x=None, r=None):
        """Parameters: h is a function and x is a number
        in S representing the current state."""
        self.h, self.x, self.r = h, x, r
    def update(self):
        "Update the state of the system by applying h."
        if self.r==None:
            self.x = self.h(self.x)
        else:
            self.x = self.h(self.x,self.r)
    def trajectory(self , n):
        """Generate a trajectory of length n, starting
    at the current state."""
        traj = []
        for i in range(n):
            traj.append(self.x)
            self.update ()
        return traj

'''
sample(phi) samples out of distribution phi
'''

def sample(phi):
    """Returns i with probability phi[i], where phi is an
    array (e.g., list or tuple)."""
    a = 0.0
    U = uniform(0,1)  
    for i in range(len(phi)):
        if a < U <= a + phi[i]:
            return i
        a = a + phi[i]
'''
MC class defines a markov chain and samples out of it
'''

class MC:
    """For generating sample paths of finite Markov chains 
    on state space S = {0,...,N-1}."""
    
    def __init__(self, p=None, X=None):
        """Create an instance with stochastic kernel p and 
        current state X. Here p[x] is an array of length N
        for each x, and represents p(x,dy).  
        The parameter X is an integer in S."""
        self.p, self.X = p, X

    def update(self):
        "Update the state by drawing from p(X,dy)."
        self.X = sample(self.p[self.X])  

    def sample_path(self, n):
        """Generate a sample path of length n, starting from 
        the current state."""
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        return path

'''
pathprob(p,psi,X) determines the probability of having trajectory X, when X0 is distributed according to
psi and X transitions according to p
'''
def path_prob(p, psi , X): # X a sequence giving the path
    prob = psi[X[0]]
    for t in range(len(X) - 1):
        #prob = prob * p[X[t]][X[t+1]]
        if type(p)==tuple:
            prob = prob * p[X[t]][X[t+1]]
        else:
            prob = prob * p[X[t],X[t+1]]
    return prob

'''
stationary(p) computes the statinary distribution implied by a transition matrix p
'''
def stationary(p):
    ''' p is a transition matrix (K x K)
    stationary(p) returns the stationary distribution of matrix p.
    '''
    k=len(p)
    I = identity(k)                    # k by k identity matrix
    Q, b = ones((k, k)), ones((k, 1))  # Matrix and vector of ones
    A = transpose(I - p + Q)
    return solve(A,b) 

def dobrushin(p):
    ''' find the Dobrushin coefficient for a transision matrix pass p
    '''
    return min([sum(np.array([p[row1],p[row2]]).min(axis=0)) for row1 in range(len(p)) for row2 in range(len(p))])

def mindobrushin(p,T=1000):
    '''find minimum t such that p**t has a positive Dobrushin coefficient'''
    for t in range(T):
        alpha=dobrushin(Mpower(p,t))
        if alpha>0:
            break
        if (t==T) and (alpha==0):
            print('Dobrushin coefficient is zero for all t<= %i ' %T)
    return t

def createF(p):
    """Takes a kernel p on S = {0,...,N-1} and returns a
    function F(x,z) which represents it as an SRS.
    Parameters: p is a sequence of sequences , so that p[x][y]
    represents p(x,y) for x,y in S.
    Returns: A function F with arguments (x,z)."""
    S = range(len(p[0]))
    def F(x,z):
        a = 0
        for y in S:
            if a < z <= a + p[x][y]:
                return y
            a = a + p[x][y]
    return F
'''
Stochastic Recursive system
'''
class SRS:
    
    def __init__(self, F=None, phi=None, X=None, mu=None, sigma=None):
        """Represents X_{t+1} = F(X_t, W_{t+1}); W ~ phi.
        Parameters: F and phi are functions, where phi() 
        returns a draw from phi. X is a number representing 
        the initial condition."""
        self.F, self.phi, self.X, self.mu, self.sigma = F, phi, X, mu, sigma
        self.sample=[]
        self.samples=[]

    def update(self):
        "Update the state according to X = F(X, W)."
        if self.mu==None:
            self.X = self.F(self.X, self.phi())
        else:
            self.X = self.F(self.X, self.phi(self.mu,self.sigma))
        

    def sample_path(self, n):
        "Generate path of length n from current state."
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        self.sample=path
        return path
    
    def sample_paths(self, n, state):
        self.X=state
        "Generate path of length n from current state."
        path = []
        for i in range(n):
            path.append(self.X)
            self.update()
        self.sample=path
        return path
    
    def stationary(self, y=None, n=None, FInv=None, dF=None, phinv=None):
        '''Finv is the inverse function of F, such that FInv(X_{t+1},X_t)=W_{t+1}
        and dF is the derivative of the inverse function with respect to X_{t+1}
        Use this function to construct stationary distributions given a certain path
        i.e. you do not need to reconstruct the path'''
        sample=np.array(self.sample[0:n])
        if phinv==lognorm.pdf:
            dist=sum(phinv(FInv(y,sample),self.sigma)*dF(y,sample))
        else:
            dist=sum(phinv(FInv(y,sample))*dF(y,sample))
        return dist
    
    def stationaryDist(self, n=100, FInv=None, dF=None, phinv=None, dx=0.1,xmin=0,xmax=5):
        sample=np.array(self.sample_path(n))
        #print xmin,xmax
        dist=[]
        if phinv==lognorm.pdf:
            [dist.append(sum(phinv(FInv(x,sample),self.sigma)*dF(x,sample) )) for x in np.arange(xmin,xmax,dx)]
        else:
            [dist.append(sum(phinv(FInv(x,sample))*dF(x,sample) )) for x in np.arange(xmin,xmax,dx)]
        return dist/sum(dist)

class ECDF:

    def __init__(self, observations):
        self.observations = observations
        self.val=[]

    def __call__(self, x):
        '''
        this is what the function does
        counter = 0.0
        for obs in self.observations:
            if obs <= x:
                counter += 1
        return counter / len(self.observations)
        
        I implement it faster with Numpy
        '''
        return sum(np.array(self.observations)<=x)/len(self.observations)
    
    def plot(self, interval):
        self.val=[]
        #for i in range(len(interval)):
         #   self.val.append(self(interval[i]))
        [self.val.append(self(interval[i])) for i in range(len(interval))] 
        plt.plot(interval,self.val)
        plt.draw()
        return 0
        
class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.
        """
        self.X, self.Y = X, Y

    def __call__(self, z):
        """Parameters: z is a number, sequence or array.
        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.
        """
        if isinstance(z, int) or isinstance(z, float):
            return interp ([z], self.X, self.Y)[0]
        else:
            return interp(z, self.X, self.Y)


