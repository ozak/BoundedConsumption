#!/usr/bin/env python
# coding: utf-8
'''
======================================================
Author:  Ömer Özak, 2013--2014 (ozak at smu.edu)
Website: http://omerozak.com
GitHub:  https://github.com/ozak/BoundedConsumption
======================================================
'''
from __future__ import division
import numpy as np
cimport numpy as np
from scipy import linspace, mean, exp, randn 
from scipy.optimize import fminbound 
from dynsys import *   

ctypedef np.float_t DTYPE_t

# First phinv function...not so efficient
def phinv2(np.ndarray[DTYPE_t, ndim=2] x):
    cdef np.ndarray[DTYPE_t, ndim=1] p = np.array([0.2, 0.6,0.2], dtype=np.float)        # Probability if income value i
    cdef np.ndarray[DTYPE_t, ndim=1] y1 = np.array([0.7,1,1.3], dtype=np.float)             # Income values
    cdef np.ndarray[DTYPE_t, ndim=2] p0
    cdef np.ndarray[DTYPE_t, ndim=2] mins
    cdef int xran=x.shape[0]
    cdef int i
    mins=np.min(np.abs(np.dot(np.array([x]).transpose(),np.array([np.ones_like(y1)]))-np.dot(np.array([y1]).transpose(),np.array([np.ones_like(x)])).transpose()),axis=0)
    mins=np.where(mins<.1,mins,.1)
    p0=np.sum([np.where(abs(x[0][i]-y1)==mins,p,0) for i in range(x.shape[1])],axis=1)
    #p0=np.sum([np.where(abs(x[i]-y1)<1e-12,p,0) for i in range(x.shape[0])],axis=1)
    if len(p0)==0:
        p0=np.zeros_like(x)
    return p0    # Sample out of the income process

# More efficient
def phinv(np.ndarray[DTYPE_t, ndim=1] x):
    cdef Py_ssize_t xran=x.shape[0]
    #cdef Py_ssize_t xran0=x.shape[0]
    cdef np.int i
    cdef np.ndarray[DTYPE_t, ndim=1] p = np.array([0.2, 0.6,0.2], dtype=np.float)        # Probability if income value i
    cdef np.ndarray[DTYPE_t, ndim=1] y1 = np.array([0.7,1,1.3], dtype=np.float)             # Income values
    cdef np.ndarray[DTYPE_t, ndim=1] p0 = np.zeros_like(x, dtype=np.float)
    cdef np.float min1=np.float(min(np.min(abs(x-y1[0])),0.1))
    cdef np.float min2=np.float(min(np.min(abs(x-y1[1])),0.1))
    cdef np.float min3=np.float(min(np.min(abs(x-y1[2])),0.1))
    for i in range(xran):
        if abs(x[i]-y1[0])<=min1:
            p0[i]=p[0]
        elif abs(x[i]-y1[1])<=min2:
            p0[i]=p[1]
        elif abs(x[i]-y1[2])<=min3:
            p0[i]=p[2]
    return p0    # Sample out of the income process

# Maximize function h on interval [a,b]
def maximum(object h, double a, double b):
    return np.float(h(fminbound(lambda x: -h(x), a, b)))

# Return Maximizer
def maximizer(object h, double a, double b):
    return np.float(fminbound(lambda x: -h(x), a, b))

# Define CRRA Utility function 
def U(object c, double theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return np.ln(c)
    else:
        return (c**theta1-1)/theta1 

def U2(c,theta1):
    # CRRA Utility. theta1=1-theta 
    if theta1==0:
        return np.ln(c)
    else:
        return (c**theta1-1)/theta1 

# Wealth transition equation (s is savings in period t)
def f(object s, object y, double R): 
    return R*s+y     

def f2(s, y, R): return R*s+y     


# Bellman Operator
# fast
def bellman(object w,int t,object y, double rho, double theta1,double R):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
    """
    # Grid of values for wealth over which function will be approximated
    cdef int gridmax=5 
    cdef int gridsize = 300
    cdef np.ndarray grid = linspace(0.01, gridmax**1e-1, gridsize)**10

    vals = []
    for W in grid:
        def h(float k): 
            return U(max(W - k,0),theta1) + rho * mean(w(f(k,y,R)))
        vals.append(maximum(h, 0, W))
    return LinInterp(grid, vals)

# a little slower
def bellman2(object w,int t,object y, double rho, double theta1,double R):
    """The approximate Bellman operator.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that represents the optimal operator.
    """
    # Grid of values for wealth over which function will be approximated
    cdef int gridmax=5 
    cdef int gridsize = 300
    cdef np.ndarray grid = linspace(0.01, gridmax**1e-1, gridsize)**10

    vals = []
    for W in grid:
        def h(float k): 
            return U2(max(W - k,0),theta1) + rho * mean(w(f2(k,y,R)))
        vals.append(maximum(h, 0, W))
    return LinInterp(grid, vals)



# Optimal policy
def policy(object w,int t,object y, double rho, double theta1,double R):
    """The approximate optimal policy operator w-greedy.
    Parameters: w is a vectorized function (i.e., a 
    callable object which acts pointwise on arrays).
    Returns: An instance of LinInterp that captures the optimal policy.
    """
    cdef int gridmax=5 
    cdef int gridsize = 300
    cdef np.ndarray grid = linspace(0.01, gridmax**1e-1, gridsize)**10
    
    vals = []
    for W in grid:
        def h(float k): 
            return U(max(W - k,0),theta1) + rho * mean(w(f(k,y,R)))
        vals.append(maximizer(h, 0, W))
    return LinInterp(grid, vals)

