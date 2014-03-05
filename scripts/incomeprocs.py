#!/Users/omer/Library/Enthought/Canopy_64bit/User/bin/Python
# coding: utf-8
# This code Computes statistics for the income processes used in:
# Howitt, Peter and Özak, Ömer, "Adaptive Consumption Behavior" Journal of Economic Dynamics and Control, 2014, Vol. 39: 37-61 (http://dx.doi.org/10.1016/j.jedc.2013.11.003)
# Author: Ömer Özak
# email: ozak (at) smu.edu
# Date: April 2013
from __future__ import division
import numpy as np
from random import uniform
from scipy.stats import kurtosis, skew
import time,sys,os

# Seed the random number generator
np.random.seed(100)

# Sampling function
def sample(phi):
    """Returns i with probability phi[i], where phi is an
    array (e.g., list or tuple)."""
    a = 0.0
    U = uniform(0,1)  
    for i in range(len(phi)):
        if a < U <= a + phi[i]:
            return i
        a = a + phi[i]


# Income process 1
y1=np.array([.7,1,1.3])
p1=np.array([.2,.6,.2])
# Income process 2
y2=2*y1
p2=np.array([.2,.6,.2])
# Income process 3
y3=np.array([1,1.4,2,4.1])
p3=np.array([.1,.2,.6,.1])
# Income process 4
y4=np.array([0.3,0.7,1,2.1])
p4=np.array([0.05,0.25,0.6,0.1])
# Income process 4
y5=np.array([0.1,0.7,1,1.3,1.])
p5=np.array([0.05,0.15,0.6,0.15,0.05])

# Basic stats
# For each income process generate a sample of 100000 to approximate distribution and use python tools
n=100000
y=np.array(y1[[sample(p1) for i in range(n)]])
print 'Income Process & Mean & Std & Kurtosis & Skewness\\\\'
print('$Y^1$ & %1.2f & %1.2f & %1.2f & %1.2f \\\\' %(y.mean(),y.std(),kurtosis(y),skew(y)))
y=np.array(y2[[sample(p2) for i in range(n)]])
print('$Y^2$ & %1.2f & %1.2f & %1.2f & %1.2f \\\\' %(y.mean(),y.std(),kurtosis(y),skew(y)))
y=np.array(y3[[sample(p3) for i in range(n)]])
print('$Y^3$ & %1.2f & %1.2f & %1.2f & %1.2f \\\\' %(y.mean(),y.std(),kurtosis(y),skew(y)))
y=np.array(y4[[sample(p4) for i in range(n)]])
print('$Y^4$ & %1.2f & %1.2f & %1.2f & %1.2f \\\\' %(y.mean(),y.std(),kurtosis(y),skew(y)))
y=np.array(y5[[sample(p5) for i in range(n)]])
print('$Y^5$ & %1.2f & %1.2f & %1.2f & %1.2f \\\\' %(y.mean(),y.std(),kurtosis(y),skew(y)))

