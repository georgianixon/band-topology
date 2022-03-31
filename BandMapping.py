# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:43:02 2022

@author: Georgia Nixon
"""

import numpy as np
from numpy import sqrt, exp, pi, cos, sin

place = "Georgia Nixon"
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
import matplotlib.pyplot as plt
from hamiltonians import GetEvalsAndEvecsGen
from datetime import datetime

# generate random floating point values
from random import seed
from random import random
# seed random number generator
seed(datetime.now())
# generate random numbers between 0-1
    
    
def Gen3Ham(delta1, delta2, delta3, omega12, omega23, omega31):
    H = np.array([[delta1, omega12, omega31],[omega12, delta2, omega23],[omega31, omega23, delta3]])
    return H

delta1 = random()
delta2 = random()

omega12 =  random()
omega23 = random()
omega31 = random()

print(omega12, omega23, omega31)

N = 100
evals = np.empty((100,3))
evec0s = np.empty((100,3))
evec1s = np.empty((100,3))
evec2s = np.empty((100,3))
delta3s = np.linspace(-0.1,0.1,100)
for i, delta3 in enumerate(delta3s):
    H = Gen3Ham(delta1, delta2, delta3, omega12, omega23, omega31)
    dx,vx = GetEvalsAndEvecsGen(H)
    evals[i] = dx
    evec0s[i] = vx[:,0]
    evec1s[i] = vx[:,1]
    evec2s[i] = vx[:,2]
    
    
fig, ax = plt.subplots()
ax.plot(delta3s, evals[:,0],'.', ms = 10, color="#090446", label="e0")
ax.plot(delta3s, evals[:,1], '.', ms=6, color = "#F71735", label="e1")
ax.plot(delta3s, evals[:,2], '.', ms = 2, color = "#FA9F42", label="e2")
plt.legend()
plt.show()


fig, ax = plt.subplots()
ax.plot(delta3s, evec0s[:,0],'.', ms = 10,label="a")
ax.plot(delta3s, evec0s[:,1],'.',  ms=6,label="b")
ax.plot(delta3s, evec0s[:,2],'.',  ms = 2, label="c")
plt.suptitle("evec0")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(delta3s, evec1s[:,0],'.', ms = 10,label="a")
ax.plot(delta3s, evec1s[:,1],'.',  ms=6,label="b")
ax.plot(delta3s, evec1s[:,2],'.',  ms = 2, label="c")
plt.suptitle("evec1")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(delta3s, evec2s[:,0],'.', ms = 10,label="a")
ax.plot(delta3s, evec2s[:,1],'.',  ms=6,label="b")
ax.plot(delta3s, evec2s[:,2],'.',  ms = 2, label="c")
plt.suptitle("evec2")
plt.legend()
plt.show()


#%%
