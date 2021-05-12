# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:33:14 2021

@author: Georgia Nixon
"""

import numpy as np
from numpy import sin, cos, pi, sqrt
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib as mpl

PauliX = np.array([[0,1], [1,0]])
PauliY = 1j*np.array([[0,-1], [1,0]])
PauliZ = np.array([[1,0], [0,-1]])

normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)
cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)


A = np.array([[1,0], [-0.5, sqrt(3)/2], [-0.5,-sqrt(3)/2]])
B = np.array([[0,sqrt(3)], [-1.5,-sqrt(3)/2], [-1.5,sqrt(3)/2]])
           
def HGraphene(t1,t2, M, K):
    H0 = t1*np.sum([PauliX*cos(np.dot(K, A[i])) - PauliY*sin(np.dot(K, A[i])) for i in range(3)], axis=0)
    SLIS = M*PauliZ
    TRSB = t2*np.sum([PauliZ*sin(np.dot(K, B[i])) for i in range(3)], axis=0)
    return H0 + SLIS + TRSB

t1 = 1
M = 0
t2 = 0
qpoints = 201

qlist = np.linspace(-pi,pi, qpoints, endpoint=True)
eiglist = np.zeros((qpoints,qpoints,2), dtype=np.complex128)
for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        eigs, evecs = eig(HGraphene(t1, t2, M, np.array([qx, qy])))
        eiglist[xi,yi] = eigs
        
eiglist = np.real(eiglist)
X, Y = np.meshgrid(qlist, qlist)

fig = plt.figure()
ax = plt.axes(projection='3d')
groundband = ax.contour3D(X, Y, eiglist[:,:,0], 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, eiglist[:,:,1], 50,cmap=cmap, norm=normaliser)
fig.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser))
plt.show()

ax.view_init(5, 45)
fig


#%% 