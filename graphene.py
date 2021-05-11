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
cmap = mpl.cm.get_cmap('PuOr')


#
kx = 0
ky = 0
t = 0.1
A = np.array([[1,0], [-0.5, sqrt(3)/2], [-0.5,-sqrt(3)/2]])
K = np.array([kx, ky])
H = t*np.sum([PauliX*cos(np.dot(K, A[i])) - PauliY*sin(np.dot(K, A[i])) for i in range(3)], axis=0)
def HGraphene(t,K):
    return t*np.sum([PauliX*cos(np.dot(K, A[i])) - PauliY*sin(np.dot(K, A[i])) for i in range(3)], axis=0)
HGraphene(1, K)

t = 1
#qlist = np.linspace(-1,1,201, endpoint=True)
qpoints = 201
qlist = np.linspace(-pi,pi, 201, endpoint=True)
eiglist = np.zeros((201,201,2), dtype=np.complex128)
for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        eigs, evecs = eig(HGraphene(t, np.array([qx, qy])))
        eiglist[xi,yi] = eigs
        
eiglist = np.real(eiglist)
X, Y = np.meshgrid(qlist, qlist)

fig = plt.figure()
ax = plt.axes(projection='3d')
groundband = ax.contour3D(X, Y, eiglist[:,:,0], 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, eiglist[:,:,1], 50,cmap=cmap, norm=normaliser)
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=normaliser))
plt.show()

#ax.view_init(60, 35)
#fig

#%%