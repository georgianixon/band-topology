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


sh = "/Users/Georgia Nixon/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

size=16
params = {
        'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          }
mpl.rcParams.update(params)


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
delta = 0
t2 = 0
qpoints = 300
n = 0


qlist = np.linspace(-pi,pi, qpoints, endpoint=True)
dq = qlist[1] - qlist[0]
eiglist = np.zeros((qpoints,qpoints,2), dtype=np.complex128) # for both bands
eigveclist = np.zeros((qpoints, qpoints, 2), dtype=np.complex128) # for band n

for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        eigs, evecs = eig(HGraphene(t1, t2, delta, np.array([qx, qy])))
        eiglist[xi,yi] = eigs
        eigveclist[xi,yi] = evecs[:,n]
        
eiglist = np.real(eiglist)
X, Y = np.meshgrid(qlist, qlist)

fig = plt.figure()
ax = plt.axes(projection='3d')
groundband = ax.contour3D(X, Y, eiglist[:,:,0], 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, eiglist[:,:,1], 50,cmap=cmap, norm=normaliser)
ax.set_zlabel("E")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels([r"$-\pi$", 0, r"$\pi$"])
fig.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser))
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "+str(t2)+r"$")
plt.savefig(sh + "graphene_topangle.pdf", format="pdf")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(5, 45)
ax.set_zlabel("E")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels([r"$-\pi$", 0, r"$\pi$"])
groundband = ax.contour3D(X, Y, eiglist[:,:,0], 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, eiglist[:,:,1], 50,cmap=cmap, norm=normaliser)
ax.set_zlabel("E")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser))
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "+str(t2)+r"$")
plt.savefig(sh + "graphene_sideangle.pdf", format="pdf")




#%% 

uq_Acomponant = eigveclist[:,:,0]
uq_Bcomponant = eigveclist[:,:,1]

uq_Acomponant_dqy = np.diff(uq_Acomponant)/dq
uq_Acomponant_dqx = np.diff(uq_Acomponant, axis=0)/dq
uq_Bcomponant_dqy = np.diff(uq_Bcomponant)/dq
uq_Bcomponant_dqx = np.diff(uq_Bcomponant, axis=0)/dq







