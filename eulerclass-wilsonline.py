# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:37:33 2021

@author: Georgia
"""
place = "Georgia Nixon"
import numpy as np
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology')
from eulerclass import  EulerHamiltonian, GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import pi, cos, sin


""" should u(\Gamma) have only first eigenstate in it?"""


sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

def Normalise(v):
    norm1=np.linalg.norm(v)
    return v/norm1

def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  [(cos(x)*r+centre[0],sin(x)*r+centre[1]) for x in np.linspace(0, 2*pi, points, endpoint=True)]
    return CircleLine

def CreateLinearLine(qxEnd, qyEndm,  multiplier):
    c1 = qxEnd*np.array([1, 0])
    c2 = qyEnd*np.array([0, 1])
    cvec = c1 + c2
    kline = np.outer(multiplier, cvec)
    return kline
    
#energy levels we are considering to calculate W^{n0,n1}
n0 = 2
n1 = 2

#reciprocal lattice vectors

#think u are qpoints?
qpoints=201

wilsonline00abelian = np.zeros(qpoints, dtype=np.complex128)

#step for abelian version
#find u at first k
multiplier = np.linspace(0, 2*pi, qpoints, endpoint=True)
kline = CreateCircleLine(0.5, qpoints)
kline = CreateLinearLine()
k0 = kline[0]
H = EulerHamiltonian(k0[0],k0[1])
_, evecs = GetEvalsAndEvecs(H)
uInitial = evecs[:,n0]





#%%
# go through possible end points for k
for i, kpoint in enumerate(kline):
    
    #do abeliean version,
    #find u at other k down the line
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,n1]
    wilsonline00abelian[i] = np.dot(np.conj(uFinal), uInitial)

#%%

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(multiplier, np.square(np.abs(wilsonline00abelian)))
ax.set_ylabel(r"$|W["+str(n0) +","+str(n1)+"]|^2 = |<\Phi_{q_f}^"+str(n1)+" | \Phi_{q_i}^"+str(n0)+">|^2$")
ax.set_xticks([0, pi, 2*pi])
ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"Final quasimomentum point (going around a circle)")
# ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (\1,-1)$ away from $\Gamma$ )")
# ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{G}_2 = (0,1)$ away from $\Gamma$ )")
# plt.legend()
plt.savefig(sh+ "WilsonLineEulerCircle22.pdf", format="pdf")
# plt.savefig(sh+ "WilsonLineEuler2.pdf", format="pdf")
plt.show()    
