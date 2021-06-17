# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:03:44 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import cos, sin, exp
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology')
from eulerclass import  EulerHamiltonian, GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import norm

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

#reciprocal lattice vectors
c1 = 1*np.array([1, 0])
c2 = 1*np.array([0, 1])

cvec = c1-c2
#think u are qpoints?
qpoints=201

wilsonline00abelian = np.zeros(qpoints, dtype=np.complex128)

#step for abelian version
#find u at first k
H = EulerHamiltonian(0,0)
_, evecs = GetEvalsAndEvecs(H)
u1 = evecs[:,0]
u2 = evecs[:,1]
u3 = evecs[:,2]
multiplier = np.linspace(0, 2, qpoints, endpoint=True)
kline = np.outer(multiplier, cvec)

thetasLine = np.zeros(qpoints, dtype=np.complex128)
alphasLine = np.zeros(qpoints, dtype=np.complex128)
psisLine = np.zeros(qpoints, dtype=np.complex128)
phisLine = np.zeros(qpoints, dtype=np.complex128)

# go through possible end points for k, get andlges
for i, kpoint in enumerate(kline):
    #do abeliean version,
    #find u at other k down the line
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,0]
    
    #get theta
    theta = 2*np.arcsin(np.real(np.dot(np.conj(u1), uFinal)))
    alpha = 2*np.arccos(np.linalg.norm(np.dot(np.conj(u2), uFinal)/(cos(theta/2))))
    g = np.dot(np.conj(u2), uFinal)/(cos(theta/2)*cos(alpha/2))
    g1 = np.dot(np.conj(u3), uFinal)/(cos(theta/2)*sin(alpha/2))
    psi = -np.angle(g/g1)
    phi = np.angle(g1/(exp(1j*psi/2)))
    
    wilsonline00abelian[i] = np.dot(np.conj(uFinal), u1)
    
    thetasLine[i] = theta
    alphasLine[i] = alpha
    psisLine[i] = psi
    phisLine[i] = phi





#%%
fs = (8,6)
fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, thetasLine, label=r"$\theta$")
ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (1,-1)$ away from $\Gamma$ )")
plt.legend()
plt.savefig(sh+ "thetas-v=(1,-1).pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, alphasLine, label=r"$\alpha$")
ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (1,-1)$ away from $\Gamma$ )")
plt.legend()
plt.savefig(sh+ "alphas-v=(1,-1).pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, psisLine, label=r"$\psi$")
ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (1,-1)$ away from $\Gamma$ )")
plt.legend()
plt.savefig(sh+ "psis-v=(1,-1).pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, phisLine, label=r"$\phi$")
ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (1,-1)$ away from $\Gamma$ )")
plt.legend()
plt.savefig(sh+ "phis-v=(1,-1).pdf", format="pdf")
plt.show()    


