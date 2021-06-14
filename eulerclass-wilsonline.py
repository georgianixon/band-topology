# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:37:33 2021

@author: Georgia
"""
import numpy as np
import sys
sys.path.append('/Users/Georgia/Code/MBQD/band-topology')
from eulerclass import  EulerHamiltonian, GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import norm

place = "Georgia"
sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD-MBQD-WS-1/Notes/Topology Bloch Bands/"

def Normalise(v):
    norm=np.linalg.norm(v)
    return v/norm

def CalculatebBerryConnect(kx,ky, band0, band1):
    H =EulerHamiltonian(kx,ky)
    d0,v0 = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0=v0[:,band0]
    u1=v0[:,band1]
    
    h = 0.0001;
    
    #dx direction
    H = EulerHamiltonian(kx+h, ky)
    dx,vx =  GetEvalsAndEvecs(H)
    ux1 = vx[:,band1]
    
    #dy direction

    H = EulerHamiltonian(kx,ky+h)
    dy,vy = GetEvalsAndEvecs(H)
    uy1=vy[:,band1]

    xder = (ux1-u1)/h
    yder = (uy1-u1)/h
    
    berryconnect = 1j*np.array([np.dot(np.conj(u0),xder),np.dot(np.conj(u0),yder)])

    return berryconnect


#reciprocal lattice vectors
c1 = 1*np.array([1, 0])
c2 = 1*np.array([0, 1])

cvec = c1 - c2
#think u are qpoints?
qpoints=201

wilsonline00abelian = np.zeros(qpoints, dtype=np.complex128)

#step for abelian version
#find u at first k
H = EulerHamiltonian(0,0)
_, evecs = GetEvalsAndEvecs(H)
uInitial = evecs[:,0]
multiplier = np.linspace(0, 2, qpoints, endpoint=True)
kline = np.outer(multiplier, cvec)


#%%
# go through possible end points for k
for i, kpoint in enumerate(kline):
    
    #do abeliean version,
    #find u at other k down the line
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,0]
    wilsonline00abelian[i] = np.dot(np.conj(uFinal), uInitial)

#%%

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(multiplier, np.square(np.abs(wilsonline00abelian)), label=r"abelian $<u_{q_i}^n | u_{q_f}^m>$")
ax.set_ylabel(r"$|W[0,0]|^2$")
ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{v} = (1,-1)$ away from $\Gamma$ )")
# ax.set_xlabel(r"Final quasimomentum (in units of $\mathbf{G}_2 = (0,1)$ away from $\Gamma$ )")
plt.legend()
# plt.savefig(sh+ "WilsonLineEuler4.pdf", format="pdf")
# plt.savefig(sh+ "WilsonLineEuler2.pdf", format="pdf")
plt.show()    
