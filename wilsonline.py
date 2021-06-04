# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:29:05 2021

@author: Georgia Nixon
"""

import numpy as np
from numpy import sin, cos, pi, sqrt, exp
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.linalg import expm

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

sh = "/Users/Georgia Nixon/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

def getevalsandevecs(HF):
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        phi = np.angle(evecs[0,vec])
        evecs[:,vec] = exp(-1j*phi)*evecs[:,vec]
        
#        evecs[:,vec] = np.conj(evecs[0,vec])/np.abs(evecs[0,vec])*evecs[:,vec]
        
        #nurs normalisation
        evecs[:,vec] = np.conj(evecs[1,vec])/np.abs(evecs[1,vec])*evecs[:,vec]
    return evals, evecs


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

alength = 1

def hamiltonian(k, phi, M, t1, t2):
    #nearest neighbor vectors
    d1 = alength*np.array([1, 0])
    d2 = alength*np.array([-1/2, sqrt(3)/2])
    d3 = alength*np.array([-1/2, -sqrt(3)/2])
    
    #second nearest neighbor vectors -> relative chirality important
    b1 = alength*np.array([3/2, sqrt(3)/2])
    b2 = alength*np.array([0,-sqrt(3)])
    b3 = alength*np.array([-3/2, sqrt(3)/2])

    cosasum = cos(np.dot(d1,k)) + cos(np.dot(d2,k)) + cos(np.dot(d3,k))
    sinasum = sin(np.dot(d1,k)) + sin(np.dot(d2, k)) + sin(np.dot(d3, k))
    cosbsum = cos(np.dot(b1, k)) + cos(np.dot(b2,k)) + cos(np.dot(b3,k))
    sinbsum = sin(np.dot(b1,k)) + sin(np.dot(b2, k)) + sin(np.dot(b3, k))
    
    H = np.zeros([2,2], dtype=np.complex128)
    H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
    H[1,1] = -M+2*t2*cos(phi)*cosbsum + 2*t2*sin(phi)*sinbsum
    H[0,1]= -t1*(cosasum+1j*sinasum);
    H[1,0]=np.conj(H[0,1]);
    return H

h = 0.0001;

def calculateberryconnect(k, phi, M, t1, t2, n0, n1):
    H = hamiltonian(k, phi, M, t1, t2)
    
    d0,v0 = getevalsandevecs(H)
    
    #first eigenvector
    u0=v0[:,n0]
    u1=v0[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = hamiltonian(kxx, phi, M, t1, t2)
    dx,vx = getevalsandevecs(H)
    ux1 = vx[:,n1]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = hamiltonian(kyy, phi, M, t1, t2)
    dy,vy = getevalsandevecs(H)
    uy1=vy[:,n1]

    xder = (ux1-u1)/h
    yder = (uy1-u1)/h
    
    berryconnect = 1j*np.array([np.dot(np.conj(u0),xder),np.dot(np.conj(u0),yder)])

    return berryconnect

phi = 3*pi/2
t1=1
t2=0.1
M = t2*3*sqrt(3) * sin(phi)-0.1

#reciprocal lattice vectors
c1 = (2*pi/(3*alength))*np.array([1, sqrt(3)])
c2 = (2*pi/(3*alength))*np.array([1, -sqrt(3)])

#think u are qpoints?
dlt = 0.005
qpoints=201


jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi

wilsonline00 =  np.zeros([31], dtype=np.complex128)

for i, kend in enumerate(np.linspace(0,3,31, endpoint=True)):
    
    u10 = np.linspace(0, kend, int(1/dlt + 1), endpoint=True)
    kline = np.outer(u10,c1)
    
    berryconnect00 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect01 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect10 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect11 = np.zeros([qpoints, 2], dtype=np.complex128)
    

    for cnt, k in enumerate(kline):
    
        berryconnect00[cnt] = calculateberryconnect(k, phi, M, t1, t2, 0, 0)
        berryconnect01[cnt] = calculateberryconnect(k, phi, M, t1, t2, 0, 1)
        berryconnect10[cnt] = calculateberryconnect(k, phi, M, t1, t2, 1, 0)
        berryconnect11[cnt] = calculateberryconnect(k, phi, M, t1, t2, 1, 1)
    
    
    
    dq = kline[1]-kline[0]
    wilsonline = np.zeros([2,2], dtype=np.complex128)
    wilsonline[0,0] = np.sum(1j*np.dot(berryconnect00, dq))
    wilsonline[0,1] = np.sum(1j*np.dot(berryconnect01, dq))
    wilsonline[1,0] = np.sum(1j*np.dot(berryconnect10, dq))
    wilsonline[1,1] = np.sum(1j*np.dot(berryconnect11, dq))
    
    wilsonline = expm(wilsonline)
    evals, _ = getevalsandevecs(wilsonline)
    wilsonline00[i]=wilsonline[0,0]

#%%


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.linspace(0,3,31, endpoint=True), wilsonline00)
ax.set_ylabel("W[0,0]")
ax.set_xlabel(r"Final quasimomentum (in units of $\vec{G}$ away from $\Gamma$ )")
plt.savefig(sh+ "WilsonLine.pdf", format="pdf")
plt.show()    

 