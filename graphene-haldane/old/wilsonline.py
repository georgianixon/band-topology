# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:29:05 2021

@author: Georgia Nixon
"""
place = "Georgia Nixon"

import numpy as np
from numpy import sin, cos, pi, sqrt, exp
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.linalg import expm
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
from EulerClassHamiltonian import  GetEvalsAndEvecs

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD-MBQD-WS-1/Notes/Topology Bloch Bands/"

# def getevalsandevecs(HF):
#     #order by evals, also order corresponding evecs
#     evals, evecs = eig(HF)
#     idx = np.real(evals).argsort()
#     evals = evals[idx]
#     evecs = evecs[:,idx]
    
#     #make first element of evecs real and positive
#     for vec in range(np.size(HF[0])):
#         phi = np.angle(evecs[0,vec])
#         evecs[:,vec] = exp(-1j*phi)*evecs[:,vec]
        
# #        evecs[:,vec] = np.conj(evecs[0,vec])/np.abs(evecs[0,vec])*evecs[:,vec]
        
#         #nurs normalisation
#         evecs[:,vec] = np.conj(evecs[1,vec])/np.abs(evecs[1,vec])*evecs[:,vec]
#     return evals, evecs


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

def DifferenceLine(array2D):
    X = np.append(np.append(array2D[[-2]], array2D, axis=0), array2D[[1]], axis=0)
    xDiff = np.zeros((len(array2D), 2))
    for i in range(len(array2D)):
        xDiff[i] = np.array([X[i+2,0] - X[i,0], X[i+2,1] - X[i,1]])
    return xDiff

def GrapheneHamiltonian(k, phi, M, t1, t2):
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
    H[0,1]= -t1*(cosasum+1j*sinasum)
    H[1,0]=np.conj(H[0,1]);
    return H


def CalculateBerryConnectGraphene(k, phi, M, t1, t2, n0, n1):
    
    h = 0.0001;

    H = GrapheneHamiltonian(k, phi, M, t1, t2)
    
    d0,v0 = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0=v0[:,n0]
    u1=v0[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = GrapheneHamiltonian(kxx, phi, M, t1, t2)
    dx,vx = GetEvalsAndEvecs(H)
    ux1 = vx[:,n1]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = GrapheneHamiltonian(kyy, phi, M, t1, t2)
    dy,vy = GetEvalsAndEvecs(H)
    uy1=vy[:,n1]

    xder = (ux1-u1)/h
    yder = (uy1-u1)/h
    
    berryconnect = 1j*np.array([np.dot(np.conj(u0),xder),np.dot(np.conj(u0),yder)])

    return berryconnect

def CalculateBerryConnectMatrixGraphene(k, phi, M, t1, t2):
    dgbands=2
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnectGraphene(k, phi, M, t1, t2, n0, n1)
    return berryConnect

phi = pi/3
t1=1
t2=0.6
M = t2*3*sqrt(3) * sin(phi)-0.1

#reciprocal lattice vectors
c1 = (2*pi/(3*alength))*np.array([1, sqrt(3)])
c2 = (2*pi/(3*alength))*np.array([1, -sqrt(3)])

#think u are qpoints?
dlt = 0.005
qpoints=51

nline = 51
wilsonline00 =  np.zeros([nline], dtype=np.complex128)
wilsonline00abelian = np.zeros([nline], dtype=np.complex128)

#step for abelian version
#find u at first k
H = GrapheneHamiltonian(np.array([0,0]), phi, M, t1, t2)
_, evecs = GetEvalsAndEvecs(H)
uInitial = evecs[:,0]
    
# go through possible end points for k
for i, kend in enumerate(np.linspace(0,3,nline, endpoint=True)):
    
    # u10 is amout we are going down the line from \Gamma to \Gamma + 3G
    u10 = np.linspace(0, kend, qpoints, endpoint=True)
    # kline is q values for various points between \Gamma and kend (max being \Gamma + 3G)
    kline = np.outer(u10,c1)
    
    berryconnect00 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect01 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect10 = np.zeros([qpoints, 2], dtype=np.complex128)
    berryconnect11 = np.zeros([qpoints, 2], dtype=np.complex128)
    
    

    for cnt, k in enumerate(kline):
        #calculate berry connection at each of the k points on the k line
        berryconnect00[cnt] = CalculateBerryConnectGraphene(k, phi, M, t1, t2, 0, 0)
        berryconnect01[cnt] = CalculateBerryConnectGraphene(k, phi, M, t1, t2, 0, 1)
        berryconnect10[cnt] = CalculateBerryConnectGraphene(k, phi, M, t1, t2, 1, 0)
        berryconnect11[cnt] = CalculateBerryConnectGraphene(k, phi, M, t1, t2, 1, 1)
        
       
    
    dq = kline[1]-kline[0]
    wilsonline = np.zeros([2,2], dtype=np.complex128)
    wilsonline[0,0] = np.sum(1j*np.dot(berryconnect00, dq))
    wilsonline[0,1] = np.sum(1j*np.dot(berryconnect01, dq))
    wilsonline[1,0] = np.sum(1j*np.dot(berryconnect10, dq))
    wilsonline[1,1] = np.sum(1j*np.dot(berryconnect11, dq))
    
    wilsonline = expm(wilsonline)
#    evals, _ = getevalsandevecs(wilsonline)
    wilsonline00[i]=wilsonline[0,0]
    
    #do abeliean version,
    #find u at other k down the line
    H = GrapheneHamiltonian(kline[-1], phi, M, t1, t2)
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,0]
    wilsonline00abelian[i] = np.dot(np.conj(uFinal), uInitial)

#%%

# new way
wilsonLineNonAbelian = np.zeros([qpoints, 2, 2], dtype=np.complex128)
currentArgument = np.zeros([2,2], dtype=np.complex128)

u10 = np.linspace(0,3,qpoints)
kline = np.outer(u10,c1)
dq = kline[1] - kline[0]

for i, kpoint in enumerate(kline):
    berryConnect = CalculateBerryConnectMatrixGraphene(kpoint, phi, M, t1, t2)
    berryConnectAlongKLine =  1j*np.dot(berryConnect, dq)
    currentArgument = currentArgument + berryConnectAlongKLine
    wilsonLineNonAbelian[i] = expm(currentArgument)



#%%


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.linspace(0,3,qpoints, endpoint=True), np.square(np.abs(wilsonline00abelian)), label=r"abelian $<u_{q_i}^n | u_{q_f}^m>$")
ax.plot(np.linspace(0,3,qpoints, endpoint=True), np.square(np.abs(wilsonLineNonAbelian[:,0,0])),  'x', label=r"non abelian simple method")
ax.plot(np.linspace(0,3,qpoints, endpoint=True), np.square(np.abs(wilsonline00)), label=r'non abelian $ \exp [\Pi_{n=1}^{N} i \Delta_{\mathbf{q}} \cdot \mathbf{A}(\mathbf{q}_n)]$')
ax.set_ylabel(r"$|W[0,0]|^2$")
ax.set_xlabel(r"Final quasimomentum (in units of $\vec{G}$ away from $\Gamma$ )")
plt.legend()
plt.show()    

 