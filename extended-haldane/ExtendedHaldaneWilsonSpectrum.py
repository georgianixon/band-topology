# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:04:07 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"

import numpy as np
from numpy import sqrt, pi
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/extended-haldane')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from ExtendedHaldaneModel import  ExtendedHaldaneHamiltonian
from hamiltonians import GetEvalsAndEvecs, getevalsandevecs
from scipy.linalg import expm
import matplotlib.pyplot as plt

def AbelianCalcWilsonLine(evecsFinal, evecsInitial, dgbands=4):
    """
    Stolen from EulerClassWilsonLine.py
    """
    wilsonLineAbelian = np.zeros([dgbands, dgbands], dtype=np.complex128)
    
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            wilsonLineAbelian[n0,n1] = np.dot(np.conj(evecsFinal[:,n1]), evecsInitial[:,n0])
        
    return wilsonLineAbelian

def CalculateBerryConnect(k, params, n0, n1):
    
    h = 0.00001;
    
    H = ExtendedHaldaneHamiltonian(k, params)
    evals, evecs = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0 = evecs[:,n0]
    u1 = evecs[:,n1]
    
    #dx direction
    H = ExtendedHaldaneHamiltonian(k +np.array([h,0]), params)
    _,evecsX = GetEvalsAndEvecs(H)
    ux1 = evecsX[:,n1]
 
    #dy direction
    H = ExtendedHaldaneHamiltonian(k + np.array([0,h]), params)
    _,evecsY = GetEvalsAndEvecs(H)
    uy1=evecsY[:,n1]

    xdiff = (ux1-u1)/h
    ydiff = (uy1-u1)/h
    
    berryConnect = 1j*np.array([np.dot(np.conj(u0),xdiff),np.dot(np.conj(u0),ydiff)])

    return berryConnect


def CalculateBerryConnectMatrix(k, params, dgbands=4):
    """
    Stolen from EulerClassWilsonLine.py
    """
    # dimension is dgbands x dgbands x 2, for dx and dy
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnect(k, params, n0, n1)
    return berryConnect
            
def DifferenceLine(array2D):
    X = np.append(np.append(array2D[[-2]], array2D, axis=0), array2D[[1]], axis=0)
    xDiff = np.zeros((len(array2D), 2))
    for i in range(len(array2D)):
        xDiff[i] = np.array([X[i+2,0] - X[i,0], X[i+2,1] - X[i,1]])
    return xDiff


#pauli matrics
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])


# rotation matrices
Q6 = np.array([[1/2, -sqrt(3)/2], [sqrt(3)/2, 1/2]])
Q3 = np.array([[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]])


# X Y lattice
#nearest neighbor vecs
a1 = np.array([0, 1])
a2 = np.array([sqrt(3)/2, -1/2])
a3 = np.array([-sqrt(3)/2, -1/2])
a = np.array([a3, a1, a2])

#n2 vec
b1 = np.array([sqrt(3), 0])
b2 = np.array([sqrt(3)/2, 3/2])
b3 = np.array([-sqrt(3)/2, 3/2])
b4 = -b1
b5 = -b2
b6 = -b3
b = np.array([b1, b2, b3, b4, b5, b6])

#n3 vecs
c1 = np.array([-sqrt(3), 1])
c2 = np.array([sqrt(3), 1])
c3 = np.array([0, -2])
c = np.array([c1, c3, c2])


#reciprocal lattice
#nearest neighbour vecs
d1 = (4*pi/3/sqrt(3))*np.array([1,0])
d2 = (4*pi/3/sqrt(3))*np.array([1/2,sqrt(3)/2])
d3 = (4*pi/3/sqrt(3))*np.array([-1/2,sqrt(3)/2])





kline = np.concatenate([np.linspace(np.array([0,0]), 0.5*d1, 100, endpoint=False),
          np.linspace(0.5*d1, 0.5*d1 + 0.5*d2, 100, endpoint=False),
          np.linspace(0.5*d1+0.5*d2,  0.5*d2, 100, endpoint=False),
          np.linspace(0.5*d2,  np.array([0,0]), 101, endpoint=True)], axis=0)

dqs = DifferenceLine(kline)



#params - Extended Haldane
M = 0.1
lambdaR = 0.3 
t1 = 1
t2 = 0.6
t3 = 0.6
params = [M, lambdaR, t1, t2, t3]
    

#%%

'''
Calculate Wilson Line - Abelian
'''

k0 = kline[0]
totalPoints = len(kline)
H = ExtendedHaldaneHamiltonian(k0, params)
_, evecsInitial = GetEvalsAndEvecs(H)

wilsonLineAbelian = np.zeros([totalPoints, 4,4], dtype=np.complex128)
# go through possible end points for k
for i, kpoint in enumerate(kline):
    #Find evec at k point, calculate Wilson Line abelian method
    H = ExtendedHaldaneHamiltonian(kpoint, params)
    _, evecsFinal = getevalsandevecs(H)
    wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsFinal, evecsInitial)

    

#%%
"""
Calculate Wilson Line - Non Abelian
"""

wilsonLineNonAbelian = np.zeros([totalPoints, 4, 4], dtype=np.complex128)
currentArgument = np.zeros([4,4], dtype=np.complex128)


for i, kpoint in enumerate(kline):
    berryConnect = CalculateBerryConnectMatrix(kpoint, params)
    dq = dqs[i]
    berryConnectAlongKLine =  1j*np.dot(berryConnect, dq)
    currentArgument = currentArgument + berryConnectAlongKLine
    wilsonLineNonAbelian[i] = expm(currentArgument)
    
    

#%%
'''
Plot
'''

m1 = 1
m2 = 1
multiplier = np.linspace(0,4,totalPoints, endpoint=True)
fig, ax = plt.subplots(figsize=(12,9))

ax.plot(multiplier, np.square(np.abs(wilsonLineNonAbelian[:,m1,m2])), label="Non Abelian")
ax.plot(multiplier, np.square(np.abs(wilsonLineAbelian[:,m1,m2])), label="Abelian")

ax.set_ylabel(r"$|W["+str(m1) +","+str(m2)+"]|^2 = |<\Phi_{q_f}^"+str(m1)+" | \Phi_{q_i}^"+str(m2)+">|^2$")
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])


# ax.set_ylim([0,1.01])
# ax.set_xlabel(r"Final quasimomentum point, going around circle with centre (1,1)")
plt.legend(loc="upper right")
# plt.savefig(sh+ "WilsonLineEulerRectangle00.pdf", format="pdf")
# plt.savefig(sh+ "WilsonLineEulerCircle2,"+str(m1)+str(m2)+".pdf", format="pdf")
plt.show()   


#%%
"""
Eigenvalue flow
""" 

evalsNonAbelian = np.empty((totalPoints, 4), dtype=np.complex128)
evalsAbelian = np.empty((totalPoints, 4), dtype=np.complex128)
for i in range(totalPoints):
    evalsNA, _ = GetEvalsAndEvecs(wilsonLineNonAbelian[i,:,:])
    evalsA, _ = GetEvalsAndEvecs(wilsonLineAbelian[i,:,:])
    evalsNonAbelian[i] = evalsNA
    evalsAbelian[i] = evalsA
    

#%%
fig, ax = plt.subplots(figsize=(12,9))
ax.plot(multiplier, np.real(evalsNonAbelian[:,0]), label="Non Abelian, first eigenvalue")
ax.plot(multiplier, np.real(evalsAbelian[:,0]), label="Abelian, first eigenvalue")
plt.legend()
plt.show()




