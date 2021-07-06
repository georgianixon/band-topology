# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:03:44 2021

@author: Georgia Nixon
"""

place = "Georgia"
import numpy as np
from numpy import cos, sin, exp, pi, tan
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology')
from EulerClassHamiltonian import  EulerHamiltonian
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"


def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  [(cos(x)*r+centre[0],sin(x)*r+centre[1]) for x in np.linspace(0, 2*pi, points, endpoint=True)]
    #get rid of duplicates
    CircleLine = list(dict.fromkeys(CircleLine) )
    return CircleLine

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints)
    return kline

def GetEvalsAndEvecs(HF, realPositive = 0):
    """
    Get e-vals and e-vecs of Haniltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        phase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = phase*evecs[:,vec]

    
    if np.all((np.round(np.imag(evals),7) == 0)) == True:
        return np.real(evals), evecs
    else:
        print('evals are imaginary!')
        return evals, evecs

def AlignEvecs(evecs0, evecsP, N):
    """
    """
    return evecsP

def OrderEvecs(evecs0, N):
    """
    Make first nonzero element of evecs real and positive
    """
    evecs0_R = roundcomplex(evecs0, 5)
    for vec in range(N):
        #index of first non zero element of this vector
        firstnonzero = (evecs0_R[:,vec]!=0).argmax()
        #make this value real and positive, so arg(this value)=2*pi*n for integer n
        angle = np.angle(evecs0[firstnonzero,vec])
        evecs0[:,vec] = evecs0[:,vec]*exp(-1j*angle)
    return evecs0


qpoints=51

kline = CreateCircleLine(0.5, qpoints)

# kline0 = CreateLinearLine(0.5, 0, 0.5, 2,  qpoints)
# kline1 = CreateLinearLine(0.5, 2, 1.5, 2, qpoints)
# kline2 = CreateLinearLine(1.5, 2, 1.5, 0, qpoints)
# kline3 = CreateLinearLine(1.5, 0, 0.5, 0, qpoints)
# kline =np.vstack((kline0,kline1,kline2, kline3))

totalPoints = len(kline)

#step for abelian version
#find u at first k
k0 = kline[0]
H = EulerHamiltonian(k0[0],k0[1])
_, evecs = GetEvalsAndEvecs(H)
u0 = evecs[:,0]
u1 = evecs[:,1]
u2 = evecs[:,2]

thetasLine = np.zeros(totalPoints, dtype=np.complex128)
alphasLine = np.zeros(totalPoints, dtype=np.complex128)
psisLine = np.zeros(totalPoints, dtype=np.complex128)
phisLine = np.zeros(totalPoints, dtype=np.complex128)

# go through possible end points for k, get andlges
for i, kpoint in enumerate(kline):
    #do abeliean version,
    #find u at other k down the line
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H, realPositive=2)
    uFinal = evecs[:,2]
    
    #get theta
    # check right element is real and positive
    assert(np.imag(np.dot(np.conj(u2), uFinal))==0)
    theta = 2*np.arcsin(np.real(np.dot(np.conj(u2), uFinal)))
    alpha = 2*np.arcsin(np.linalg.norm(np.dot(np.conj(u1), uFinal)/(cos(theta/2))))
    expIPsi = np.dot(np.conj(u1), uFinal)/(np.dot(np.conj(u0), uFinal)*tan(alpha/2))
    psi = np.angle(expIPsi)
    expIPhi = np.dot(np.conj(u0), uFinal)*exp(1j*psi/2)/(cos(theta/2)*cos(alpha/2))
    phi = np.angle(expIPhi)

    
    thetasLine[i] = theta
    alphasLine[i] = alpha
    psisLine[i] = psi
    phisLine[i] = phi





#%%
#plot kline
multiplier = np.linspace(0, 4, totalPoints)
fs = (8,6)
x,y = zip(*kline)
fig, ax = plt.subplots(figsize=fs)
ax.plot(x, y, label=r"k line")
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$q_y$", rotation=0, labelpad=15)
# plt.savefig(sh+ "CircleTrajectory.pdf", format="pdf")
plt.show()    


fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(thetasLine), label=r"$\theta$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_ylabel(r"$\theta$")
plt.savefig(sh+ "thetasSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(alphasLine), label=r"$\alpha$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_ylabel(r"$\alpha$")
plt.savefig(sh+ "alphasSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(phisLine), label=r"$\phi$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_ylabel(r"$\phi$")
plt.savefig(sh+ "phisSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    


fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(psisLine), label=r"$\psi$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_ylabel(r"$\psi$")
plt.savefig(sh+ "psisSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    



