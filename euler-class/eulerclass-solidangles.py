# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:03:44 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import cos, sin, exp, pi, tan
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
from EulerClassHamiltonian import  EulerHamiltonian, GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm
import matplotlib as mpl

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"


size=25
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
params = {
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':True,
          "axes.facecolor":"0.9",
          "grid.color":"1"
          }

mpl.rcParams.update(params)


def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  [(cos(x)*r+centre[0],sin(x)*r+centre[1]) for x in np.linspace(0, 2*pi, points, endpoint=True)]
    #get rid of duplicates
    CircleLine = list(dict.fromkeys(CircleLine) )
    return CircleLine

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints)
    return kline


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
    #find evecs at other k down the line
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,2]

    #multiply state evec by the conjugate phase
    c = np.dot(np.conj(u2), uFinal)
    conjPhase = np.conj(c)/np.abs(c)
    uFinal = conjPhase*uFinal
    
    # make sure uFinal is in the right gauge, to 20dp
    c = np.dot(np.conj(u2), uFinal)
    assert(round(np.imag(c), 20)==0)
    
    # get params
    theta = 2*np.arcsin(np.dot(np.conj(u2), uFinal))
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
# #plot kline
# multiplier = np.linspace(0, 4, tot~alPoints)
# fs = (8,6)
# x,y = zip(*kline)
# fig, ax = plt.subplots(figsize=fs)
# ax.plot(x, y, label=r"k line")
# ax.set_xlabel(r"$q_x$")
# ax.set_ylabel(r"$q_y$", rotation=0, labelpad=15)
# ax.set_facecolor('1')
# ax.grid(b=1, color='0.6')
# # plt.savefig(sh+ "CircleTrajectory.pdf", format="pdf")
# plt.show()    


fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(thetasLine), label=r"$\theta$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, pi/2, pi])
ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
# plt.savefig(sh+ "thetasSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(alphasLine), label=r"$\alpha$")
ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# plt.legend()
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, pi/2, pi])
ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
# plt.savefig(sh+ "alphasSquareTrajectory2ndExcitedBand.pdf", format="pdf")
plt.show()    

# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier, np.real(phisLine), label=r"$\phi$")
# ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# # plt.legend()
# # ax.set_xticks([0, pi, 2*pi])
# # ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
# ax.set_ylabel(r"$\phi$")
# # plt.savefig(sh+ "phisSquareTrajectory2ndExcitedBand.pdf", format="pdf")
# plt.show()    


# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier, np.real(psisLine), label=r"$\psi$")
# ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# # plt.legend()
# # ax.set_xticks([0, pi, 2*pi])
# # ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
# ax.set_ylabel(r"$\psi$")
# # plt.savefig(sh+ "psisSquareTrajectory2ndExcitedBand.pdf", format="pdf")
# plt.show()    



