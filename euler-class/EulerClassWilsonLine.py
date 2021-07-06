# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 20:37:33 2021

@author: Georgia
"""
place = "Georgia Nixon"
import numpy as np
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
from EulerClassHamiltonian import  EulerHamiltonian, GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import pi, cos, sin
import matplotlib as mpl


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


sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"
def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def Normalise(v):
    norm1=np.linalg.norm(v)
    return v/norm1

def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  [(cos(x)*r+centre[0],sin(x)*r+centre[1]) for x in np.linspace(0, 2*pi, points, endpoint=True)]
    return CircleLine

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints)
    return kline

#energy levels we are considering to calculate W^{n0,n1}
n0 = 2
n1 = 2

#num of points to calculate the wilson Line of
qpoints = 201

# kline0 = CreateLinearLine(0.5, 0, 0.5, 2,  qpoints)
# kline1 = CreateLinearLine(0.5, 2, 1.5, 2, qpoints)
# kline2 = CreateLinearLine(1.5, 2, 1.5, 0, qpoints)
# kline3 = CreateLinearLine(1.5, 0, 0.5, 0, qpoints)
# kline =np.vstack((kline0,kline1,kline2, kline3))

kline = CreateCircleLine(0.5, qpoints)

totalPoints = len(kline)
k0 = kline[0]
H = EulerHamiltonian(k0[0],k0[1])
evals, evecs = GetEvalsAndEvecs(H)
uInitial = evecs[:,n0]

for vec in range(3):
    for entry in range(3):
        assert(RoundComplex(np.dot(EulerHamiltonian(k0[0], k0[1]), evecs[:,vec])[entry],
                            12) == RoundComplex((evals[vec]*evecs[:,vec])[entry], 12))


#%%
'''
Calculate Wilson Line
'''
wilsonLineAbelian = np.zeros(totalPoints, dtype=np.complex128)
# go through possible end points for k
for i, kpoint in enumerate(kline):
    #Find evec at k point, calculate Wilson Line abelian method
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecs = GetEvalsAndEvecs(H)
    uFinal = evecs[:,n1]
    wilsonLineAbelian[i] = np.dot(np.conj(uFinal), uInitial)


#%%
'''
Plot
'''
multiplier = np.linspace(0,4,totalPoints, endpoint=True)
fig, ax = plt.subplots(figsize=(12,9))
ax.plot(multiplier, np.square(np.abs(wilsonLineAbelian)))
ax.set_ylabel(r"$|W["+str(n0) +","+str(n1)+"]|^2 = |<\Phi_{q_f}^"+str(n1)+" | \Phi_{q_i}^"+str(n0)+">|^2$")
# ax.set_xticks([0, pi, 2*pi])
# ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_xlabel(r"Final quasimomentum point (going around square)")
# plt.savefig(sh+ "WilsonLineEulerRectangle00.pdf", format="pdf")
plt.savefig(sh+ "WilsonLineEulerCircle22.pdf", format="pdf")
plt.show()    
