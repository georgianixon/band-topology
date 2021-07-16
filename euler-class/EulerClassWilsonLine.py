# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:55:37 2021

@author: Georgia Nixon
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
from scipy.linalg import expm


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
    norm_=np.linalg.norm(v)
    return v/norm_

def DifferenceLine(array2D):
    X = np.append(np.append(array2D[[-2]], array2D, axis=0), array2D[[1]], axis=0)
    xDiff = np.zeros((len(array2D), 2))
    for i in range(len(array2D)):
        xDiff[i] = np.array([X[i+2,0] - X[i,0], X[i+2,1] - X[i,1]])
    return xDiff

def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  np.array([[cos(x)*r+centre[0],sin(x)*r+centre[1]] for x in np.linspace(0, 2*pi, points, endpoint=True)])
    return CircleLine

def CircleDiff(points):
    """
    Gives normalised vectors tangent to the circle for various theta between 0 and 2*pi
    """
    CircleDiff = np.array([[-sin(theta),cos(theta)] for theta in np.linspace(0, 2*pi, points, endpoint=True)])
    return CircleDiff

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints, endpoint=False)
    return kline

def CalculateBerryConnect(k, n0, n1):
    
    h = 0.00001;
    
    kx = k[0]
    ky = k[1]
    H = EulerHamiltonian(kx,ky)
    evals, evecs = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0 = evecs[:,n0]
    u1 = evecs[:,n1]
    
    #dx direction
    H = EulerHamiltonian(kx+h,ky)
    _,evecsX = GetEvalsAndEvecs(H)
    ux1 = evecsX[:,n1]
 
    #dy direction
    H = EulerHamiltonian(kx,ky+h)
    _,evecsY = GetEvalsAndEvecs(H)
    uy1=evecsY[:,n1]

    xdiff = (ux1-u1)/h
    ydiff = (uy1-u1)/h
    
    berryConnect = 1j*np.array([np.dot(np.conj(u0),xdiff),np.dot(np.conj(u0),ydiff)])

    return berryConnect

def CalculateBerryConnectMatrix(k, dgbands=3):
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnect(k, n0, n1)
    return berryConnect
            
def AbelianCalcWilsonLine(evecsFinal, evecsInitial, dgbands=3):
    wilsonLineAbelian = np.zeros([dgbands, dgbands], dtype=np.complex128)
    
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            wilsonLineAbelian[n0,n1] = np.dot(np.conj(evecsFinal[:,n1]), evecsInitial[:,n0])
    return wilsonLineAbelian



#%%
"""
Define Wilson Line path (kline)
"""

#num of points to calculate the wilson Line of
qpoints = 1000
radius=0.5
centre = [0,0]

dtheta = 2*pi/(qpoints-1)
dqlength = 2*radius*sin(dtheta)

# create rectangle line
# kline0 = CreateLinearLine(0.5, 0, 0.5, 2,  qpoints)
# kline1 = CreateLinearLine(0.5, 2, 1.5, 2, qpoints)
# kline2 = CreateLinearLine(1.5, 2, 1.5, 0, qpoints)
# kline3 = CreateLinearLine(1.5, 0, 0.5, 0, qpoints)
# kline =np.vstack((kline0,kline1,kline2, kline3))

# create circle line
kline = CreateCircleLine(radius, qpoints, centre = centre)
# dqs = DifferenceLine(kline)
dqs1 = CircleDiff( qpoints)*dqlength
dqs2 = DifferenceLine(kline)
totalPoints = len(kline)



#%%
'''
Calculate Wilson Line - Abelian
'''

k0 = kline[0]
H = EulerHamiltonian(k0[0],k0[1])
_, evecsInitial = GetEvalsAndEvecs(H)

wilsonLineAbelian = np.zeros([totalPoints, 3, 3], dtype=np.complex128)
# go through possible end points for k
for i, kpoint in enumerate(kline):
    #Find evec at k point, calculate Wilson Line abelian method
    H = EulerHamiltonian(kpoint[0], kpoint[1])
    _, evecsFinal = GetEvalsAndEvecs(H)
    wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsFinal, evecsInitial)

    

#%%
"""
Calculate Wilson Line - Non Abelian
"""

wilsonLineNonAbelian1 = np.zeros([totalPoints, 3, 3], dtype=np.complex128)
wilsonLineNonAbelian2 = np.zeros([totalPoints, 3, 3], dtype=np.complex128)
currentArgument1 = np.zeros([3,3], dtype=np.complex128)
currentArgument2 = np.zeros([3,3], dtype=np.complex128)


for i, kpoint in enumerate(kline):
    berryConnect = CalculateBerryConnectMatrix(kpoint)
    dq1 = dqs1[i]
    dq2 = dqs2[i]
    berryConnectAlongKLine1 =  1j*np.dot(berryConnect, dq1)
    berryConnectAlongKLine2 =  1j*np.dot(berryConnect, dq2) 
    currentArgument1 = currentArgument1 + berryConnectAlongKLine1
    currentArgument2 = currentArgument2 + berryConnectAlongKLine2
    wilsonLineNonAbelian1[i] = expm(currentArgument1)
    wilsonLineNonAbelian2[i] = expm(currentArgument2)
    
    

#%%
'''
Plot
'''

m1 = 2
m2 = 2
multiplier = np.linspace(0,2*pi,totalPoints, endpoint=True)
fig, ax = plt.subplots(figsize=(12,9))

ax.plot(multiplier, np.square(np.abs(wilsonLineNonAbelian1[:,m1,m2])), label="Non Abelian differentiation")
ax.plot(multiplier, np.square(np.abs(wilsonLineNonAbelian2[:,m1,m2])), 'x', label="Non Abelian difference")
ax.plot(multiplier, np.square(np.abs(wilsonLineAbelian[:,m1,m2])), label="Abelian")

ax.set_ylabel(r"$|W["+str(m1) +","+str(m2)+"]|^2 = |<\Phi_{q_f}^"+str(m1)+" | \Phi_{q_i}^"+str(m2)+">|^2$")
ax.set_xticks([0, pi, 2*pi])
ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])


# ax.set_ylim([0,1.01])
# ax.set_xlabel(r"Final quasimomentum point (going around square)")
# plt.savefig(sh+ "WilsonLineEulerRectangle00.pdf", format="pdf")
# plt.savefig(sh+ "WilsonLineEulerCircle22.pdf", format="pdf")
# ax.set_title("2nd Way NOn Abelian")
plt.legend(loc="upper right")
plt.show()    
