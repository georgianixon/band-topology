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
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from EulerClassHamiltonian import  EulerHamiltonian
from hamiltonians import GetEvalsAndEvecs
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

def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  np.array([[cos(x)*r+centre[0],sin(x)*r+centre[1]] for x in np.linspace(0, 2*pi, points, endpoint=True)])
    return CircleLine

def CircleDiff(points, radius):
    """
    Gives vectors tangent to the circle for various theta between 0 and 2*pi
    Vector size = distance between points on the circle
    """
    circleDiffNormalised = np.array([[-sin(theta),cos(theta)] for theta in np.linspace(0, 2*pi, points, endpoint=True)])
    dtheta = 2*pi/(points-1)
    dqlength = 2*radius*sin(dtheta)
    circleDiff = circleDiffNormalised*dqlength
    return circleDiff

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints)
    return kline


def AlignGaugeBetweenVecs(vec1, vec2):
    """
    Make <vec1|vec2> real and positive by shifting overall phase of vec2
    Return phase shifted vec2
    """
    #overlap between vec1 and vec2
    c = np.vdot(vec1, vec2)
    #find conj phase of overlap
    conjPhase = np.conj(c)/np.abs(c)
    #remove phase, so overlap is real and positive
    vec2 = conjPhase*vec2
    
    # make sure vec1 is in the right gauge, to 20dp
    c = np.dot(np.conj(vec1), vec2)
    
    #try again if still not within..
    if round(np.imag(c), 30)!=0:
        conjPhase = np.conj(c)/np.abs(c)
        vec2 = conjPhase*vec2
        c = np.dot(np.conj(vec1), vec2)
        assert(round(np.imag(c), 26)==0)
    
    return vec2

#%%
#band that we looking at eigenstate trajectory
n1 = 2

"""
Define path (kline)
"""

#num of points to calculate the wilson Line of
qpoints = 1000
radius=0.5
centre = [1,1]

# create circle line
kline = CreateCircleLine(radius, qpoints, centre = centre)
dqs = CircleDiff(qpoints, radius)
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
    uFinal = evecs[:,n1]
    
    #get correct overall phase for uFinal
    uFinal = AlignGaugeBetweenVecs(u2, uFinal)

    # get params
    theta = 2*np.arcsin(np.dot(np.conj(u2), uFinal))
    #sometimes you will get nans for these, if theta = pi
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
multiplier = np.linspace(0, 2*pi, totalPoints)
fs = (12,9)
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
ax.set_xlabel(r"final quasimomentum, going around circle with centre (1,1), $2^{\mathrm{nd}}$ excited band")
ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
ax.set_yticks([0, pi/2, pi])
ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
plt.savefig(sh+ "thetasCircle2Trajectory.pdf", format="pdf")
plt.show()    

# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier, np.real(alphasLine), label=r"$\alpha$")
# ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), 2^{\mathrm{nd}}$ excited band")
# ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
# ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
# ax.set_yticks([0, pi/2, pi])
# ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
# ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
# plt.savefig(sh+ "alphasCircle1Trajectory.pdf", format="pdf")
# plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(phisLine), label=r"$\phi$")
ax.set_xlabel(r"final quasimomentum, going around circle with centre (1,1), $2^{\mathrm{nd}}$ excited band")
# plt.legend()
ax.set_xticks([0, pi, 2*pi])
ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
ax.set_ylabel(r"$\phi$")
plt.savefig(sh+ "phisCircle2Trajectory.pdf", format="pdf")
plt.show()    


# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier, np.real(psisLine), label=r"$\psi$")
# ax.set_xlabel(r"Final quasimomentum, square trajectory, $2^{\mathrm{nd}}$ excited band")
# # plt.legend()
# # ax.set_xticks([0, pi, 2*pi])
# # ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
# ax.set_ylabel(r"$\psi$")
# # plt.savefig(sh+ "psisSquareTrajectory2ndExcitedBand.pdf", format="pdf")
# plt.show()    


#%% 
"""
Theta over the BZ
"""
#band that we looking to describe
n1 = 2

#points in the line
qpoints=51

# arbitrary point I guess, but not a diract point
gammaPoint = np.array([0,-0.5])

#get evecs at gamma point
H = EulerHamiltonian(gammaPoint[0],gammaPoint[1])
_, evecs = GetEvalsAndEvecs(H)
u0 = evecs[:,0]
u1 = evecs[:,1]
u2 = evecs[:,2]
#check it is not a dirac point
assert(np.round(np.linalg.norm(u2),10)==1)

#Go through all other points in BZ;
kmin = -1
kmax = 1
qpoints = 201 # easier for meshgrid when this is odd
K1 = np.linspace(kmin, kmax, qpoints, endpoint=True)
K2 = np.linspace(kmin, kmax, qpoints, endpoint=True)
thetas = np.zeros((qpoints,qpoints))

eiglist = np.empty((qpoints,qpoints,3)) # for three bands

for xi, qx in enumerate(K1):
    for yi, qy in enumerate(K2):
        eigs, evecs = GetEvalsAndEvecs(EulerHamiltonian(qx,qy))
        
        uFinal = evecs[:,n1]
    
        #get correct overall phase for uFinal
        uFinal = AlignGaugeBetweenVecs(u2, uFinal)
    
        # get params
        argument = np.dot(np.conj(u2), uFinal)
        assert(round(np.imag(argument), 26)==0)
        argument = np.real(argument)
        theta = 2*np.arcsin(argument)
        assert(round(np.imag(theta), 26)==0)
        theta = np.real(theta)
        
        thetas[xi,yi] = theta
        
#%%
"""
Plot theta over BZ
"""

params = {
          'axes.grid':False,
          }

mpl.rcParams.update(params)

# turn x -> along bottom, y |^ along LHS
thetas =  np.flip(thetas.T, axis=0)

# plot 
sz = 15
fig, ax = plt.subplots(figsize=(sz/2,sz/2))
pos = plt.imshow(thetas, cmap='plasma')
ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
ax.set_yticklabels([kmax, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmin])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
fig.colorbar(pos)
plt.savefig(sh+"ThetaOverBZSecondBand,Gamma=(0,-0p5).pdf", format="pdf")
plt.show()