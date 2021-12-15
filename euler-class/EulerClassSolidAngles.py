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
from EulerClass2Hamiltonian import  Euler2Hamiltonian, GetEvalsAndEvecsEuler, AlignGaugeBetweenVecs
from EulerClass4Hamiltonian import Euler4Hamiltonian

# from hamiltonians import GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm
import matplotlib as mpl

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/Euler Class/"


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





#%% 
"""
Theta over the BZ
"""
#band that we looking to describe
n1 = 0

#points in the line
qpoints=51

# arbitrary point I guess, but not a dirac point
gammaPoint = np.array([-0.501,-0.5])

#get evecs at gamma point
H = Euler2Hamiltonian(gammaPoint)
_, evecs = GetEvalsAndEvecsEuler(H)
u0 = evecs[:,0]
u1 = evecs[:,1]
u2 = evecs[:,2]
#check it is not a dirac point

assert(np.round(np.linalg.norm(evecs[:,n1]),10)==1)

#Go through all other points in BZ;
kmin = -1
kmax = 1
qpoints = 201 # easier for meshgrid when this is odd
K1 = np.linspace(kmin, kmax, qpoints, endpoint=True)
K2 = np.linspace(kmin, kmax, qpoints, endpoint=True)

thetas0 = np.zeros((qpoints,qpoints))
alphas0 = np.zeros((qpoints, qpoints))

# thetas1 = np.zeros((qpoints,qpoints))
# alphas1 = np.zeros((qpoints, qpoints))

eiglist = np.empty((qpoints,qpoints,3)) # for three bands

lambda3 = np.diag([1,-1,0])
nurThings = np.zeros((qpoints, qpoints))
for xi, qx in enumerate(K1):
    for yi, qy in enumerate(K2):
        k = np.array([qx,qy])
        H = Euler2Hamiltonian(k)
        eigs, evecs = GetEvalsAndEvecsEuler(H)
        
        uFinal = evecs[:,n1]
    
        #get correct overall phase for uFinal
        uFinal = AlignGaugeBetweenVecs(u0, uFinal)
        # uFinal1 = AlignGaugeBetweenVecs(u1, uFinal)
        
        # get params
        argument = np.dot(np.conj(u0), uFinal)
        assert(round(np.imag(argument), 26)==0)
        argument = np.real(argument)
        theta0 = 2*np.arcsin(argument)
        thetas0[xi,yi] = theta0
        
        alphaarg = np.vdot(u1, uFinal)/cos(theta0/2)
        assert(round(np.imag(alphaarg), 26)==0)
        alphaarg = np.real(alphaarg)
        alpha = 2*np.arcsin(alphaarg)
        alphas0[xi,yi] = alpha
        
        """random lambda 3 thing Nur asked for"""
        u0overlap = np.vdot(u0, uFinal)
        u1overlap = np.vdot(u1, uFinal)
        u2overlap = np.vdot(u2, uFinal)
        kvec_u_basis = np.array([u0overlap, u1overlap, u2overlap])
        nurThing = np.vdot(kvec_u_basis, np.dot(lambda3, kvec_u_basis))
        nurThings[xi,yi] = nurThing
        
        # get params
        # argument = np.dot(np.conj(u0), uFinal1)
        # assert(round(np.imag(argument), 26)==0)
        # argument = np.real(argument)
        # theta1 = 2*np.arcsin(argument)
        # thetas1[xi,yi] = theta1
        
        # alphaarg = np.vdot(u1, uFinal1)/cos(theta1/2)
        # assert(round(np.imag(alphaarg), 26)==0)
        # alphaarg = np.real(alphaarg)
        # alpha = 2*np.arcsin(alphaarg)
        # alphas1[xi,yi] = alpha
        
#%%
"""
Plot theta over BZ
"""

params = {
          'axes.grid':False,
          }

mpl.rcParams.update(params)

# turn x -> along bottom, y |^ along LHS
thetas0Plot =  np.flip(thetas0.T, axis=0)
alphas0Plot =  np.flip(alphas0.T, axis=0)
nurThings = np.flip(nurThings.T, axis=0)
# thetas1Plot =  np.flip(thetas1.T, axis=0)
# alphas1Plot =  np.flip(alphas1.T, axis=0)

xx = "-0p501"
yy = "-0p5"
plotnames = [
            # "ThetaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
             # "AlphaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
             # "ThetaOverBZ-Euler2-,Gamma=(0p01,0),FixGaugeTo-u1.pdf",
             # "AlphaOverBZ-Euler2-,Gamma=(0p01,0),FixGaugeTo-u1.pdf",
             "nurThings.pdf"
             ]

plotvars = [
            # thetas0Plot, alphas0Plot, 
            # thetas1Plot, alphas1Plot
            nurThings
            ]
for plotvar, savename in zip(plotvars, plotnames):
    # plot 
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap='plasma')
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
    fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    # plt.savefig(sh+savename, format="pdf", bbox_inches="tight")
    plt.show()


# # turn x -> along bottom, y |^ along LHS
# alphasPlot =  np.flip(alphas.T, axis=0)

# # plot 
# sz = 15
# fig, ax = plt.subplots(figsize=(sz/2,sz/2))
# pos = plt.imshow(alphasPlot, cmap='plasma')
# ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
# ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
# ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
# ax.set_yticklabels([kmax, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmin])
# ax.set_xlabel(r"$k_x$")
# ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
# fig.colorbar(pos)
# # plt.savefig(sh+"ThetaOverBZ-Euler4-,Gamma=(0p5,0).pdf", format="pdf", bbox_inches="tight")
# plt.show()




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
H = EulerHamiltonian(k0)
_, evecs = GetEvalsAndEvecsEuler(H)
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
    H = EulerHamiltonian(kpoint)
    _, evecs = GetEvalsAndEvecsEuler(H)
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
# plt.savefig(sh+ "thetasCircle2Trajectory.pdf", format="pdf")
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
# plt.savefig(sh+ "phisCircle2Trajectory.pdf", format="pdf")
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