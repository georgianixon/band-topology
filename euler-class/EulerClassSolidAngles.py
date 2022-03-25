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
from EulerClass0Hamiltonian import Euler0Hamiltonian

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

def CreateLinearLine(qBegin, qEnd,  qpoints):
    kline = np.linspace(qBegin, qEnd, qpoints)
    return kline





#%% 
"""
Theta/alpha/inner manifold over the BZ
"""

#band that we looking to describe
n1 = 0

Ham = Euler4Hamiltonian

#points in the line
qpoints=51

# arbitrary point I guess, but not a dirac point
gammaPoint = np.array([0,-0.5])

#get evecs at gamma point
H = Ham(gammaPoint)
_, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix = 1) # may as well gauge fix here
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

# S3 = np.zeros((qpoints, qpoints))
# S8 = np.zeros((qpoints, qpoints))

# #calculate internal manifold
# lambda3 = np.diag([0,-1,1])
# lambda8 = (1/np.sqrt(3))*np.diag([-2,1,1])

# thetas1 = np.zeros((qpoints,qpoints))
# alphas1 = np.zeros((qpoints, qpoints))

# thetas2 = np.zeros((qpoints,qpoints))
# alphas2 = np.zeros((qpoints, qpoints))

eiglist = np.empty((qpoints,qpoints,3)) # for three bands

for xi, qx in enumerate(K1):
    for yi, qy in enumerate(K2):
        k = np.array([qx,qy])
        H = Ham(k)
        eigs, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0) # will gauge fix later

        uFinal = evecs[:,n1]
        # uFinal0 = uFinal
    
        #get correct overall phase for uFinal
        uFinal0 = AlignGaugeBetweenVecs(u0, uFinal)
        # uFinal1 = AlignGaugeBetweenVecs(u1, uFinal)
        # uFinal2 = AlignGaugeBetweenVecs(u2, uFinal)
        
        # get params
        argument = np.vdot(u0, uFinal0)
        assert(round(np.imag(argument), 26)==0)
        argument = np.real(argument)
        
        theta0 = np.arccos(argument)
        thetas0[xi,yi] = theta0

        alphaarg = np.vdot(u1, uFinal0)/sin(theta0)
        assert(round(np.imag(alphaarg), 26)==0)
        alphaarg = np.real(alphaarg)
        alpha0 = np.arcsin(alphaarg)
        alphas0[xi,yi] = alpha0

        
        # calculate interior sphere only
        # u0overlap = np.vdot(u0, uFinal)
        # u1overlap = np.vdot(u1, uFinal)
        # u2overlap = np.vdot(u2, uFinal)
        # kvec_u_basis = np.array([u0overlap, u1overlap, u2overlap])
        # internalSphere = np.vdot(kvec_u_basis, np.dot(lambda3, kvec_u_basis))
        # externalSphere = np.vdot(kvec_u_basis, np.dot(lambda8, kvec_u_basis))
        # S3[xi,yi] = internalSphere
        # S8[xi,yi] = externalSphere
        
        # argument = np.dot(np.conj(u0), uFinal1)
        # assert(round(np.imag(argument), 26)==0)
        # argument = np.real(argument)
        # theta1 = np.arccos(argument)
        # thetas1[xi,yi] = theta1
        
        # alphaarg = np.vdot(u1, uFinal1)/sin(theta1)
        # assert(round(np.imag(alphaarg), 26)==0)
        # alphaarg = np.real(alphaarg)
        # alpha1 = np.arcsin(alphaarg)
        # alphas1[xi,yi] = alpha1
        
        # # get params
        # argument = np.dot(np.conj(u0), uFinal2)
        # assert(round(np.imag(argument), 26)==0)
        # argument = np.real(argument)
        # theta2 = np.arccos(argument)
        # thetas2[xi,yi] = theta2
        
        # alphaarg = np.vdot(u1, uFinal2)/sin(theta2)
        # assert(round(np.imag(alphaarg), 26)==0)
        # alphaarg = np.real(alphaarg)
        # alpha2 = np.arcsin(alphaarg)
        # alphas1[xi,yi] = alpha2

#%%
"""
Plot theta over BZ
"""

params = {
          'axes.grid':False,
          }

mpl.rcParams.update(params)
cmap = "RdYlGn"#"plasma"#"RdYlGn"#"plasma"

# turn x -> along bottom, y |^ along LHS
thetas0Plot =  np.flip(thetas0.T, axis=0)
alphas0Plot =  np.flip(alphas0.T, axis=0)
# S30Plot = np.flip(S3.T, axis=0)
# S80Plot = np.flip(S8.T, axis=0)
# thetas1Plot =  np.flip(thetas1.T, axis=0)
# alphas1Plot =  np.flip(alphas1.T, axis=0)
# thetas2Plot =  np.flip(thetas2.T, axis=0)
# alphas2Plot =  np.flip(alphas2.T, axis=0)

xx = "0"
yy = "-0p5"
plotnames = [
            # "ThetaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
             # "AlphaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
             "ThetaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
              "AlphaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
              # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
              # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
               # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u1.pdf",
               # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u1.pdf",
               # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u2.pdf",
               # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u2.pdf",

             # "S8-Refk=(-0p5,-0p5).pdf"
             ]

plotvars = [
            thetas0Plot, alphas0Plot, 
            # thetas1Plot, alphas1Plot,
            # thetas2Plot, alphas2Plot,
            # S30Plot,
            # S80Plot
            ]
for plotvar, savename in zip(plotvars, plotnames):
    # to ensure zero is in the middle, optional
    pmin = np.nanmin(plotvar)
    pmax = np.nanmax(plotvar)
    bignum = np.max([np.abs(pmin), np.abs(pmax)])
    if bignum < pi/2:
        bignum = pi/2
    normaliser = mpl.colors.Normalize(vmin=-bignum, vmax=bignum)
    
    # plot 
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap=cmap, norm=normaliser)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

    fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    plt.savefig(sh+savename, format="pdf", bbox_inches="tight")
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

"""
Theta over a line - 4 BZ
"""

CB91_Blue = 'darkblue'#'#2CBDFE'
oxfordblue = "#061A40"
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"
newred = "#E4265C"
flame = "#DD6031"

color_list = [CB91_Blue, flame, CB91_Pink, CB91_Green, CB91_Amber,
                CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey', newred]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
def q0ToString(q0):
    a = q0[0]
    b = q0[1]
    if type(a) != int and type(a) != np.int32:
        
        if a.is_integer():
            a = int(a)
    if type(b) != int and type(b) != np.int32:
        if b.is_integer():
            b = int(b)
        
    return "("+str(a).replace(".", "p")+","+str(b).replace(".", "p")+")"

#band that we looking to describe
n1 = 0



# fs = (12,9)
# fig, ax = plt.subplots(figsize=fs)
# multiplier = np.linspace(0, 4, qpoints)
# qpoints = 10000
# for pp in np.linspace(-1,1,21):
#     q0 = np.array([pp,0])
#     qf = q0+ np.array([0,4])
#     kline = np.linspace(q0, qf, qpoints)
#     x,y = zip(*kline)
#     ax.plot(x, y, color ='darkblue', label=r"k line")
#     ax.plot(kline[0][0], kline[0][1], 'x', color = "#DD6031", markersize=20, label=r"$\Gamma=("+str(q0[0])+r","+str(q0[0])+r")$")
# ax.set_xlabel(r"$q_x$")
# ax.set_ylabel(r"$q_y$", rotation=0, labelpad=15)
# ax.set_facecolor('1')
# ax.grid(b=1, color='0.6')
# ax.set_xticks(np.linspace(-1,1,11))
# # ax.legend()
# plt.savefig(sh+"LineTrajes,Euler=4,VecF=(0,4).png", format="png", bbox_inches="tight")
# plt.show()


for pp in np.linspace(-1,1,21):
    pp = round(pp, 2)
    
    Ham = Euler4Hamiltonian
    
    #define path
    #num of points
    qpoints = 10000
    q0 = np.array([pp,0])
    qf = q0+ np.array([0,4])
    kline = np.linspace(q0, qf, qpoints)
    
    
    #get evecs at gamma point
    H = Ham(q0)
    _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix = 1) # may as well gauge fix here
    u0 = evecs[:,0]
    u1 = evecs[:,1]
    u2 = evecs[:,2]
    #check it is not a dirac point
    assert(np.round(np.linalg.norm(evecs[:,n1]),10)==1)
    
    
    thetasLine = np.zeros(qpoints)
    alphasLine = np.zeros(qpoints)
    
    # go through possible end points for k, get angles
    for i, kpoint in enumerate(kline):
        #do abeliean version,
        #find evecs at other k down the line
        H = Ham(kpoint)
        _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0)
        uFinal = evecs[:,n1]
        
        #get correct overall phase for uFinal
        uFinal = AlignGaugeBetweenVecs(u0, uFinal)
    
        # get params
        
        # get params
        argument = np.vdot(u0, uFinal)
        assert(round(np.imag(argument), 26)==0)
        argument = np.real(argument)
        
        theta0 = np.arccos(argument)
        # thetaPrime = 
        
        alphaarg = np.vdot(u1, uFinal)/sin(theta0)
        assert(round(np.imag(alphaarg), 26)==0)
        alphaarg = np.real(alphaarg)
        alpha = np.arcsin(alphaarg)
    
        
        thetasLine[i] = theta0
        alphasLine[i] = alpha
    
    
    
    #differentiate thetas
    # dCircAngle = 2*pi/(qpoints-1)
    # dThetadAngle = np.empty(qpoints-1)
    # dAlphadAngle = np.empty(qpoints-1)
    # for i in range(qpoints-1):
    #     dThetadAngle[i] = (thetasLine[i+1] - thetasLine[i])/dCircAngle
    #     dAlphadAngle[i] = (alphasLine[i+1] - alphasLine[i])/dCircAngle
    
    
    # #plot kline
    
    
    print(q0ToString(q0))
    

    saveLine = "LineTraj,Euler=4,GroundState,GaugeFixToGamma="+q0ToString(q0)+",VecF=(0,4).png"
    # saveLineDifferentiate = "DifferentiatedOverCircle,GroundState,GaugeFixToGamma=(0p6,0),CircleTraj,Centre=0,R=0p6.pdf"
    saveTheta = "Theta"+saveLine
    saveAlpha = "Alpha"+saveLine
    
    # saveThetaDifferentiated = "Theta"+saveLineDifferentiate
    # saveAlphaDifferentiated = "Alpha"+saveLineDifferentiate
    
    
    

    
    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, thetasLine, '.', markersize=3, label=r"$\theta$")
    ax.set_yticks([0, pi/2, pi])
    ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
    ax.set_xlabel(r"$q_x$")
    ax.grid(b=1, color='1')
    plt.savefig(sh+saveTheta, format="png", bbox_inches="tight")
    plt.show()    
    
    
    
    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, alphasLine, '.', markersize=3, label=r"$\alpha$")
    # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
    ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
    ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
    ax.set_xlabel(r"$q_x$")
    ax.grid(b=1, color='1')
    plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()    



# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier[:-1], dThetadAngle, '.', markersize=3)
# # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
# ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
# ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
# # ax.set_yticks([0, pi/2, pi])
# # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
# ax.set_ylabel(r"$\frac{\partial \theta}{\partial \phi}$", rotation=0, labelpad=15)
# ax.set_xlabel(r"$\phi$")
# # plt.savefig(sh+saveThetaDifferentiated, format="pdf", bbox_inches="tight")
# plt.show()  


# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier[:-1], dAlphadAngle, '.', markersize=3)
# # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
# ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
# ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
# ax.set_ylim([-2.5,2.5])
# # ax.set_yticks([0, pi/2, pi])
# # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
# ax.set_ylabel(r"$\frac{\partial \alpha}{\partial \phi}$", rotation=0, labelpad=15)
# ax.set_xlabel(r"$\phi$")
# plt.savefig(sh+saveAlphaDifferentiated, format="pdf", bbox_inches="tight")
# plt.show()    

#%% 
"""
Theta over a line - circle
"""

#band that we looking to describe
n1 = 0



Ham = Euler2Hamiltonian
#define path
#num of points
qpoints = 10000
radius=0.6
centre = [0,0]

# create circle line
kline = CreateCircleLine(radius, qpoints, centre = centre)
dqs = CircleDiff(qpoints, radius)
totalPoints = len(kline)




# arbitrary point I guess, but not a dirac point
gammaPoint = kline[0]

#get evecs at gamma point
H = Ham(gammaPoint)
_, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix = 1) # may as well gauge fix here
u0 = evecs[:,0]
u1 = evecs[:,1]
u2 = evecs[:,2]
#check it is not a dirac point
assert(np.round(np.linalg.norm(evecs[:,n1]),10)==1)


thetasLine = np.zeros(totalPoints, dtype=np.complex128)
alphasLine = np.zeros(totalPoints, dtype=np.complex128)

# go through possible end points for k, get andlges
for i, kpoint in enumerate(kline):
    #do abeliean version,
    #find evecs at other k down the line
    H = Ham(kpoint)
    _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0)
    uFinal = evecs[:,n1]
    
    #get correct overall phase for uFinal
    uFinal = AlignGaugeBetweenVecs(u0, uFinal)

    # get params
    
    # get params
    argument = np.vdot(u0, uFinal)
    assert(round(np.imag(argument), 26)==0)
    argument = np.real(argument)
    theta0 = np.arccos(argument)
    
    alphaarg = np.vdot(u1, uFinal)/sin(theta0)
    assert(round(np.imag(alphaarg), 26)==0)
    alphaarg = np.real(alphaarg)
    alpha = np.arcsin(alphaarg)

    
    thetasLine[i] = theta0
    alphasLine[i] = alpha



#differentiate thetas
# dCircAngle = 2*pi/(qpoints-1)
# dThetadAngle = np.empty(qpoints-1)
# dAlphadAngle = np.empty(qpoints-1)
# for i in range(qpoints-1):
#     dThetadAngle[i] = (thetasLine[i+1] - thetasLine[i])/dCircAngle
#     dAlphadAngle[i] = (alphasLine[i+1] - alphasLine[i])/dCircAngle

#%%
# #plot kline


CB91_Blue = 'darkblue'#'#2CBDFE'
oxfordblue = "#061A40"
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"
newred = "#E4265C"
flame = "#DD6031"

color_list = [CB91_Blue, flame, CB91_Pink, CB91_Green, CB91_Amber,
                CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey', newred]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
saveLine = "OverCircle,Euler=0,GroundState,GaugeFixToGamma=(0p6,0),CircleTraj,Centre=0,R=0p6.pdf"
# saveLineDifferentiate = "DifferentiatedOverCircle,GroundState,GaugeFixToGamma=(0p6,0),CircleTraj,Centre=0,R=0p6.pdf"
saveTheta = "Theta"+saveLine
saveAlpha = "Alpha"+saveLine

# saveThetaDifferentiated = "Theta"+saveLineDifferentiate
# saveAlphaDifferentiated = "Alpha"+saveLineDifferentiate



multiplier = np.linspace(0, 2*pi, totalPoints)
fs = (12,9)
x,y = zip(*kline)
fig, ax = plt.subplots(figsize=fs)
ax.plot(x, y, label=r"k line")
ax.plot(kline[0][0], kline[0][1], 'x', markersize=20, label=r"$\Gamma$")
ax.set_xlabel(r"$q_x$")
ax.set_ylabel(r"$q_y$", rotation=0, labelpad=15)
ax.set_facecolor('1')
ax.grid(b=1, color='0.6')
ax.legend()
plt.savefig(sh+saveLine, format="pdf", bbox_inches="tight")
plt.show()


fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(thetasLine), '.', markersize=3, label=r"$\theta$")
# ax.set_xlabel(r"$\theta$ over circle trajectory, centre=(0,0), radius=0.6, gauge fix to $\Theta=(0.6,0)$, ground band")
ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
ax.set_yticks([0, pi/2, pi])
ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
ax.set_xlabel(r"$\phi$")
plt.savefig(sh+saveTheta, format="pdf", bbox_inches="tight")
plt.show()    

fig, ax = plt.subplots(figsize=fs)
ax.plot(multiplier, np.real(alphasLine), '.', markersize=3, label=r"$\alpha$")
# ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
ax.set_xlabel(r"$\phi$")
plt.savefig(sh+saveAlpha, format="pdf", bbox_inches="tight")
plt.show()    




# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier[:-1], dThetadAngle, '.', markersize=3)
# # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
# ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
# ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
# # ax.set_yticks([0, pi/2, pi])
# # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
# ax.set_ylabel(r"$\frac{\partial \theta}{\partial \phi}$", rotation=0, labelpad=15)
# ax.set_xlabel(r"$\phi$")
# # plt.savefig(sh+saveThetaDifferentiated, format="pdf", bbox_inches="tight")
# plt.show()  


# fig, ax = plt.subplots(figsize=fs)
# ax.plot(multiplier[:-1], dAlphadAngle, '.', markersize=3)
# # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
# ax.set_xticks([0, pi/2, pi, 3*pi/2, 2*pi])
# ax.set_xticklabels(['0', "", r"$\pi$", "",  r"$2\pi$"])
# ax.set_ylim([-2.5,2.5])
# # ax.set_yticks([0, pi/2, pi])
# # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$", r"$\pi$"])
# ax.set_ylabel(r"$\frac{\partial \alpha}{\partial \phi}$", rotation=0, labelpad=15)
# ax.set_xlabel(r"$\phi$")
# plt.savefig(sh+saveAlphaDifferentiated, format="pdf", bbox_inches="tight")
# plt.show()    
  


#%%



import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True
fig = plt.figure(figsize=(7,3.5))
ax = fig.add_subplot(111, projection='3d')
m = 1
N = 3
data = np.empty((100,100,3))
for i, kx in enumerate(np.linspace(-1, 1, 100)):
    for j, ky in enumerate(np.linspace(-1,1, 100)):
        data[i, j, 0] = m - cos(kx) - cos(ky)
        data[i, j, 1] = sin(kx)
        data[i, j, 2] = sin(ky)
data = data/N
z, x, y = data.nonzero()

ax.scatter(x, y, z, c=z, alpha=1)
plt.show()


    



    

