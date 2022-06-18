# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:03:44 2021

@author: Georgia Nixon
"""

place = "Georgia"
import numpy as np
from numpy import cos, sin, exp, pi, tan
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from EulerClass2Hamiltonian import  Euler2Hamiltonian, GetEvalsAndEvecsEuler
from EulerClass4Hamiltonian import Euler4Hamiltonian
from EulerClass0Hamiltonian import Euler0Hamiltonian

# from hamiltonians import GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm
import matplotlib as mpl
from Funcs import VecToStringSave, ProjReal, InverseSin, InverseCos, FindOverlap, VecToString, FivePointLine, AlignGaugeBetweenVecs

def FirstElementVecPositive(vec):
    if vec[0] >= 0:
        return vec
    else:
        return -vec
    
sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/Euler Class/"


size=25
params = {
        'legend.fontsize': size*0.9,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
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


for value in [ 0]:

    gammaPoint = np.array([value,0])
    
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
    
    thetasG1 = np.zeros((qpoints,qpoints))
    thetasG2 = np.zeros((qpoints,qpoints))
    
    
    alphasG1 = np.zeros((qpoints, qpoints))
    alphasG2 = np.zeros((qpoints, qpoints))
    
    
    # alphasG11 = np.zeros((qpoints, qpoints))
    # alphasG12 = np.zeros((qpoints, qpoints))
    # alphasG13 = np.zeros((qpoints, qpoints))
    # alphasG14 = np.zeros((qpoints, qpoints))
    # alphasG21 = np.zeros((qpoints, qpoints))
    # alphasG22 = np.zeros((qpoints, qpoints))
    # alphasG23 = np.zeros((qpoints, qpoints))
    # alphasG24 = np.zeros((qpoints, qpoints))
    
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
            _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0) # will gauge fix later
    
            uFinal = evecs[:,n1]
            # uFinal0 = uFinal
        
            #get correct overall phase for uFinal
            uFinalG1 = AlignGaugeBetweenVecs(u0, uFinal)
            
            uFinalG2 = -uFinalG1
            
            # uFinal1 = AlignGaugeBetweenVecs(u1, uFinal)
            # uFinal2 = AlignGaugeBetweenVecs(u2, uFinal)
            
            # # get params
            # argument = np.vdot(u0, uFinal0)
            # assert(round(np.imag(argument), 26)==0)
            # argument = np.real(argument)
            
            # theta0 = np.arccos(argument)
            # thetas0[xi,yi] = theta0
    
            # alphaarg = np.vdot(u1, uFinal0)/sin(theta0)
            # assert(round(np.imag(alphaarg), 26)==0)
            # alphaarg = np.real(alphaarg)
            # alpha0 = np.arcsin(alphaarg)
            # alphas0[xi,yi] = alpha0
    
    
        
            
            # get params
            argumentG1 = ProjReal(np.vdot(u0, uFinalG1))
            #only want the values of theta between 0 and pi because we on bloch sphere
            # therefore we neglect second arccos value
            thetaG1 = np.arccos(argumentG1)
            # thetaOtherGauge = pi-theta0
            argumentG2 = ProjReal(np.vdot(u0, uFinalG2))
            thetaG2 = np.arccos(argumentG2)
            
            
            
            alphaargG1 = ProjReal(np.vdot(u1, uFinalG1)/sin(thetaG1))
            alphaG1_1, alphaG1_2 = InverseSin(alphaargG1)
            # alphaOtherGauge = alpha - pi
            
            alphaargG2 = ProjReal(np.vdot(u1, uFinalG2)/sin(thetaG2))
            alphaG2_1, alphaG2_2 = InverseSin(alphaargG2)
        
        
            alphaCosArgG1 = ProjReal(np.vdot(u2, uFinalG1)/sin(thetaG1))
            alphaCosG1_1, alphaCosG1_2 = InverseCos(alphaCosArgG1)
            
            alphaCosArgG2 = ProjReal(np.vdot(u2, uFinalG2)/sin(thetaG2))
            alphaCosG2_1, alphaCosG2_2 = InverseCos(alphaCosArgG2)
                                   
                                                  
            thetasG1[xi, yi] = thetaG1
            thetasG2[xi, yi] = thetaG2
            
            # alphasG11[xi, yi] = alphaG1_1
            # alphasG12[xi, yi] = alphaG1_2
            # alphasG13[xi, yi] = alphaCosG1_1
            # alphasG14[xi, yi] = alphaCosG1_2
            
            # alphasG21[xi, yi] = alphaG2_1
            # alphasG22[xi, yi] = alphaG2_2
            # alphasG23[xi, yi] = alphaCosG2_1
            # alphasG24[xi, yi] = alphaCosG2_2
            
            alphasG1[xi, yi] = FindOverlap(alphaG1_1, alphaG1_2, alphaCosG1_1, alphaCosG1_2)
            alphasG2[xi, yi] = FindOverlap(alphaG2_1, alphaG2_2, alphaCosG2_1, alphaCosG2_2)
            
            
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
    
    """
    Plot theta over BZ
    """
    
    params = {
              'axes.grid':False,
              }
    
    mpl.rcParams.update(params)
    cmap = "RdYlGn"#"plasma"#"RdYlGn"#"plasma"
    
    # turn x -> along bottom, y |^ along LHS
    thetasG1Plot =  np.flip(thetasG1.T, axis=0)
    thetasG2Plot =  np.flip(thetasG2.T, axis=0)
    alphasG1Plot =  np.flip(alphasG1.T, axis=0)
    alphasG2Plot =  np.flip(alphasG2.T, axis=0)
    # S30Plot = np.flip(S3.T, axis=0)
    # S80Plot = np.flip(S8.T, axis=0)
    # thetas1Plot =  np.flip(thetas1.T, axis=0)
    # alphas1Plot =  np.flip(alphas1.T, axis=0)
    # thetas2Plot =  np.flip(thetas2.T, axis=0)
    # alphas2Plot =  np.flip(alphas2.T, axis=0)
    
    # xx = VecToStringSave(gammaPoint)
    # yy = VecToStringSave(gammaPoint[1])
    
    # xx = "0p5"
    # yy = "0"
    
    #plot theta
    # to ensure zero is in the middle, optional
    plotvar = thetasG1Plot
    # pmin = np.nanmin(plotvar)
    # pmax = np.nanmax(plotvar)
    # bignum = np.max([np.abs(pmin), np.abs(pmax)])
    # if bignum < pi/2:
    #     bignum = pi/2
    normaliser = mpl.colors.Normalize(vmin=0, vmax=pi)
    
    # plot 
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap="RdYlGn", norm=normaliser)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    cbar.set_ticks([0, pi/2, pi])
    cbar.set_ticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
    plt.savefig(sh+"ThetaOverBZ,G1,Euler4,Gamma="+VecToStringSave(gammaPoint)+".png", format="png", bbox_inches="tight")
    plt.show()
    
    
    # plot 
    plotvar = thetasG2Plot
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap="RdYlGn", norm=normaliser)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    cbar.set_ticks([0, pi/2, pi])
    cbar.set_ticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$"])
    # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    plt.savefig(sh+"ThetaOverBZ,G2,Euler4,Gamma="+VecToStringSave(gammaPoint)+".png", format="png", bbox_inches="tight")
    plt.show()
    
    
    plotvar = alphasG1Plot
    normaliser = mpl.colors.Normalize(vmin=0, vmax=2*pi)
    
    # plot 
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap="twilight_shifted_r", norm=normaliser)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    cbar.set_ticks([0, pi/2, pi, 3*pi/2, 2*pi])
    cbar.set_ticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$", r"$2 \pi$"])
    plt.savefig(sh+"AlphaOverBZ,G1,Euler4,Gamma="+VecToStringSave(gammaPoint)+".png", format="png", bbox_inches="tight")
    plt.show()
    
    
    plotvar = alphasG2Plot
    normaliser = mpl.colors.Normalize(vmin=0, vmax=2*pi)
    
    # plot 
    sz = 15
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(plotvar, cmap="twilight_shifted_r", norm=normaliser)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    cbar.set_ticks([0, pi/2, pi, 3*pi/2, 2*pi])
    cbar.set_ticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$", r"$2 \pi$"])
    # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    plt.savefig(sh+"AlphaOverBZ,G2,Euler4,Gamma="+VecToStringSave(gammaPoint)+".png", format="png", bbox_inches="tight")
    plt.show()

    
    # plotnames = [
    #             # "ThetaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
    #               # "AlphaOverBZ-Euler4-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
    #               "ThetaOverBZ,Euler0,Gamma="+VecToStringSave(gammaPoint)+",FixGaugeTo-u0.png",
    #               "AlphaOverBZ,Euler0,Gamma="+VecToStringSave(gammaPoint)+",FixGaugeTo-u0.png",
    #               # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
    #               # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u0.pdf",
    #                 # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u1.pdf",
    #                 # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u1.pdf",
    #                 # "CompareGauge-HalfThetaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u2.pdf",
    #                 # "CompareGauge-HalfAlphaOverBZ-Euler2-,Gamma=("+xx+","+yy+"),FixGaugeTo-u2.pdf",
    
    #               # "S8-Refk=(-0p5,-0p5).pdf"
    #               ]
    
    # cmaps = ["RdYlGn", "twilight_shifted_r"]
    # plotvars = [
    #             thetasG1Plot, alphasG1Plot, 
    #             # thetas1Plot, alphas1Plot,
    #             # thetas2Plot, alphas2Plot,
    #             # S30Plot,
    #             # S80Plot
    #             ]
    # for plotvar, savename,cmap in zip(plotvars, plotnames, cmaps):
    #     # to ensure zero is in the middle, optional
    #     pmin = np.nanmin(plotvar)
    #     pmax = np.nanmax(plotvar)
    #     bignum = np.max([np.abs(pmin), np.abs(pmax)])
    #     if bignum < pi/2:
    #         bignum = pi/2
    #     normaliser = mpl.colors.Normalize(vmin=-bignum, vmax=bignum)
        
    #     # plot 
    #     sz = 15
    #     fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    #     pos = plt.imshow(plotvar, cmap=cmap, norm=normaliser)
    #     ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    #     ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    #     ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    #     ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    #     ax.set_xlabel(r"$k_x$")
    #     ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
    
    #     fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    #     # fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
    #     # plt.savefig(sh+savename, format="png", bbox_inches="tight")
    #     plt.show()



#%%

"""
Theta over a line - 4 BZ
"""

G1colour = 'darkblue'
G2colour = "#DD6031"
#band that we looking to describe
n1 = 0


col1 = "darkblue"
col2 = "#DD6031"
col4 = "#E980FC"
col3 = "#9DD1F1"
col5 = "#F0C808"


    
# def GetInverseCosOutsideRange(alpha):
    

# fs = (15,9)
# fs = (12, 9)
# fig, ax = plt.subplots(figsize=fs)
# multiplier = np.linspace(0, 4, qpoints)
# qpoints = 10000
# for pp in [-0.99, -0.5, -0.1, 0, 0.1, 0.5, 0.99]:
#     q0 = np.array([pp,0])
    
#     # v1 = np.array([0,2])
#     # v2 = np.array([2,0])
#     # q1 = q0+v1
#     # q2 = q0+v1+v2
#     # q3 = q0+v2
#     # q4 = q0
    
#     qf = q0+ np.array([0,8])
#     kline = np.linspace(q0, qf, qpoints)
#     # kline = SquareLine(q0, v1, v2, int((qpoints+3)/4))
    
#     # kline = FivePointLine(q0, q1, q2, q3, q4, int((qpoints+3)/4))
    
    
#     # qf = q0+ np.array([0,8])
#     # kline = np.linspace(q0, qf, qpoints)
#     x,y = zip(*kline)
#     ax.plot(x, y, color ='darkblue', label=r"k line")
#     ax.plot(kline[0][0], kline[0][1], 'x', color = "#DD6031", markersize=20, label=r"$\Gamma=("+str(q0[0])+r","+str(q0[0])+r")$")
# ax.set_xlabel(r"$q_x$")
# ax.set_ylabel(r"$q_y$", rotation=0, labelpad=15)
# ax.set_facecolor('1')
# ax.grid(b=1, color='0.6')
# # ax.set_xticks(np.linspace(-1,3,9))
# ax.set_xticks(np.linspace(-1,1,11))
# # ax.legend()
# plt.savefig(sh+"VerticalLineTrajes,Euler=2,V1=(0,8).png", format="png", bbox_inches="tight")
# plt.show()

Ham = Euler4Hamiltonian
    


    
for pp in [0, 0.1, 0.3, 0.5, 0.8, 1]:#p.linspace(0,1,21):#[0]:#np.linspace(-1,1,21):
    pp = round(pp, 4)
    

    #define path
    #num of points
    qpoints = 1001
    q0 = np.array([pp,0])
    v1 = np.array([0,2])
    v2 = np.array([2,0])
    q1 = q0+v1
    q2 = q0+2*v1
    q3 = q0+3*v1
    q4 = q0+4*v1
    
    # qf = q0+ np.array([0,4])
    # kline = np.linspace(q0, qf, qpoints)
    # kline = SquareLine(q0, v1, v2, int((qpoints+3)/4))
    
    kline = FivePointLine(q0, q1, q2, q3, q4, int((qpoints+3)/4))
    
    #get evecs at gamma point
    H = Ham(q0)
    _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix = 1) # may as well gauge fix here
    #define our basis
    u0 = evecs[:,0]
    u1 = evecs[:,1]
    u2 = evecs[:,2]
    #check it is not a dirac point
    assert(np.round(np.linalg.norm(evecs[:,n1]),10)==1)
    
    
    thetasLineG1 = np.zeros(qpoints)
    thetasLineG2 = np.zeros(qpoints)
    
    alphasLineG1 = np.zeros(qpoints)
    alphasLineG2 = np.zeros(qpoints)
    
    alphasLineG11 = np.zeros(qpoints)
    alphasLineG12 = np.zeros(qpoints)
    alphasLineG21 = np.zeros(qpoints)
    alphasLineG22 = np.zeros(qpoints)
    alphasLineCosG11 = np.zeros(qpoints)
    alphasLineCosG12 = np.zeros(qpoints)
    alphasLineCosG21 = np.zeros(qpoints)
    alphasLineCosG22 = np.zeros(qpoints)
    
    # go through possible end points for k, get angles
    for i, kpoint in enumerate(kline):
        #do abeliean version,
        #find evecs at other k down the line
        H = Ham(kpoint)
        _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0)
        uFinal = evecs[:,n1]
        
        #get correct overall phase for uFinal
        uFinalG1 = AlignGaugeBetweenVecs(u0, uFinal)
        # uFinalG1 = FirstElementVecPositive(uFinal)
        uFinalG2 = -uFinalG1
    
        # get params
        
        # get params
        argumentG1 = ProjReal(np.vdot(u0, uFinalG1))
        #only want the values of theta between 0 and pi because we on bloch sphere
        # therefore we neglect second arccos value
        thetaG1 = np.arccos(argumentG1)
        # thetaOtherGauge = pi-theta0
        argumentG2 = ProjReal(np.vdot(u0, uFinalG2))
        thetaG2 = np.arccos(argumentG2)
        
        
        
        alphaargG1 = ProjReal(np.vdot(u1, uFinalG1)/sin(thetaG1))
        alphaG1_1, alphaG1_2 = InverseSin(alphaargG1)
        # alphaOtherGauge = alpha - pi
        
        alphaargG2 = ProjReal(np.vdot(u1, uFinalG2)/sin(thetaG2))
        alphaG2_1, alphaG2_2 = InverseSin(alphaargG2)
    
    
        alphaCosArgG1 = ProjReal(np.vdot(u2, uFinalG1)/sin(thetaG1))
        alphaCosG1_1, alphaCosG1_2 = InverseCos(alphaCosArgG1)
        
        alphaCosArgG2 = ProjReal(np.vdot(u2, uFinalG2)/sin(thetaG2))
        alphaCosG2_1, alphaCosG2_2 = InverseCos(alphaCosArgG2)
        
        # print(alphaargG1)
                               
                                              
        thetasLineG1[i] = thetaG1
        thetasLineG2[i] = thetaG2
        
        alphasLineG1[i] = FindOverlap(alphaG1_1, alphaG1_2, alphaCosG1_1, alphaCosG1_2)
        alphasLineG2[i] = FindOverlap(alphaG2_1, alphaG2_2, alphaCosG2_1, alphaCosG2_2)
        
        alphasLineG11[i] = alphaG1_1
        alphasLineG12[i] = alphaG1_2
        alphasLineG21[i] = alphaG2_1
        alphasLineG22[i] = alphaG2_2
        alphasLineCosG11[i] = alphaCosG1_1
        alphasLineCosG12[i] = alphaCosG1_2
        alphasLineCosG21[i] = alphaCosG2_1
        alphasLineCosG22[i] = alphaCosG2_2
    

    
    print(VecToStringSave(q0))
    

    # saveLine = "LineTraj,Euler=4,GroundState,GaugeFixToGamma="+VecToStringSave(q0)+",V1="+VecToStringSave(v1)+",V2="+VecToStringSave(v2)+".png"
    # saveLine = ("LineTraj,Euler=4,GroundState,GaugeFixToGamma="+VecToStringSave(q0)
    #             +",v1="+VecToStringSave(v1)+",v2="+VecToStringSave(v2)+".png")
    saveLine = ("LineTraj,Euler=4,GroundState,GaugeFixToGamma="+VecToStringSave(q0)
                +",qf="+VecToStringSave(4*v1)+".png")
    saveTheta = "Theta"+saveLine
    saveAlpha = "Alpha"+saveLine
    # ThetaLineTraj,Euler=4,GroundState,GaugeFixToGamma=(-0p8,0),qf=(-0p8,8).png
    # saveThetaDifferentiated = "Theta"+saveLineDifferentiate
    # saveAlphaDifferentiated = "Alpha"+saveLineDifferentiate
    
    # fs = (10,7.5)
    
    ms = 1
    multiplier = np.linspace(0, 4, qpoints)
    
    # fig, ax = plt.subplots(figsize=fs)
    # ax.plot(multiplier[0], thetasLine[0], '.', color=G1colour, markersize = 3, label=r"$\theta_{G1}$")
    # ax.plot(multiplier, thetasLine, '.', color = G1colour, markersize=ms)
    # ax.plot(multiplier, -thetasLine, '.', color = G1colour, markersize=ms)
    # # ax.plot(multiplier, 2*pi-thetasLine, '.', color = G1colour, markersize=3)
    # # ax.plot(multiplier, -2*pi+thetasLine, '.', color = G1colour, markersize=3)
    # ax.plot(multiplier[0], pi - thetasLine[0], '.', color = G2colour, markersize=3, label=r"$\theta_{G2}$")
    # ax.plot(multiplier, pi - thetasLine, '.', color = G2colour, markersize=ms)
    # ax.plot(multiplier, pi + thetasLine, '.', color = G2colour, markersize=ms)
    # ax.plot(multiplier, -pi + thetasLine, '.', color = G2colour, markersize=ms)
    # ax.plot(multiplier, -pi - thetasLine, '.', color = G2colour, markersize=ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # # ax.set_yticks([0, pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$"])
    # # ax.set_ylim(-0.1, pi/2+0.1)
    # ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
    # # ax.set_xlabel(r"$q_x$")
    # ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    # # ax.set_xlabel(r"square path")
    # ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # # plt.savefig(sh+saveTheta, format="png", bbox_inches="tight")
    # plt.show()    
    
    
    
    
    # fig, ax = plt.subplots(figsize=fs)
    # ax.plot(multiplier[0], alphasLine[0], '.', color = G1colour, markersize=3, label=r"$\alpha_{G1}$")
    # ax.plot(multiplier, alphasLine, '.', color = G1colour, markersize=ms)
    # ax.plot(multiplier, pi - alphasLine, '.', color = G1colour, markersize=ms)
    # ax.plot(multiplier, -pi - alphasLine, '.', color = G1colour, markersize=ms)
    # # ax.plot(multiplier, -2*pi + alphasLine, '.', color = G1colour, markersize=3)
    # ax.plot(multiplier[0],  - alphasLine[0], '.',color = G2colour,  markersize=3, label=r"$\alpha_{G2}$")
    # ax.plot(multiplier,  - alphasLine, '.',color = G2colour,  markersize=ms)
    # ax.plot(multiplier,  - pi + alphasLine, '.',color = G2colour,  markersize=ms)
    # ax.plot(multiplier,  pi + alphasLine, '.',color = G2colour,  markersize=ms)
    # # ax.plot(multiplier,  2*pi - alphasLine, '.',color = G2colour,  markersize=3)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # # ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
    # # ax.set_yticks([-pi/2, 0, pi/2])
    # # ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # # ax.set_yticklabels([ r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$"])
    # # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    # ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
    # # ax.set_xlabel(r"$q_x$")
    # ax.set_xticks([0,1,2,3,4])
    # # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    # ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    # ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    # plt.show()    

    
    # fig, ax = plt.subplots(figsize=fs)
    # ax.plot(multiplier[0], alphasLineCos[0], '.', color = G1colour, markersize=3, label=r"$\alpha_{G1}$")
    # ax.plot(multiplier, alphasLineCos, '.', color = G1colour, markersize=ms)
    # ax.plot(multiplier, - alphasLineCos, '.', color = G1colour, markersize=ms)

    # ax.plot(multiplier[0],  pi + alphasLine[0], '.',color = G2colour,  markersize=3, label=r"$\alpha_{G2}$")
    # ax.plot(multiplier,   alphasLineCos - pi, '.',color = G2colour,  markersize=ms)
    # ax.plot(multiplier,  pi - alphasLineCos, '.',color = G2colour,  markersize=ms)
    # ax.plot(multiplier,  pi + alphasLineCos, '.',color = G2colour,  markersize=ms)
    # ax.plot(multiplier,  -pi - alphasLineCos, '.',color = G2colour,  markersize=ms)
    # # ax.set_xlabel(r"final quasimomentum, going around circle with centre (0,0), ground band")
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # # ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
    # # ax.set_yticks([-pi/2, 0, pi/2])
    # # ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # # ax.set_yticklabels([ r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$"])
    # # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    # ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
    # # ax.set_xlabel(r"$q_x$")
    # ax.set_xticks([0,1,2,3,4])
    # # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    # ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    # ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    # plt.show()    
    

    




    
    fs = (10,3.5)
    fig, ax = plt.subplots(figsize=fs)
    # ax.plot(multiplier[0], thetasLineG1[0], '.', color=G1colour, markersize = 3, label=r"$\theta_{G1}$")
    ax.plot(multiplier, thetasLineG1,'.', color = col1,  markersize=ms,label=r"$\theta_{G1}$")
    ax.plot(multiplier, thetasLineG2, '.', color = col2,  markersize=ms, label=r"$\theta_{G2}$")
    ax.set_yticks([ 0, pi/2, pi])
    # ax.set_yticks([0, pi/2])
    ax.set_yticklabels([ "0",r"$\frac{\pi}{2}$", r"$\pi$"])
    # ax.set_yticklabels(['0',r"$\frac{\pi}{2}$"])
    # ax.set_ylim(-0.1, pi/2+0.1)
    ax.set_ylabel(r"$\theta$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    # ax.set_xlabel(r"square path")
    ax.grid(b=1, color='1')
    ax.legend(loc="upper right")
    plt.savefig(sh+saveTheta, format="png", bbox_inches="tight")
    plt.show()   
    
    fs = (10,6.5)
    fig, ax = plt.subplots(figsize=fs)
    
    ax.plot(multiplier, alphasLineG11, '.',  color = col1, markersize=ms+5, label=r"$\alpha_{G1}^1$")
    ax.plot(multiplier, alphasLineG12, '.',  color = col1, markersize=ms+5, label=r"$\alpha_{G1}^1$")
    ax.plot(multiplier, alphasLineG21, '.',  color = col2, markersize=ms+5, label=r"$\alpha_{G2}^1$")
    ax.plot(multiplier, alphasLineG22, '.',  color = col2, markersize=ms+5, label=r"$\alpha_{G2}^1$")
    
    ax.plot(multiplier, alphasLineCosG11, '.',  color = col3, markersize=ms, label=r"$\alpha_{G1}^2$")
    ax.plot(multiplier, alphasLineCosG12, '.',  color = col3, markersize=ms, label=r"$\alpha_{G1}^2$")
    ax.plot(multiplier, alphasLineCosG21, '.',  color = col5, markersize=ms, label=r"$\alpha_{G2}^2$")
    ax.plot(multiplier, alphasLineCosG22, '.',  color = col5, markersize=ms, label=r"$\alpha_{G2}^2$")
    ax.set_yticks([ 0, pi/2, pi, 3*pi/2, 2*pi])
    # ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
    # ax.set_yticks([-pi/2, 0, pi/2])
    # ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_yticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$", r"$2 \pi$"])
    # ax.set_yticklabels([ r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()  
    
    fs = (10,6.5)
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, alphasLineG1, '.',  color = col1, markersize=ms, label=r"$\alpha_{G1}$")
    ax.plot(multiplier, alphasLineG2, '.',  color = col2, markersize=ms, label=r"$\alpha_{G2}$")
    ax.set_yticks([ 0, pi/2, pi, 3*pi/2, 2*pi])
    # ax.set_yticks([-pi, -pi/2, 0, pi/2, pi])
    # ax.set_yticks([-pi/2, 0, pi/2])
    # ax.set_yticklabels([ r"$-\pi$", r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$", r"$\pi$"])
    ax.set_yticklabels([ '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$", r"$2 \pi$"])
    # ax.set_yticklabels([ r"$-\frac{\pi}{2}$",'0',r"$\frac{\pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$\alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    ax.legend(loc="upper right")
    plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()  



        

#%% 

"""
Plot actual values of cos(theta), not thetha
"""

G1colour = 'darkblue'
G2colour = "#DD6031"
#band that we looking to describe
n1 = 0



Ham = Euler2Hamiltonian
    

for pp in [-0.1]:#p.linspace(0,1,21):#[0]:#np.linspace(-1,1,21):
    pp = round(pp, 2)
    

    #define path
    #num of points
    qpoints = 1001
    q0 = np.array([pp,0])
    v1 = np.array([0,2])
    v2 = np.array([2,0])
    q1 = q0+v1
    q2 = q0+2*v1
    q3 = q0+3*v1
    q4 = q0+4*v1
    
    # qf = q0+ np.array([0,4])
    # kline = np.linspace(q0, qf, qpoints)
    # kline = SquareLine(q0, v1, v2, int((qpoints+3)/4))
    
    kline = FivePointLine(q0, q1, q2, q3, q4, int((qpoints+3)/4))
    
    #get evecs at gamma point
    H = Ham(q0)
    _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix = 1) # may as well gauge fix here
    u0 = evecs[:,0]
    u1 = evecs[:,1]
    u2 = evecs[:,2]
    #check it is not a dirac point
    assert(np.round(np.linalg.norm(evecs[:,n1]),10)==1)
    
    
    arg0 = np.zeros(qpoints)
    arg1 = np.zeros(qpoints)
    arg2 = np.zeros(qpoints)
    
    sinAlphaG1s = np.zeros(qpoints)
    sinAlphaG2s = np.zeros(qpoints)
    cosAlphaG1s = np.zeros(qpoints)
    cosAlphaG2s = np.zeros(qpoints)

    
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
        thetaG1 = np.arccos(argument)
        thetaG2 = pi - thetaG1
        arg0[i] = argument

        argument = np.vdot(u1, uFinal)
        sinAlphaG1 = argument/sin(thetaG1)
        sinAlphaG2 = argument/sin(thetaG2)
        arg1[i] = argument
        sinAlphaG1s[i] = sinAlphaG1 
        sinAlphaG2s[i] = sinAlphaG2
        
        argument = np.vdot(u2, uFinal)
        cosAlphaG1 = argument/sin(thetaG1)
        cosAlphaG2 = argument/sin(thetaG2)
        arg2[i] = argument
        cosAlphaG1s[i] = cosAlphaG1 
        cosAlphaG2s[i] = cosAlphaG2
        
                                              

    
    fs = (10,7.5)
    ms = 0.7
    multiplier = np.linspace(0, 4, qpoints)
    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, arg0, '.', color=G1colour, markersize = ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # ax.set_ylim(-0.1, pi/2+0.1)
    ax.set_ylabel(r"$cos(\theta)$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    # ax.set_xlabel(r"square path")
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveTheta, format="png", bbox_inches="tight")
    plt.show()    
    
    
    
    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, arg1, '.', color = G1colour, markersize=ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$\sin \theta \sin \alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()    

    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, arg2, '.', color = G1colour, markersize=ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$\sin \theta \cos \alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show() 
    
    
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, sinAlphaG1s, '.', color = G1colour, markersize=ms)
    ax.plot(multiplier, sinAlphaG2s, '.', color = G2colour, markersize=ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$ \sin \alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()   



    fig, ax = plt.subplots(figsize=fs)
    ax.plot(multiplier, cosAlphaG1s, '.', color = G1colour, markersize=ms)
    ax.plot(multiplier, cosAlphaG2s, '.', color = G2colour, markersize=ms)
    # ax.set_yticks([-3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2])
    # ax.set_yticklabels([ r"$-\frac{3 \pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$",
    #                     '0',r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3 \pi}{2}$"])
    # ax.set_ylim([-pi/2-0.1, pi/2+0.1])
    ax.set_ylabel(r"$ \cos \alpha$", rotation=0, labelpad=15)
    # ax.set_xlabel(r"$q_x$")
    ax.set_xticks([0,1,2,3,4])
    # ax.set_xticklabels([VecToString(q0), VecToString(q0+v1), VecToString(q0+v1+v2), VecToString(q0+v2), VecToString(q0)])
    ax.set_xticklabels([VecToString(q0), VecToString(q1), VecToString(q2), VecToString(q3), VecToString(q4)])
    ax.grid(b=1, color='1')
    # ax.legend(loc="upper right")
    # plt.savefig(sh+saveAlpha, format="png", bbox_inches="tight")
    plt.show()   
    
#%% 
"""
Theta over a line - circle
"""

#band that we looking to describe
n1 = 0



Ham = EulerHamiltonian
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
  




    

