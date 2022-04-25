# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:52:38 2022

@author: Georgia
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
from Funcs import VecToStringSave, AlignGaugeBetweenVecs
# from hamiltonians import GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm
import matplotlib as mpl

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/Euler Class/"


params = {
        # 'legend.fontsize': size*0.75,
        #   'axes.labelsize': size,
        #   'axes.titlesize': size,
        #   'xtick.labelsize': size*0.75,
        #   'ytick.labelsize': size*0.75,
        #   'font.size': size,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          'axes.grid':False,
          }
mpl.rcParams.update(params)

#%% 
"""
Brouwer Degree over the BZ
"""

#band that we looking to describe
n1 = 0
h = 0.0001

Ham = Euler0Hamiltonian

#points in the line
qpoints=51

for xval in [ 0]:
        
    # arbitrary point I guess, but not a dirac point
    gammaPoint = np.array([xval,0])
    
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
    
    brouwerDegG1 = np.zeros((qpoints,qpoints))
    brouwerDegG2 = np.zeros((qpoints,qpoints))
    
    eiglist = np.empty((qpoints,qpoints,3)) # for three bands
    
    for xi, qx in enumerate(K1):
        print(xi)
        for yi, qy in enumerate(K2):
            k = np.array([qx,qy])
            H = Ham(k)
            _, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0) # will gauge fix later
    
            uK = evecs[:,n1]
        
            #get correct overall phase for uFinal
            uK_G1 = AlignGaugeBetweenVecs(u0, uK)
            uK_G2 = -uK_G1
            
            #dx direction
            qxx = k + np.array([h,0])
            H = Ham(qxx)
            _,vx = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0)
            uKx = vx[:,n1] # first eigenvector
            #chose neighbouring gauge
            uKx_G1 = AlignGaugeBetweenVecs(u0, uKx)
            uKx_G2 = -uKx_G1
            
            #dy direction
            kyy = k+np.array([0,h])
            H = Ham(kyy)
            _, vy = GetEvalsAndEvecsEuler(H)
            uKy=vy[:,n1] # first eigenvector
            
            #choose neighbouring gauge
            uKy_G1 = AlignGaugeBetweenVecs(u0, uKy)
            uKy_G2 = -uKy_G1
        
            xder_G1 = (uKx_G1-uK_G1)/h
            yder_G1 = (uKy_G1-uK_G1)/h
            
            xder_G2 = (uKx_G2-uK_G2)/h
            yder_G2 = (uKy_G2-uK_G2)/h
    
            brouwD_G1 = (uK_G1[0]*(xder_G1[1]*yder_G1[2] - xder_G1[2]*yder_G1[1]) 
                       + uK_G1[1]*(xder_G1[2]*yder_G1[0] - xder_G1[0]*yder_G1[2]) 
                       + uK_G1[2]*(xder_G1[0]*yder_G1[1] - xder_G1[1]*yder_G1[0]))
            
            brouwD_G2 = (uK_G2[0]*(xder_G2[1]*yder_G2[2] - xder_G2[2]*yder_G2[1]) 
                       + uK_G2[1]*(xder_G2[2]*yder_G2[0] - xder_G2[0]*yder_G2[2]) 
                       + uK_G2[2]*(xder_G2[0]*yder_G2[1] - xder_G2[1]*yder_G2[0]))
            
            brouwerDegG1[xi,yi] = brouwD_G1
            brouwerDegG2[xi,yi] = brouwD_G2
    
    cmap = "RdYlGn"#"plasma"#"RdYlGn"#"plasma"
    
    brouwerDegSignG1 = np.sign(brouwerDegG1)
    brouwerDegSignG2 = np.sign(brouwerDegG2)
    brouwerDegSignPlotG1 =  np.flip(brouwerDegSignG1.T, axis=0)
    brouwerDegSignPlotG2 =  np.flip(brouwerDegSignG2.T, axis=0)
    
    
    sz = 9
    title = "sum="+str(int(np.sum(brouwerDegSignPlotG1)))
    savename = "BrouwerDeg,Euler=0,G1,Ref="+VecToStringSave(gammaPoint)+".png"
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(brouwerDegSignPlotG1)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
    ax.set_title(title)
    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    cbar.set_ticks([-1, +1])
    cbar.set_ticklabels(["-1", "+1"])
    plt.savefig(sh+savename, format="png", bbox_inches="tight")
    plt.show()
    
    savename = "BrouwerDeg,Euler=0,G2,Ref="+VecToStringSave(gammaPoint)+".png"
    
    sz = 9
    title = "sum="+str(int(np.sum(brouwerDegSignPlotG2)))
    fig, ax = plt.subplots(figsize=(sz/2,sz/2))
    pos = plt.imshow(brouwerDegSignPlotG2)
    ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
    ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
    ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
    ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
    ax.set_title(title)
    cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
    cbar.set_ticks([-1, +1])
    cbar.set_ticklabels(["-1", "+1"])
    plt.savefig(sh+savename, format="png", bbox_inches="tight")
    plt.show()










