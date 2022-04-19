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
from EulerClass2Hamiltonian import  Euler2Hamiltonian, GetEvalsAndEvecsEuler, AlignGaugeBetweenVecs
from EulerClass4Hamiltonian import Euler4Hamiltonian
from EulerClass0Hamiltonian import Euler0Hamiltonian

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
          }
mpl.rcParams.update(params)

#%% 
"""
Brouwer Degree over the BZ
"""

#band that we looking to describe
n1 = 0
h = 0.0001

Ham = Euler2Hamiltonian

#points in the line
qpoints=51

# arbitrary point I guess, but not a dirac point
gammaPoint = np.array([0.5,0])

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

brouwerDeg = np.zeros((qpoints,qpoints))

eiglist = np.empty((qpoints,qpoints,3)) # for three bands

for xi, qx in enumerate(K1):
    for yi, qy in enumerate(K2):
        k = np.array([qx,qy])
        H = Ham(k)
        eigs, evecs = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0) # will gauge fix later

        uK = evecs[:,n1]
    
        #get correct overall phase for uFinal
        uK = -AlignGaugeBetweenVecs(u0, uK)
        # uFinal1 = AlignGaugeBetweenVecs(u1, uFinal)
        # uFinal2 = AlignGaugeBetweenVecs(u2, uFinal)

        # excited (gapped) eigenvector
        # u0bx=v0[:,n0]
        # u0by=v0[:,n1]
        
        #eigenvalues
        # band1 = d0[0]
        # band2 = d0[1] 
        # band3 = d0[2]
        
        #dx direction
        qxx = k + np.array([h,0])
        H = Ham(qxx)
        _,vx = GetEvalsAndEvecsEuler(H, debug=1, gaugeFix=0)
        uKx = vx[:,n1] # first eigenvector
        
        #chose neighbouring gauge
        uKx = -AlignGaugeBetweenVecs(u0, uKx)
        
        #dy direction
        kyy = k+np.array([0,h])
        H = Ham(kyy)
        _, vy = GetEvalsAndEvecsEuler(H)
        uKy=vy[:,n1] # first eigenvector
        
        #choose neighbouring gauge
        uKy = -AlignGaugeBetweenVecs(u0, uKy)
    
        xder = (uKx-uK)/h
        yder = (uKy-uK)/h

        brouwD = (uK[0]*(xder[1]*yder[2] - xder[2]*yder[1]) 
                  + uK[1]*(xder[2]*yder[0] - xder[0]*yder[2]) 
                  + uK[2]*(xder[0]*yder[1] - xder[1]*yder[0]))
        
        brouwerDeg[xi,yi] = brouwD

cmap = "RdYlGn"#"plasma"#"RdYlGn"#"plasma"

brouwerDegSign = np.sign(brouwerDeg)
brouwerDegSignPlot =  np.flip(brouwerDegSign.T, axis=0)
sz = 9
fig, ax = plt.subplots(figsize=(sz/2,sz/2))
pos = plt.imshow(brouwerDegSignPlot)
ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
ax.set_xticklabels([kmin, round(kmin+(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+3*(kmax-kmin)/4, 2), kmax])
ax.set_yticklabels([kmax, round(kmin+3*(kmax-kmin)/4, 2), int((kmin+kmax)/2), round(kmin+(kmax-kmin)/4, 2), kmin])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
cbar = fig.colorbar(pos, cax = plt.axes([0.93, 0.128, 0.04, 0.752]))
cbar.set_ticks([-1, +1])
cbar.set_ticklabels(["-1", "+1"])
# fig.colorbar(pos, cax = plt.axes([0.98, 0.145, 0.045, 0.79]))
# plt.savefig(sh+savename, format="pdf", bbox_inches="tight")
plt.show()









