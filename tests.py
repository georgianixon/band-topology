# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:25:06 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"

import numpy as np
from numpy import sin, cos, pi, sqrt, exp
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.linalg import expm
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/graphene-haldane')
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/extended-haldane')
from GrapheneFuncs import  HaldaneHamiltonian,  HaldaneHamiltonianPaulis, HaldaneHamiltonianNur

# from ExtendedHaldaneModel import  ExtendedHaldaneHamiltonian
from ExtendedHaldaneModel import HaldaneHamiltonian1, HaldaneHamiltonian2


#params

phi = pi/4
t1=1;
t2=1;
M=16#t2*3*sqrt(3)*sin(phi)-0.1;

params= [phi, M, t1, t2]



k = np.array([0.6,0.6])

HN = HaldaneHamiltonian1(k, params)
HM = HaldaneHamiltonian2(k, params)


apply = [
         np.abs, 
         np.real, np.imag]


hMax = np.max(np.stack((np.real(HN), np.imag(HN), np.abs(HN), 
                        np.real(HM), np.imag(HM), np.abs(HM)
                        )))
hMin = np.min(np.stack((np.real(HN), np.imag(HN), np.abs(HN), 
                        np.real(HM), np.imag(HM), np.abs(HM)
                        )))

bound = np.max((np.abs(hMax), np.abs(hMin)))
norm = mpl.colors.Normalize(vmin=-bound, vmax=bound)

for H in [HN, HM]:
    sz = 20
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                           figsize=(sz,sz/2))

    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr',  norm=norm)
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')
    ax[0].set_ylabel('n', rotation=0, labelpad=10)
    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(pcm)
    plt.show()
    
 #%%