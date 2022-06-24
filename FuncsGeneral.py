# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:50:18 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import GetEvalsAndEvecsGen, PhiString, getevalsandevecs





def AbelianCalcWilsonLine(evecsFinal, evecsInitial, dgbands=2):
    wilsonLineAbelian = np.zeros([dgbands, dgbands], dtype=np.complex128)
    
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            wilsonLineAbelian[n0,n1] = np.dot(np.conj(evecsFinal[:,n1]), evecsInitial[:,n0])
    return wilsonLineAbelian



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
        assert(round(np.imag(c), 9)==0)
    
    return vec2


def DifferenceLine(array2D):
    X = np.append(np.append(array2D[[-2]], array2D, axis=0), array2D[[1]], axis=0)
    xDiff = np.zeros((len(array2D), 2))
    for i in range(len(array2D)):
        xDiff[i] = np.array([X[i+2,0] - X[i,0], X[i+2,1] - X[i,1]])
    return xDiff



def CalculateBerryConnect(Hamiltonian, k, params, n0, n1):
    
    h = 0.0001;

    H = Hamiltonian(k, params)
    
    d0,v0 = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0=v0[:,n0]
    u1=v0[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = Hamiltonian(kxx, params)
    dx,vx = GetEvalsAndEvecs(H)
    ux1 = vx[:,n1]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = Hamiltonian(kyy, params)
    dy,vy = GetEvalsAndEvecs(H)
    uy1=vy[:,n1]

    xder = (ux1-u1)/h
    yder = (uy1-u1)/h
    
    berryconnect = 1j*np.array([np.dot(np.conj(u0),xder),np.dot(np.conj(u0),yder)])

    return berryconnect



def CalculateBerryConnectMatrix(Hamiltonian, k, params, dgbands = 2):
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnect(Hamiltonian, k, params, n0, n1)
    return berryConnect