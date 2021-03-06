# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:15:47 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import sqrt, sin, cos
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
sys.path.append("/Users/"+place+"/Code/MBQD/band-topology/")
from hamiltonians import GetEvalsAndEvecsGen, PhiString, getevalsandevecs
from FuncsGeneral import AlignGaugeBetweenVecs

def HaldaneHamiltonianNur(k, params):
    """
    Can confirm these three methods are all the same Hamiltonian
    """
    
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]

    #nearest neighbor vectors
    a1 = np.array([1, 0])
    a2 = np.array([-1/2, sqrt(3)/2])
    a3 = np.array([-1/2, -sqrt(3)/2])
    
    #second nearest neighbor vectors -> relative chirality important, going clockwise
    b1 = np.array([3/2, sqrt(3)/2]) 
    b2 = np.array([0,-sqrt(3)]) 
    b3 = np.array([-3/2, sqrt(3)/2]) 

    cosasum = cos(np.dot(a1,k)) + cos(np.dot(a2,k)) + cos(np.dot(a3,k))
    sinasum = sin(np.dot(a1,k)) + sin(np.dot(a2, k)) + sin(np.dot(a3, k))
    cosbsum = cos(np.dot(b1, k)) + cos(np.dot(b2,k)) + cos(np.dot(b3,k))
    sinbsum = sin(np.dot(b1,k)) + sin(np.dot(b2, k)) + sin(np.dot(b3, k))

    
    H = np.zeros([2,2], dtype=np.complex128)
    H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
    H[1,1] = -M+2*t2*cos(phi)*cosbsum + 2*t2*sin(phi)*sinbsum
    H[0,1]= -t1*(cosasum+1j*sinasum);
    H[1,0]=np.conj(H[0,1]);
    return H



def HaldaneHamiltonian(k, params):
    """
    Can confirm these three methods are all the same Hamiltonian1
    """
    
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]
    
    #nearest neighbor vecs
    a1 = np.array([0, 1])
    a2 = np.array([sqrt(3)/2, -1/2])
    a3 = np.array([-sqrt(3)/2, -1/2])
    a = np.array([a3, a1, a2])
    
    #n2 vec
    b1 = np.array([sqrt(3), 0])
    b2 = np.array([sqrt(3)/2, 3/2])
    b3 = np.array([-sqrt(3)/2, 3/2])
    b4 = -b1
    b5 = -b2
    b6 = -b3
    b = np.array([b1, b2, b3, b4, b5, b6])

    cosasum = np.sum([cos(np.dot(a[i],k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i],k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])

    
    H = np.zeros([2,2], dtype=np.complex128)
    H[0,0] =  M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
    H[1,1] = -M + 2*t2*cos(phi)*cosbsum + 2*t2*sin(phi)*sinbsum
    H[0,1]= -t1*(cosasum+1j*sinasum);
    H[1,0]=np.conj(H[0,1]);
    return H


def HaldaneHamiltonianPaulis(k, params):
    """
    Can confirm these three methods are all the same Hamiltonian!
    """
    
    #params
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]
    
    #pauli matrics
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s2 = np.array([[0,-1j],[1j,0]])
    s3 = np.array([[1,0],[0,-1]])
    
    #nearest neighbor vecs
    a1 = np.array([0, 1])
    a2 = np.array([sqrt(3)/2, -1/2])
    a3 = np.array([-sqrt(3)/2, -1/2])
    a = np.array([a3, a1, a2])
    
    #n2 vec
    b1 = np.array([sqrt(3), 0])
    b2 = np.array([sqrt(3)/2, 3/2])
    b3 = np.array([-sqrt(3)/2, 3/2])
    b4 = -b1
    b5 = -b2
    b6 = -b3
    b = np.array([b1, b2, b3, b4, b5, b6])

    cosasum = np.sum([cos(np.dot(a[i],k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i],k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])

    H = np.zeros([2,2], dtype=np.complex128)
    d0 = 2*t2*cos(phi)*cosbsum
    d1 = - t1*cosasum
    d2 = t1*sinasum
    d3 = M - 2*t2*sin(phi)*sinbsum
    H = H + d0*s0 + d1*s1 + d2*s2 + d3*s3
    return H



def CalculateBerryConnectGraphene(k, params, n0, n1):
    
    h = 0.0001;

    H = HaldaneHamiltonianPaulis(k, params)
    
    d0,v0 = GetEvalsAndEvecsGen(H)
    
    #first eigenvector
    u0=v0[:,n0]
    u1=v0[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = HaldaneHamiltonianPaulis(kxx, params)
    dx,vx = GetEvalsAndEvecsGen(H)
    ux1 = vx[:,n1]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = HaldaneHamiltonianPaulis(kyy, params)
    dy,vy = GetEvalsAndEvecsGen(H)
    uy1=vy[:,n1]

    xder = (ux1-u1)/h
    yder = (uy1-u1)/h
    
    berryconnect = 1j*np.array([np.dot(np.conj(u0),xder),np.dot(np.conj(u0),yder)])

    return berryconnect



def CalculateBerryConnectMatrixGraphene(k, params):
    dgbands=2
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnectGraphene(k, params, n0, n1)
    return berryConnect


def BerryCurvature(Hamiltonian, k, n0, n1, params):
    
    h = 0.0001
    
    H = Hamiltonian(k,params)
    
    d0,v0 = GetEvalsAndEvecsGen(H)
                
    #first eigenvector
    u0bx=v0[:,n0]
    u0by=v0[:,n1]
    
    #eigenvalues
    lowerband = d0[0]
    upperband = d0[1] 
    
    #dx direction
    kxx = k + np.array([h,0])
    H = Hamiltonian(kxx, params)
    dx,vx = GetEvalsAndEvecsGen(H)
    ux = vx[:,n0] # first eigenvector
    
    ux = AlignGaugeBetweenVecs(u0bx, ux)
    
    #dy direction
    kyy = k+np.array([0,h])
    H = Hamiltonian(kyy, params)
    dy,vy = GetEvalsAndEvecsGen(H)
    uy=vy[:,n1] # first eigenvector
    
    uy = AlignGaugeBetweenVecs(u0by, uy)

    xder = (ux-u0bx)/h
    yder = (uy-u0by)/h
    
    berrycurve = 2*np.imag(np.dot(np.conj(xder), yder))
    
    return berrycurve, lowerband, upperband


def BerryCurvature2(Hamiltonian, k, params):
    
    h = 0.0001
    
    H = Hamiltonian(k,params)
    
    d0,v0 = getevalsandevecs(H)
                
    #first eigenvector
    u0=v0[:,0]
    lowerband = d0[0]
    upperband = d0[1] 
    
    #dx direction
    kxx = k + np.array([h,0])
    H = Hamiltonian(kxx, params)
    dx,vx = getevalsandevecs(H)
    ux = vx[:,0]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = Hamiltonian(kyy, params)
    dy,vy = getevalsandevecs(H)
    uy=vy[:,0]

    xder = (ux-u0)/h
    yder = (uy-u0)/h
    
    berrycurve = 2*np.imag(np.dot(np.conj(xder), yder))
    
    return berrycurve, lowerband, upperband
