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
from hamiltonians import GetEvalsAndEvecs, PhiString, getevalsandevecs


def HaldaneHamiltonianNur(k, params):
    
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
    Can confirm this forms the same Hamiltonian as Nur's version above 
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
    Same as in GrapheneFuncs.py
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

    H = HaldaneHamiltonian(k, params)
    
    d0,v0 = GetEvalsAndEvecs(H)
    
    #first eigenvector
    u0=v0[:,n0]
    u1=v0[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = HaldaneHamiltonian(kxx, params)
    dx,vx = GetEvalsAndEvecs(H)
    ux1 = vx[:,n1]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = HaldaneHamiltonian(kyy, params)
    dy,vy = GetEvalsAndEvecs(H)
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



