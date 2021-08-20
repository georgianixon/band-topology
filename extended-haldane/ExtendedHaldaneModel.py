# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:14:05 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import sqrt, sin, cos, exp, pi

import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import GetEvalsAndEvecs


#pauli matrics
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

#nearest neighbor vecs
a1 = np.array([0, 1])
a2 = np.array([sqrt(3)/2, -1/2])
a3 = np.array([-sqrt(3)/2, -1/2])
a = np.array([a3, a1, a2])


# rotation matrices
Q6 = np.array([[1/2, -sqrt(3)/2], [sqrt(3)/2, 1/2]])
Q3 = np.array([[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]])

#n2 vec
b1 = np.array([sqrt(3), 0])
b2 = np.array([sqrt(3)/2, 3/2])
b3 = np.array([-sqrt(3)/2, 3/2])
b4 = -b1
b5 = -b2
b6 = -b3
b = np.array([b1, b2, b3, b4, b5, b6])


#n3 vecs
c1 = np.array([-sqrt(3), 1])
c2 = np.array([sqrt(3), 1])
c3 = np.array([0, -2])
c = np.array([c1, c3, c2])



def ExtendedHaldaneHamiltonian(k, params):
    
    M = params[0]
    lambdaR = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    
    H = np.zeros((4,4), dtype=np.complex128)

    d1 = np.sum([t1*cos(np.dot(k, a[i]))+t3*cos(np.dot(k, c[i])) for i in range(3)])
    d2 = np.sum([-t1*sin(np.dot(k, a[i])) - t3*sin(np.dot(k, c[i])) for i in range(3)])
    d3p = M+np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])
    d3m = M-np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])

    H[:2,:2] = d1*s1 + d2*s2 + d3p*s3
    H[2:,2:] = d1*s1 + d2*s2 + d3m*s3


    HR = 1j*lambdaR*np.sum(np.array([(c[i,0]*s2 - c[i,1]*s1)*
                                      exp(1j*np.dot(k, c[i])) for i in range(3)]), 
                            axis=0)

    H[0,1] = HR[0,0]
    H[1,0] = np.conj(HR[0,0])
    H[0,3] = HR[0,1]
    H[3,0] = np.conj(HR[0,1])
    H[2,1] = HR[1,0]
    H[1,2] = np.conj(HR[1,0])
    H[2,3] = HR[1,1]
    H[3,2] = np.conj(HR[1,1])
    
    return H

def ExtendedHaldaneHamiltonian0(k, params):
    
    M = params[0]
    lambdaR = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    
    H = np.zeros((2,2), dtype=np.complex128)

    d1 = np.sum([t1*cos(np.dot(k, a[i]))+t3*cos(np.dot(k, c[i])) for i in range(3)])
    d2 = np.sum([-t1*sin(np.dot(k, a[i])) - t3*sin(np.dot(k, c[i])) for i in range(3)])
    d3p = M+np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])
    d3m = M-np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])

    H = d1*s1 + d2*s2 + d3p*s3
    # H = d1*s1 + d2*s2 + d3m*s3
    
    return H



def HaldaneHamiltonian(k, params):
    
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]

    #nearest neighbor vectors
    a1 = np.array([1, 0])
    a2 = np.array([-1/2, sqrt(3)/2])
    a3 = np.array([-1/2, -sqrt(3)/2])
    
    #second nearest neighbor vectors -> relative chirality important
    b1 = np.array([3/2, sqrt(3)/2]) # b2
    b2 = np.array([0,-sqrt(3)]) # b4
    b3 = np.array([-3/2, sqrt(3)/2]) # b6

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


def BerryCurvature(Hamiltonian, k, params):
    
    h = 0.0001
    
    H = Hamiltonian(k,params)
    
    d0,v0 = GetEvalsAndEvecs(H)
                
    #first eigenvector
    u0=v0[:,0]
    lowerband = d0[0]
    upperband = d0[1] 
    
    #dx direction
    kxx = k + np.array([h,0])
    H = Hamiltonian(kxx, params)
    dx,vx = GetEvalsAndEvecs(H)
    ux = vx[:,0]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = Hamiltonian(kyy, params)
    dy,vy = GetEvalsAndEvecs(H)
    uy=vy[:,0]

    xder = (ux-u0)/h
    yder = (uy-u0)/h
    
    berrycurve = 2*np.imag(np.dot(np.conj(xder), yder))
    
    return berrycurve


