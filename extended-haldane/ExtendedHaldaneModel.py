# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:14:05 2021

@author: Georgia Nixon
"""

place = "Georgia"
import numpy as np
from numpy import sqrt, sin, cos, exp, pi

import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import GetEvalsAndEvecs


#pauli matrics
s0 = np.array([[1,0],[0,1]])
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])
s = np.array([s1, s2, s3])

    

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
c1 = np.array([0, -2])
c2 = np.array([-sqrt(3), 1])
c3 = np.array([sqrt(3), 1])
c = np.array([c1, c2, c3])


def HaldaneHamiltonian(k, params):
    """
    Same as in GrapheneFuncs.py HaldaneHamiltonianPaulis
    Except all vecs are rotated
    """
    
    #params
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]

    cosasum = np.sum([cos(np.dot(a[i],k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i],k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])
    coscsum = np.sum([cos(np.dot(c[i], k)) for i in range(3)])
    sincsum = np.sum([sin(np.dot(c[i], k)) for i in range(3)])

    H = np.zeros([2,2], dtype=np.complex128)
    
    d0 = 2*t2*cos(phi)*cosbsum
    d1 = - t1*cosasum - t3*coscsum
    d2 = t1*sinasum + t3*sincsum
    d3 = M - 2*t2*sin(phi)*sinbsum
    H = H + d0*s0 + d1*s1 + d2*s2 + d3*s3
    
    return H



def ExtendedHaldaneHamiltonian(k, params):
    
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    
    H = np.zeros((4,4), dtype=np.complex128)
    
    cosasum = np.sum([cos(np.dot(a[i], k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i], k)) for i in range(3)])
    coscsum = np.sum([cos(np.dot(c[i], k)) for i in range(3)])
    sincsum = np.sum([sin(np.dot(c[i], k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])
    
    H0 = np.zeros([2,2], dtype=np.complex128)
    H1 = np.zeros([2,2], dtype=np.complex128)
    
    d0 = 2*t2*cos(phi)*cosbsum
    d1 =  - t1*cosasum - t3*coscsum
    d2 = + t1*sinasum + t3*sincsum
    d3p = M - 2*t2*sin(phi)*sinbsum
    d3m = M - 2*t2*sin(phi)*sinbsum
    H0 = H0 + d0*s0 + d1*s1 + d2*s2 + d3p*s3
    H1 = H1 + d0*s0 + d1*s1 + d2*s2 + d3m*s3

    H[:2,:2] = H0
    H[2:,2:] = H1
    
    return H


def ExtendedHaldaneHamiltonianRashbaCoupling(k, params):
    
    phi = params[0]
    M = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    lambdaR = params[5]
    
    H = np.zeros((4,4), dtype=np.complex128)
    
    cosasum = np.sum([cos(np.dot(a[i], k)) for i in range(3)])
    coscsum = np.sum([cos(np.dot(c[i], k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i], k)) for i in range(3)])
    sincsum = np.sum([sin(np.dot(c[i], k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])
    
    H0 = np.zeros([2,2], dtype=np.complex128)
    H1 = np.zeros([2,2], dtype=np.complex128)
    
    d0 = 2*t2*cos(phi)*cosbsum
    d1 =  t1*cosasum + t3*coscsum
    d2 = - t1*sinasum - t3*sincsum
    d3p = M + 2*t2*sin(phi)*sinbsum
    d3m = M - 2*t2*sin(phi)*sinbsum
    H0 = H0 + d0*s0 + d1*s1 + d2*s2 + d3p*s3
    H1 = H1 + d0*s0 + d1*s1 + d2*s2 + d3m*s3

    H[:2,:2] = H0
    H[2:,2:] = H1
    
    HR = 1j*lambdaR*np.sum(np.array([(c[i,0]*s2 - c[i,1]*s1)*
                                      exp(1j*np.dot(k, c[i])) for i in range(3)]), 
                            axis=0)

    H[0,1] = H[0,1] + HR[0,0]
    H[1,0] = H[1,0] + np.conj(HR[0,0])
    H[0,3] = H[0,3] + HR[0,1]
    H[3,0] = H[3,0] + np.conj(HR[0,1])
    H[2,1] = H[2,1] + HR[1,0]
    H[1,2] = H[1,2] + np.conj(HR[1,0])
    H[2,3] = H[2,3] + HR[1,1]
    H[3,2] = H[3,2] + np.conj(HR[1,1])
    
    return H

def ExtendedHaldaneHamiltonian2(k, params):
    
    M = params[0]
    lambdaR = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    phi = 0
    
    H = np.zeros((4,4), dtype=np.complex128)
    
    cosasum = np.sum([cos(np.dot(a[i], k)) for i in range(3)])
    coscsum = np.sum([cos(np.dot(c[i], k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i], k)) for i in range(3)])
    sincsum = np.sum([sin(np.dot(c[i], k)) for i in range(3)])
    cosbsum = np.sum([cos(np.dot(b[i],k)) for i in range(1,6,2)])
    sinbsum = np.sum([sin(np.dot(b[i],k)) for i in range(1,6,2)])
    
    H0 = np.zeros([2,2], dtype=np.complex128)
    H1 = np.zeros([2,2], dtype=np.complex128)
    
    # d0 = 2*t2*cos(phi)*cosbsum
    d1 =  t1*cosasum + t3*coscsum
    d2 = - t1*sinasum - t3*sincsum
    d3p = M + 2*t2*sin(phi)*sinbsum
    d3m = M - 2*t2*sin(phi)*sinbsum
    H0 = H0 + d0*s0 + d1*s1 + d2*s2 + d3p*s3
    H1 = H1 + d0*s0 + d1*s1 + d2*s2 + d3m*s3

    H[:2,:2] = H0
    H[2:,2:] = H1
    
    HR = 1j*lambdaR*np.sum(np.array([(c[i,0]*s2 - c[i,1]*s1)*
                                      exp(1j*np.dot(k, c[i])) for i in range(3)]), 
                            axis=0)

    H[0,1] = H[0,1] + HR[0,0]
    H[1,0] = H[1,0] + np.conj(HR[0,0])
    H[0,3] = H[0,3] + HR[0,1]
    H[3,0] = H[3,0] + np.conj(HR[0,1])
    H[2,1] = H[2,1] + HR[1,0]
    H[1,2] = H[1,2] + np.conj(HR[1,0])
    H[2,3] = H[2,3] + HR[1,1]
    H[3,2] = H[3,2] + np.conj(HR[1,1])
    
    return H

def ExtendedHaldaneHamiltonianPaper(k, params):
    
    M = params[0]
    lambdaR = params[1]
    t1 = params[2]
    t2 = params[3]
    t3 = params[4]
    
    H = np.zeros((4,4), dtype=np.complex128)
    
    cosasum = np.sum([cos(np.dot(a[i], k)) for i in range(3)])
    coscsum = np.sum([cos(np.dot(c[i], k)) for i in range(3)])
    sinasum = np.sum([sin(np.dot(a[i], k)) for i in range(3)])
    sincsum = np.sum([sin(np.dot(c[i], k)) for i in range(3)])
    
    sinbsum = np.sum([sin(np.dot(b[i], k)) for i in range(6)])

    d1 =  t1*cosasum + t3*coscsum
    d2 = - t1*sinasum - t3*sincsum
    d3p = M + t2*sinbsum
    d3m = M - t2*sinbsum

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

#%%






