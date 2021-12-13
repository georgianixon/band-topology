# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:35:18 2021

@author: Georgia Nixon
"""

import numpy as np
from numpy import sqrt, exp, cos, sin
from scipy.linalg import eigh 

a1 = np.array([1,0])
a2 = 0.5*np.array([1, sqrt(3)])

M11 = np.array([[1,0,0],[0,0,0],[0,0,0]]) 
M12 = np.array([[0,1,0],[0,0,0],[0,0,0]]) 
M13 = np.array([[0,0,1],[0,0,0],[0,0,0]]) 

M22 = np.array([[0,0,0],[0,1,0],[0,0,0]]) 
M23 = np.array([[0,0,0],[0,0,1],[0,0,0]]) 

M33 = np.array([[0,0,0],[0,0,0],[0,0,1]]) 

def Kagome3(k):
    
    e1 = np.dot(k, a1)
    e2 = np.dot(k, a2)
    e3 = np.dot(k, a1-a2)
    
    p11 = 2*cos(e2)
    p22 = 2*cos(e3)
    p33 = 2*cos(e1)
    
    p12 = 1 + exp(1j*e1) + exp(1j*e2) + exp(-1j*e3)
    p13 = 1 + exp(1j*e1) + exp(-1j*e2) + exp(1j*e3)
    p23 = np.conj(p12)
    
    p21 = np.conj(p12)
    p31 = np.conj(p13)
    p32 = p12
    
    H = (M11*p11 + M22*p22 + M33*p33 + M12*p12 + M13*p13 + M23*p23
         + M12.T*p21 + M13.T*p31 + M23.T*p32)
    
    return H
    
    
    
def GetEvalsAndEvecsKagome(HF):

    #assert hermitian, to 14 dp (Euler Hamiltonian should definitely be Hermitian)
    assert(np.all(np.round(np.conj(HF.T), 14)==np.round(HF, 14)))
    evals, evecs = eigh(HF) # evals are automatically real from eigh function
    
    #need to pick gauge such that first entry is positive
    # This will also guarantee every evec is fully
    # evecs = np.round(evecs, 12)
    
    for vec in range(3):
    
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = conjugatePhase*evecs[:,vec]
    
    
    return evals, evecs   