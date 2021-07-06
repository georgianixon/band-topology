# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:23:03 2021

@author: Georgia Nixon
"""
import numpy as np
from numpy import sqrt, exp, pi, cos, sin
from numpy.linalg import eig


import numpy as np
from numpy import sqrt, exp, pi, cos, sin
from numpy.linalg import eig
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits import mplot3d


t1 = np.array([
 [0.00885997 - 0.0151357 *1j, -0.0761286 + 0.0309107 *1j, -0.0025 - 
   0.00756786 *1j, 0.0811286 - 0.015775 *1j, -0.01386],
 [-0.0761286 + 0.0309107 *1j, -0.120513 - 0.0466857 *1j, 
  0.0025 + 0.0233429 *1j, 0.115513, 0.0811286 + 0.015775 *1j],
 [-0.0025 - 0.00756786 *1j, 0.0025 + 0.0233429 *1j, -0.0025, 
  0.0025 - 0.0233429 *1j, -0.0025 + 0.00756786 *1j],
 [0.0811286 - 0.015775 *1j, 0.115513, 
  0.0025 - 0.0233429 *1j, -0.120513 + 0.0466857 *1j, -0.0761286 - 
   0.0309107 *1j],
 [-0.01386, 
  0.0811286 + 0.015775 *1j, -0.0025 + 0.00756786 *1j, -0.0761286 - 
   0.0309107 *1j, 0.00885997 + 0.0151357*1j]])
 

t3 = np.array([
 [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025],
 [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
 [0.167664, -0.0425061, -0.0025, -0.0425061, 0.167664],
 [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
 [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025]
])

t4 = np.array([
 [0. + 0.0277532 *1j, 0. + 0.227548 *1j, 0. + 0.491746 *1j, 0. + 0.227548 *1j,
   0. + 0.0277532 *1j],
 [0. + 0.0635846 *1j, 0. + 0.114151 *1j, 0. - 0.280982 *1j, 0. + 0.114151 *1j,
   0. + 0.0635846 *1j],
 [0, 0, 0, 0, 0],
 [0. - 0.0635846 *1j, 0. - 0.114151 *1j, 0. + 0.280982 *1j, 0. - 0.114151 *1j,
   0. - 0.0635846 *1j],
 [0. - 0.0277532 *1j, 0. - 0.227548 *1j, 0. - 0.491746 *1j, 0. - 0.227548 *1j,
   0. - 0.0277532 *1j]
        ])

t6 = np.array([
 [0. + 0.0277532 *1j, 0. + 0.0635846 *1j, 0, 0. - 0.0635846 *1j, 
  0. - 0.0277532 *1j],
 [0. + 0.227548 *1j, 0. + 0.114151 *1j, 0, 0. - 0.114151 *1j, 
  0. - 0.227548 *1j],
 [0. + 0.491746 *1j, 0. - 0.280982 *1j, 0, 0. + 0.280982*1j, 
  0. - 0.491746 *1j],
 [0. + 0.227548 *1j, 0. + 0.114151 *1j, 0, 0. - 0.114151*1j, 
  0. - 0.227548 *1j],
 [0. + 0.0277532 *1j, 0. + 0.0635846*1j, 0, 0. - 0.0635846 *1j, 
  0. - 0.0277532 *1j]
        ])

t8 = np.array([
 [0, -0.187857, -0.433013, -0.187857, 0],
 [-0.187857, 0, 0.30825, 0, -0.187857],
 [-0.433013, 0.30825, 0, 0.30825, -0.433013],
 [-0.187857, 0, 0.30825, 0, -0.187857],
 [0, -0.187857, -0.433013, -0.187857, 0]
        ])


#Gell-Mann Matrices
GM0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
GM1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
GM2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
GM3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
GM4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
GM5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
GM6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
GM7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
GM8 = (1/sqrt(3))*np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

        

 #%%
 
Nx = 2;
Ny = 2;
lx = np.linspace(-Nx, Nx, 2*Nx+1)
ly = np.linspace(-Ny, Ny, 2*Nx+1)
Nnx = 2*Nx + 1
Nny = 2*Ny + 1

def h1N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t1[i,j] for i in range(Nnx) for j in range(Nny)])
def h3N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t3[i,j] for i in range(Nnx) for j in range(Nny)])
def h4N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t4[i,j] for i in range(Nnx) for j in range(Nny)])
def h6N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t6[i,j] for i in range(Nnx) for j in range(Nny)])
def h8N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t8[i,j] for i in range(Nnx) for j in range(Nny)])


def EulerHamiltonian(kx,ky):
    hjk = np.array([0, h1N(kx,ky), h3N(kx,ky), h4N(kx,ky), h6N(kx,ky), h8N(kx,ky)])
    gellManns = np.array([GM0, GM1, GM3, GM4, GM6, GM8])
    HFn = np.array([hjk[i]*gellManns[i] for i in range(len(hjk))])
    return np.sum(HFn, axis=0)

#%%

def GetEvalsAndEvecs(HF):
    """
    Get e-vals and e-vecs of Haniltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = conjugatePhase*evecs[:,vec]

    # check that the evals are real
    if np.all((np.round(np.imag(evals),7) == 0)) == True:
        return np.real(evals), evecs
    else:
        print('evals are imaginary!')
        return evals, evecs

def AlignEvecs(evecs0, evecs1, i):
    """
    evecs0 is a basis for the system
    Find evecs1 in this basis
    Make overlap between evec1 and evec0[i] real and positive
    This ensures evec1, written in the basis given by evec0, 
        has coefficient evec1 = a evec0[i] + ... where a is real and positive
    """
    
    
    evecs0_R = roundcomplex(evecs0, 5)
    evecsP_R = roundcomplex(evecsP, 5)
    
    for vec in range(N):

        # else, make them start at the same value
        elif evecs0_R[0,vec] != evecsP_R[0,vec]:
            frac = evecs0[0,vec] / evecsP[0,vec]
            evecsP[:,vec] = evecsP[:,vec] * frac
            #redefine evecsP_R
            evecsP_R[:,vec] = roundcomplex(evecsP[:,vec], 5)
            
    return evecsP


