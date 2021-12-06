# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:23:03 2021

@author: Georgia Nixon
"""
import numpy as np
from numpy.linalg import eig
from scipy.linalg import eigh 
from numpy import sqrt, exp, pi, cos, sin



t1 = np.array([[0, -0.0626378 + 0.00373984*1j, -0.113747 + 0.0240204*1j, 0. + 0.0000660094*1j,
0.113747 - 0.0241524*1j, 0.0626378 - 0.00360782*1j, 0. - 0.000132019*1j],
[0.0626378 - 0.00373984*1j, 0, -0.0421371 - 0.0277602*1j, 0. + 0.00367383*1j,
0.0421371 + 0.0204126*1j, 0. + 0.00734766*1j, -0.0626378 - 0.00360782*1j],
[0.113747 - 0.0240204*1j, 0.0421371 + 0.0277602*1j, 0, 0. + 0.0240864*1j,
0. - 0.0481728*1j, -0.0421371 + 0.0204126*1j, -0.113747 - 0.0241524*1j],
[0. - 0.0000660094*1j, 0. - 0.00367383*1j, 0. - 0.0240864*1j, 0,
0. + 0.0240864*1j, 0. + 0.00367383*1j, 0. + 0.0000660094*1j],
[-0.113747 + 0.0241524*1j, -0.0421371 - 0.0204126*1j, 0. + 0.0481728*1j,
0. - 0.0240864*1j, 0, 0.0421371 - 0.0277602*1j, 0.113747 + 0.0240204*1j],
[-0.0626378 + 0.00360782*1j, 0. - 0.00734766*1j, 0.0421371 - 0.0204126*1j,
0. - 0.00367383*1j, -0.0421371 + 0.0277602*1j, 0, 0.0626378 + 0.00373984*1j],
[0. + 0.000132019*1j, 0.0626378 + 0.00360782*1j, 0.113747 + 0.0241524*1j,
0. - 0.0000660094*1j, -0.113747 - 0.0240204*1j, -0.0626378 - 0.00373984*1j, 0]])
 

t3 = np.array(
[[0.0297665, 0.012528, -0.0431533, -0.0432153, -0.0431533, 0.012528, 0.0297665],
[0.012528, 0.165214, 0.0549507, -0.0753167, 0.0549507, 0.165214, 0.012528],
[-0.0431533, 0.0549507, -0.108615, 0.0600891, -0.108615, 0.0549507, -0.0431533],
[-0.0432153, -0.0753167, 0.0600891,
0.00729899, 0.0600891, -0.0753167, -0.0432153],
[-0.0431533, 0.0549507, -0.108615, 0.0600891, -0.108615, 0.0549507, -0.0431533],
[0.012528, 0.165214, 0.0549507, -0.0753167, 0.0549507, 0.165214, 0.012528],
[0.0297665, 0.012528, -0.0431533, -0.0432153, -0.0431533, 0.012528, 0.0297665]])

t4 = np.array(
[[0, 0.0530598, 0.148045, 0.199493, 0.148045, 0.0530598, 0],
[-0.0530598, 0, 0.065518, 0.0839606, 0.065518, 0, -0.0530598],
[-0.148045, -0.065518, 0, -0.312349, 0, -0.065518, -0.148045],
[-0.199493, -0.0839606, 0.312349, 0, 0.312349, -0.0839606, -0.199493],
[-0.148045, -0.065518, 0, -0.312349, 0, -0.065518, -0.148045],
[-0.0530598, 0, 0.065518, 0.0839606, 0.065518, 0, -0.0530598],
[0, 0.0530598, 0.148045, 0.199493, 0.148045, 0.0530598, 0]])


t6 = np.array(
[[0.0507663, 0.0735257, -0.0489735, 0, 0.0489735, -0.0735257, -0.0507663],
[0.0735257, 0.1579, 0.295589, 0, -0.295589, -0.1579, -0.0735257],
[-0.0489735, 0.295589, -0.249347, 0, 0.249347, -0.295589, 0.0489735],
[0, 0, 0, 0, 0, 0, 0],
[0.0489735, -0.295589, 0.249347, 0, -0.249347, 0.295589, -0.0489735],
[-0.0735257, -0.1579, -0.295589, 0, 0.295589, 0.1579, 0.0735257],
[-0.0507663, -0.0735257, 0.0489735, 0, -0.0489735, 0.0735257, 0.0507663]])

t8 = np.array(
[[0, 0.0461421, 0, -0.114789, 0, 0.0461421, 0],
[0.0461421, 0, 0.187857, 0.433013, 0.187857, 0, 0.0461421],
[0, 0.187857, 0, -0.30825, 0, 0.187857, 0],
[-0.114789, 0.433013, -0.30825, 0, -0.30825, 0.433013, -0.114789],
[0, 0.187857, 0, -0.30825, 0, 0.187857, 0],
[0.0461421, 0, 0.187857, 0.433013, 0.187857, 0, 0.0461421],
[0, 0.0461421, 0, -0.114789, 0, 0.0461421, 0]])


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
 
Nx = 3;
Ny = 3;
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


def Euler4Hamiltonian(k):
    kx = k[0]
    ky = k[1]
    hjk = np.array([0, h1N(kx,ky), h3N(kx,ky), h4N(kx,ky), h6N(kx,ky), h8N(kx,ky)])
    gellManns = np.array([GM0, GM1, GM3, GM4, GM6, GM8])
    HFn = np.array([hjk[i]*gellManns[i] for i in range(len(hjk))])
    return np.sum(HFn, axis=0)

def GetEvalsAndEvecsEuler(HF):
    # HF = EulerHamiltonian(kx,ky)
    #assert hermitian, to 14 dp
    assert(np.all(np.round(np.conj(HF.T), 14)==np.round(HF, 14)))
    evals, evecs = eigh(HF) # evals are automatically real from eigh function
    
    #need to pick gauge such that first entry is positive
    # This will also guarantee every evec is fully
    evecs = np.round(evecs, 12)

    for vec in range(3):
    
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = conjugatePhase*evecs[:,vec]
    
    #check nothing is imaginary to 9dp
    assert(np.all(np.imag(np.round(evecs,9))==0))
    evecs = np.real(evecs)
    
    return evals, evecs

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
        assert(round(np.imag(c), 26)==0)
    
    return vec2
