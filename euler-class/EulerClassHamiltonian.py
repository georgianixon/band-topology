# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:23:03 2021

@author: Georgia Nixon
"""
import numpy as np
from numpy import sqrt, exp, pi, cos, sin
from numpy.linalg import eig


def TunnelingMatrices(num):
    if num==1:
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
        return t1
    
    elif num ==3:
        t3 = np.array([
             [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025],
             [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
             [0.167664, -0.0425061, -0.0025, -0.0425061, 0.167664],
             [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
             [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025]
            ])
        return t3
    
    elif num == 4:
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
        return t4
    
    elif num == 6:
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
        return t6
    
    elif num == 8:
        t8 = np.array([
         [0, -0.187857, -0.433013, -0.187857, 0],
         [-0.187857, 0, 0.30825, 0, -0.187857],
         [-0.433013, 0.30825, 0, 0.30825, -0.433013],
         [-0.187857, 0, 0.30825, 0, -0.187857],
         [0, -0.187857, -0.433013, -0.187857, 0]
                ])
        return t8


#Gell-Mann Matrices

def GellMannMatrices(num):
    if num == 0:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif num == 1:
        return np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    elif num == 2:
        return np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    elif num ==3:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    elif num == 4:
        return np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    elif num ==5:
        return np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    elif num == 6:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    elif num == 7:
        return np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    elif num == 8:
        return (1/sqrt(3))*np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

def GetEvalsAndEvecs(HF):
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        phi = np.angle(evecs[0,vec])
        evecs[:,vec] = exp(-1j*phi)*evecs[:,vec]
        
#        evecs[:,vec] = np.conj(evecs[0,vec])/np.abs(evecs[0,vec])*evecs[:,vec]
        
        #nurs normalisation
        evecs[:,vec] = np.conj(evecs[1,vec])/np.abs(evecs[1,vec])*evecs[:,vec]
    
    if np.all((np.round(np.imag(evals),7) == 0)) == True:
        return np.real(evals), evecs
    else:
        print('evals are imaginary!')
        return evals, evecs

        

 #%%
 
Nx = 2;
Ny = 2;
lx = np.linspace(-Nx, Nx, 2*Nx+1)
ly = np.linspace(-Ny, Ny, 2*Nx+1)
Nnx = 2*Nx + 1
Nny = 2*Ny + 1

def h1N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*TunnelingMatrices(1)[i,j] for i in range(Nnx) for j in range(Nny)])
def h3N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*TunnelingMatrices(3)[i,j] for i in range(Nnx) for j in range(Nny)])
def h4N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*TunnelingMatrices(4)[i,j] for i in range(Nnx) for j in range(Nny)])
def h6N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*TunnelingMatrices(6)[i,j] for i in range(Nnx) for j in range(Nny)])
def h8N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*TunnelingMatrices(8)[i,j] for i in range(Nnx) for j in range(Nny)])


def EulerHamiltonian(kx,ky):
    hjk = np.array([0, h1N(kx,ky), h3N(kx,ky), h4N(kx,ky), h6N(kx,ky), h8N(kx,ky)])
    gellManns = np.array([GellMannMatrices(0), GellMannMatrices(1), GellMannMatrices(3), GellMannMatrices(4), GellMannMatrices(6), GellMannMatrices(8)])
    HFn = np.array([hjk[i]*gellManns[i] for i in range(len(hjk))])
    return np.sum(HFn, axis=0)
