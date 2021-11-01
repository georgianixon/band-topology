# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:23:10 2021

@author: Georgia
"""
import numpy as np
import numpy.linalg as la

"Pauli matrices"
sigma0 = np.eye(2)
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])

# "Kronecker products of Pauli matrices"
# a00 = np.kron(np.eye(2), np.eye(2))
# a0x = np.kron(np.eye(2), sigmax)
# a0y = np.kron(np.eye(2), sigmay)
# a0z = np.kron(np.eye(2), sigmaz)
# ax0 = np.kron(sigmax, np.eye(2))
# axx = np.kron(sigmax, sigmax)
# axy = np.kron(sigmax, sigmay)
# axz = np.kron(sigmax, sigmaz)
# ay0 = np.kron(sigmay, np.eye(2))
# ayx = np.kron(sigmay, sigmax)
# ayy = np.kron(sigmay, sigmay)
# ayz = np.kron(sigmay, sigmaz)
# az0 = np.kron(sigmaz, np.eye(2))
# azx = np.kron(sigmaz, sigmax)
# azy = np.kron(sigmaz, sigmay)
# azz = np.kron(sigmaz, sigmaz)

f0001 = np.array([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
f0010 = np.array([[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]])
f0100 = np.array([[0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]])
f1000 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]])

"Useful operations and values"
def Rmat(A): #rotations by A radians
    return np.array([[np.cos(A),np.sin(A)],[-np.sin(A),np.cos(A)]])
def dag(U): #dagger
    return np.transpose(np.conj(U))
def cc(U):  #complex conjugation
    return np.conj(U)
s3 = np.sqrt(3)
pi = np.pi


#%%

def Haldane3(kvec, var_vec):
    """
    This is a version of the Haldane Chern insulator, with a TRS copy.
    I have now also added a third-nearest neighbour hopping (t3)! (https://arxiv.org/pdf/1605.04768.pdf)
    The real-space unit cell used here is C3 symmetric, unlike in the arXiv above.
    Note that I've set the primary hopping parameter t1 == 1, for simplicity.
    I've also fixed phi to be pi/2.
    Basic model explanation can be found at URL: https://topocondmat.org/w4_haldane/haldane_model.html
    """ 
    if var_vec == []:
        phi, m, t1, t2, t3, l_R = [pi/2, 0.1, 1, 0.1, 0, 0]
    else: 
        phi, m, t1, t2, t3, l_R = var_vec
    
    
    "Lattice vectors"
    a1 = np.array([0  , 1])
    a2 = np.array([np.sqrt(3)/2 , -1/2])
    a3 = -a1-a2
    b1= a2-a1
    b2 = a3-a2
    b3 = a1-a3
    c1 = a1+a2-a3
    c2 = -a1+a2+a3
    c3 = a1-a2+a3
    
    "Products of kvec with lattice vectors"
    ka1 = np.dot(kvec,a1)
    ka2 = np.dot(kvec,a2)
    ka3 = np.dot(kvec,a3)
    kb1 = np.dot(kvec,b1)
    kb2 = np.dot(kvec,b2)
    kb3 = np.dot(kvec,b3)
    kc1 = np.dot(kvec,c1)
    kc2 = np.dot(kvec,c2)
    kc3 = np.dot(kvec,c3)
    
    d1 = t1*(np.cos(ka1) + np.cos(ka2) + np.cos(ka3)) + t3*(np.cos(kc1) + np.cos(kc2) + np.cos(kc3))
    d2 = t1*(- np.sin(ka1) - np.sin(ka2) - np.sin(ka3)) + t3*(-np.sin(kc1) - np.sin(kc2) - np.sin(kc3)) 
    d3 = m + 2*t2*(np.sin(kb1) + np.sin(kb2) + np.sin(kb3))
    d3_ = m - 2*t2*(np.sin(kb1) + np.sin(kb2) + np.sin(kb3)) #time-reversed copy
    
    H  = d1*sigmax + d2*sigmay + d3*sigmaz
    H2 = d1*sigmax + d2*sigmay + d3_*sigmaz 
    
    Hupleft = np.kron(0.5*(sigma0+sigmaz), H)
    Hlowright = np.kron(0.5*(sigma0-sigmaz),H2)
    
    "Spin-orbit coupling that preserves TRS & C3 symmetry" 
    "Third-nn Rashba SOC:"
    ras1 = (-1j+s3)*np.exp(1j*kc1) + 2*1j*np.exp(1j*kc2) + (-1j - s3)*np.exp(1j*kc3)
    ras2 = (-1j-s3)*np.exp(1j*kc1) + 2*1j*np.exp(1j*kc2) + (-1j + s3)*np.exp(1j*kc3)
    #f0001 and others defined at beginning of this module!
    H_R3 = l_R*(ras1*f0001 + cc(ras2)*f0010 + ras2*f0100 + cc(ras1)*f1000) #3rd nn Rashba SOC
    
    return H_R3 + Hupleft + Hlowright 


#%%


def gap(Ham, var_vec, N):
    "Ham is the Hamiltonian, var_vec the parameters, and N the number of steps taken along each k-direction in the grid"
    kx_path = np.linspace(-np.pi, np.pi, N)
    ky_path = np.linspace(-np.pi, np.pi, N)
    H = Ham((0,0), var_vec)
    n = H.shape[0]
    evals = la.eigh(H)[0]
    gap = np.abs(evals[n//2]- evals[n//2 -1])
    for kx in kx_path:
        for ky in ky_path:
            evals = la.eigh(Ham((kx,ky), var_vec))[0]
            gap_n = np.abs(evals[n//2]- evals[n//2 -1])
            if gap_n < gap:
                gap = gap_n
    return gap

#%%

"""
Definitions of the BZ points
"""
G =    np.array([0,0])                          #Gamma

K_1 =  np.array([ 2*pi/(3*s3), 2*pi/3])       # Right upper K
K_p1 = np.array([4*pi/(3*s3),  0])            # Rightmost K'
K_2 =  np.array([2*pi/(3*s3),  -2*pi/3])      # Right lower K
K_p2 = np.array([-2*pi/(3*s3), -2*pi/3])      # Left lower K'
K_3 =  np.array([-4*pi/(3*s3),  0])           # Leftmost K
K_p3 = np.array([-2*pi/(3*s3),  2*pi/3])      # Left upper K'

M_1 =  np.array([0,      2*pi/3])             # Uppermost M
M_2 =  np.array([pi/s3,  pi/3])               # Right upper M
M_3 =  np.array([pi/s3,  -pi/3])              # Right lower M
M_4 =  np.array([0,      -2*pi/3])            # Lowest M
M_5 =  np.array([-pi/s3, -pi/3])              # Left lower M
M_6 =  np.array([-pi/s3, pi/3])               # Left upper M

def k_path(point_list, loop=True, N=200):
    '''
    This function produces the path connecting the set of points given in the point_list input.
    The path will be closed if loop is True and will be open if False.
    N is the number of points between subsequent points in point_list in the path.
    One can check the stability of the result by increasing N further.
    '''
    k_trajectory = np.empty([0,2])
    k_trajectory = np.append(k_trajectory,np.linspace(point_list[0],point_list[0],1),axis=0)
    for i in range(len(point_list)-1):
        segment = np.linspace(point_list[i], point_list[i+1], N)[1:] #note: requires numpy version > 1.15
        k_trajectory = np.append(k_trajectory, segment, axis=0)
    if loop==True:
        segment = np.linspace(point_list[-1], point_list[0], N)[1:]
        k_trajectory = np.append(k_trajectory, segment, axis=0)
    return k_trajectory

def k_loop_hex(j,N, P0 = 'G'):
    "The hexagonal loops all encircle P0 (normally Gamma, but can also be K, K')"
    "Does not go through P0"
    if P0 == 'G':
        p0 = G
    elif P0 == 'K':
        p0 = K_1
    elif P0 == 'Kp':
        p0 = K_p1
    else: raise ValueError("P0 should be 'G', 'K' or 'Kp'.")
    
    "N is the number of loops taken"
    "j is the step index in terms of size, which runs from 0 to N"
    "j = 0 corresponds to taking no loop (G-G-G-G)"
    step = j/N 
    
    "points is the sequence around Gamma, partially along each subsequent G-K direction"
    points = [p0 + step*K_2, p0 + step*K_p1, p0 + step*K_1, p0 + step*K_p3, p0 + step*K_3, p0 + step*K_p2]
    return k_path(points, True, 2*j+1)

#%%

def Wilson_loop(k_path, Hamil, var_vec, ns):
    '''
    ns is the index of bands. It must be given in pairs, e.g. ns=[0,1]  or [0,1, 2,3, 4,5].
    Namely, if we have ns = [0,1], the computation will be done over 1st and 2nd band corresponding to first Kramer pair.

    The output of this function is a Wilson loop matrix - its eigenvalues are the Z2 invariant.
    '''
    for i,k_vec in enumerate(k_path):
        H = Hamil(k_vec, var_vec)
        W = np.zeros((len(ns),len(ns)), dtype=complex)
        if i == 0:                                       #first k-point in k_path
            evals, ekets = la.eigh(H)
            if len(ns) == 2:
                result = np.array([[1+0j,0j],[0j,1+0j]]) # np.eye(2, dtype=np.complex128)
            elif len(ns) == 4:
                result=np.array([[1+0j,0j, 0j,0j],[0j,1+0j,0j,0j],[0j,0j,1+0j,0j],[0j,0j,0j,1+0j]]) #np.eye(4, dtype=np.complex128)
            evs = ekets
        else:
            evals, ekets = la.eigh(H) #uniTrans(old, H)
            for u in ns:
                for v in ns:
                    W[u%(len(ns))][v%(len(ns))] = np.vdot(old[:,u], ekets[:,v])
            result = np.dot(result, W) # matrix multiplication of wilson lines
        old=ekets
    
    "IMPORTANT: With some k-paths don't start from j=0, or you'll get an error from this function!"
    for u in ns:
        for v in ns:
            W[u%(len(ns))][v%(len(ns))] = np.vdot(old[:,u], evs[:,v])
    result = np.dot(result, W)
    
    return result

def Z2_invariant(k_path, Hamil, var_vec, ns):
    "Computes the WL eigenvalues, which should be plotted as a spectrum to determine the topological invariants in each phase"
    WL = Wilson_loop(k_path, Hamil, var_vec, ns)
    return [np.angle(la.eigvals(WL)[i%(len(ns))])/(2*np.pi) for i in ns]        #taking the eigenvalues of Wilson loop


#%%

""" Georgia Contributions"""

place = "Georgia"
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/extended-haldane')
from ExtendedHaldaneModel import  ExtendedHaldaneHamiltonianRashbaCoupling

phi = pi/2
t1 = 1
t2 = 0.5
t3 = 0.5
m = 0.1
l_R = 0.3
params = [phi, m, t1, t2, t3, l_R]
kvec = [0.6, 0.6]

Ham1 = Haldane3(kvec, params)





























