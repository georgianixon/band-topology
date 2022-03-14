# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:47:08 2022

@author: Georgia
"""

from numpy import exp, pi
import numpy as np


def gFunc(k):
    kx = k[0]
    ky = k[1]
    g = np.exp(-1j*kx*np.pi) - np.exp(-1j*ky*np.pi)
    return g

def hFunc(k):
    kx = k[0]
    ky = k[1]
    h = -2*(np.exp(1j*kx*np.pi) + np.exp(1j*ky*np.pi))
    return h

def Euler0Hamiltonian(k):
    g = gFunc(k)
    h = hFunc(k)
    H = np.zeros((3,3), dtype=np.complex)
    H[0,1] = g
    H[0,2] = np.conj(g)
    H[1,2] = h
    H = H + np.conj(H.T)
    return H

