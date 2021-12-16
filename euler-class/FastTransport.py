# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:03:26 2021

@author: Georgia Nixon
"""
place = "Georgia Nixon"
import numpy as np
from numpy import cos, sin, exp, pi, tan
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from EulerClass2Hamiltonian import  Euler2Hamiltonian, GetEvalsAndEvecsEuler, AlignGaugeBetweenVecs
from EulerClass4Hamiltonian import Euler4Hamiltonian

# from hamiltonians import GetEvalsAndEvecs
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import norm
import matplotlib as mpl
from scipy.integrate import solve_ivp



def F_Euler(t, psi, gammaPoint, Fvec):
    k = gammaPoint+ Fvec*t
    H2 = Euler2Hamiltonian(k)
    print(H2)
    return -1j*np.dot(H2, psi)


gammaPoint = np.array([0,0.5])
F = 10000
Fvec = F*np.array([1,0]) 

#get evecs at gamma point
H0 = Euler2Hamiltonian(gammaPoint)
_, evecs = np.linalg.eigh(H0)
u0 = evecs[:,0]

tol=1e-18
sol = solve_ivp(lambda t, psi : F_Euler(t, psi, gammaPoint, Fvec), 
        t_span=(0,2/F), y0=u0, rtol=tol, 
        atol=tol,
        method='RK45')
# sol=sol.y

func = np.real
plt.plot(sol.t, func(sol.y[0,:]), label="ground band")
plt.plot(sol.t, func(sol.y[1,:]), label="1 band")
plt.plot(sol.t, func(sol.y[2,:]), label="2 band")
plt.legend()
plt.show()

overlap = np.array([np.vdot(sol.y[:,i], u0) for i in range(np.shape(sol.y)[1])])



