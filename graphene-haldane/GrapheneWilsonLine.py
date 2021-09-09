
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:29:05 2021

@author: Georgia Nixon
"""
place = "Georgia Nixon"

import numpy as np
from numpy import sin, cos, pi, sqrt, exp
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy.linalg import expm
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/graphene-haldane')
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
sys.path.append("/Users/"+place+"/Code/MBQD/band-topology")
from hamiltonians import  PhiString, GetEvalsAndEvecs, getevalsandevecs
from Funcs import AbelianCalcWilsonLine, DifferenceLine
from GrapheneFuncs import (HaldaneHamiltonian, HaldaneHamiltonianPaulis,
                           CalculateBerryConnectMatrixGraphene)

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD-MBQD-WS-1/Notes/Topology Bloch Bands/"



size=16
params = {
        'legend.fontsize': size*0.75,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          }
mpl.rcParams.update(params)




phi = pi/4
t1=1
t2=1
M = 3.6#t2*3*sqrt(3) * sin(phi)
params = [phi, M, t1, t2]

#reciprocal lattice vectors
c1 = (2*pi/(3))*np.array([sqrt(3), 1])
c2 = (2*pi/(3))*np.array([-sqrt(3), 1])

#think u are qpoints?
dlt = 0.005
qpoints=200

wilsonline00abelian = np.zeros(qpoints, dtype=np.complex128)

#step for abelian version
#find u at first k
H = HaldaneHamiltonianPaulis(np.array([0,0]), params)
_, evecs = getevalsandevecs(H)
uInitial = evecs[:,0]

u10 = np.linspace(0,3,qpoints)
kline = np.outer(u10,c1)
dq = kline[1] - kline[0]
    
# go through possible end points for k
for i, kpoint in enumerate(kline):
    
    #do abeliean version,
    #find u at other k down the line
    H = HaldaneHamiltonianPaulis(kpoint, params)
    _, evecs = getevalsandevecs(H)
    uFinal = evecs[:,0]
    wilsonline00abelian[i] = np.dot(np.conj(uFinal), uInitial)
    
    

'''
Calculate Wilson Line - Abelian
'''

k0 = kline[0]
H = HaldaneHamiltonian(kline[0], params)
_, evecsInitial = GetEvalsAndEvecs(H)

wilsonLineAbelian = np.zeros([qpoints, 2, 2], dtype=np.complex128)
# go through possible end points for k
for i, kpoint in enumerate(kline):
    #Find evec at k point, calculate Wilson Line abelian method
    H =HaldaneHamiltonianPaulis(kpoint, params)
    _, evecsFinal = GetEvalsAndEvecs(H)
    wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsFinal, evecsInitial)


'''
Calculate Wilson Line - Non Abelian
'''
wilsonLineNonAbelian = np.zeros([qpoints, 2, 2], dtype=np.complex128)
currentArgument = np.zeros([2,2], dtype=np.complex128)

for i, kpoint in enumerate(kline):
    berryConnect = CalculateBerryConnectMatrixGraphene(kpoint, params)
    berryConnectAlongKLine =  1j*np.dot(berryConnect, dq)
    currentArgument = currentArgument + berryConnectAlongKLine
    wilsonLineNonAbelian[i] = expm(currentArgument)



m1 = 1
m2 = 1
multiplier = np.linspace(0,3,qpoints, endpoint=True)
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(multiplier, np.square(np.abs(wilsonLineNonAbelian[:,0,0])),   label=r'non abelian $ \exp [\Pi_{n=1}^{N} i \Delta_{\mathbf{q}} \cdot \mathbf{A}(\mathbf{q}_n)]$')
# ax.plot(multiplier, np.abs(wilsonline00abelian),   label=r'old abelian')
ax.plot(multiplier, np.square(np.abs(wilsonLineAbelian[:,0,0])), label=r"abelian $<u_{q_i}^0 | u_{q_f}^0>$")
ax.set_ylabel(r"$|W[0,0]|^2$")
ax.set_xlabel(r"Final quasimomentum (in units of $\vec{G}$ away from $\Gamma$ )")
ax.set_title(r"$t_1 = "+str(t1)+r", t_2 = "+str(t2) + r", M = "+str(M) + r", \phi = " + PhiString(phi) + r"$")
plt.legend()
plt.show()    


"""
Eigenvalue flow
""" 


evalsNonAbelian = np.empty((qpoints, 2), dtype=np.complex128)
evalsAbelian = np.empty((qpoints, 2), dtype=np.complex128)
for i in range(qpoints):
    evalsNA, _ = GetEvalsAndEvecs(wilsonLineNonAbelian[i,:,:])
    evalsA, _ = GetEvalsAndEvecs(wilsonLineAbelian[i,:,:])
    evalsNonAbelian[i] = evalsNA
    evalsAbelian[i] = evalsA
    

n1 = 0
fig, ax = plt.subplots(ncols = 2, nrows = 1, sharey = True, figsize=(24,9))
ax[0].plot(multiplier, np.angle(evalsNonAbelian[:,n1]), label="Non Abelian, eigenvalue " + str(n1))
ax[0].plot(multiplier, np.angle(evalsAbelian[:,n1]), label="Abelian, eigenvalue " + str(n1))
ax[1].plot(multiplier, np.imag(evalsNonAbelian[:,n1]), label="Non Abelian, eigenvalue " + str(n1))
ax[1].plot(multiplier, np.imag(evalsAbelian[:,n1]), label="Abelian, eigenvalue " + str(n1))
ax[0].set_title(r"real(eval)")
ax[1].set_title(r"imag(eval)")
plt.legend()
plt.suptitle("e'value flow")
plt.show()






#%%
 