# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:32:02 2021

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
import sys
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import GetEvalsAndEvecs

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

# def getevalsandevecs(HF):
#     #order by evals, also order corresponding evecs
#     evals, evecs = eig(HF)
#     idx = np.real(evals).argsort()
#     evals = evals[idx]
#     evecs = evecs[:,idx]
    
#     #make first element of evecs real and positive
#     for vec in range(np.size(HF[0])):
#         phi = np.angle(evecs[0,vec])
#         evecs[:,vec] = exp(-1j*phi)*evecs[:,vec]
        
#         evecs[:,vec] = np.conj(evecs[0,vec])/np.abs(evecs[0,vec])*evecs[:,vec]
        
#         #nurs normalisation
# #        evecs[:,vec] = np.conj(evecs[1,vec])/np.abs(evecs[1,vec])*evecs[:,vec]
#     return evals, evecs


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



def HaldaneHamiltonian(k, phi, M, t1, t2):

    #nearest neighbor vectors
    a1 = np.array([1, 0])
    a2 = np.array([-1/2, sqrt(3)/2])
    a3 = np.array([-1/2, -sqrt(3)/2])
    
    #second nearest neighbor vectors -> relative chirality important
    b1 = np.array([3/2, sqrt(3)/2])
    b2 = np.array([0,-sqrt(3)])
    b3 = np.array([-3/2, sqrt(3)/2])

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

def BerryCurvature(Hamiltonian, k, phi, M, t1, t2):
    
    h = 0.0001
    
    H = Hamiltonian(k, phi, M, t1, t2)
    
    d0,v0 = GetEvalsAndEvecs(H)
                
    #first eigenvector
    u0=v0[:,0]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = Hamiltonian(kxx, phi, M, t1, t2)
    dx,vx = GetEvalsAndEvecs(H)
    ux = vx[:,0]
    
    #dy direction
    kyy = k+np.array([0,h])
    H = Hamiltonian(kyy, phi, M, t1, t2)
    dy,vy = GetEvalsAndEvecs(H)
    uy=vy[:,0]

    xder = (ux-u0)/h
    yder = (uy-u0)/h
    
    berrycurve = 1j*(np.dot(np.conj(xder), yder) - np.dot(np.conj(yder), xder))
    berrycurve1 = np.imag(np.dot(np.conj(xder), yder))
    
    return berrycurve


#reciprocal lattice vectors
c1 = (2*pi/(3))*np.array([1, sqrt(3)])
c2 = (2*pi/(3))*np.array([1, -sqrt(3)])

#%%
# hopping strengths
t1=1
t2=0.1



#create meshgrid of momentum points
dlt = 0.005 #separation between momentum points
qpoints=201
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10
u1, u2 = np.meshgrid(u10, u20)
kx = u1*c1[0] + u2*c2[0]
ky = u1*c1[1] + u2*c2[1]


jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi


# granularity of phase diagram
nphis = 5; ndeltas=5
chernnumbers = np.zeros((nphis, ndeltas), dtype=float)

    

#calculate chern num for various params
for pn, phi in enumerate(np.linspace(0, 2*pi, nphis, endpoint=True)):
    for dn, M in enumerate(np.linspace(-3*sqrt(3)*t2, 3*sqrt(3)*t2, ndeltas, endpoint=True)):
        print(pn,dn)

        berrycurve = np.zeros([len(kx), len(kx)], dtype=np.complex128)
        
        for xcnt in range(len(u10)):
            for ycnt in range(len(u10)):
                
                #pick momentum point in meshgrid
                k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
                
                #calculate Berry Curvature at this point
                berrycurve[xcnt,ycnt] = BerryCurvature(HaldaneHamiltonian, k, phi, M, t1, t2)

        chernnumbers[pn,dn] = np.sum(berrycurve[:-1,:-1])*jacobian

# plot chern num phase diagram for different params
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(chernnumbers), axis=0)), cmap="RdBu", aspect="auto", 
                interpolation='none', extent=[0,2*pi,-3*sqrt(3), 3*sqrt(3)])
ax.set_title(r"Chern Number")
ax.set_xlabel(r"$\varphi$")
x_label_list = [r"$0$", r"$\pi$", r"$2\pi$"]
y_label_list = [r"$-3\sqrt{3}$", r"$0$", r"$3 \sqrt{3}$"]
ax.set_xticks([0,pi,2*pi])
ax.set_yticks([-3*sqrt(3), 0, 3*sqrt(3)])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
ax.set_ylabel(r"$\frac{\Delta}{ t_2}$",  rotation=0, fontsize = 23, labelpad=0)
fig.colorbar(img)
fig.suptitle(r"$t="+str(t1) + r" \quad t_2 = "
             +str(t2)+r"$", y=1.05)
#plt.savefig(sh + "chern_number.pdf", format="pdf")
plt.show()