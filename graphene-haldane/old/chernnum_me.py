# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:02:18 2021

@author: Georgia Nixon
"""


import numpy as np
from numpy import sin, cos, pi, sqrt, exp
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib as mpl

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

def getevalsandevecs(HF):
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
    return evals, evecs


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


#neareste neighbor vectors
a1 = np.array([1/2, sqrt(3)/2])
a2 = np.array([-1, 0])
a3 = np.array([1/2, -sqrt(3)/2])

#second nearest neighbor vectors
b1 = a2-a3
#b1 = a3-a2
b2 = a3-a1
#b2 = a1-a3
b3 = a1-a2
#b3 = a2-a1

def hamiltonian(k, phi, M, t1, t2):
    cosasum = cos(np.dot(a1,k)) + cos(np.dot(a2,k))+cos(np.dot(a3,k))
    sinasum = sin(np.dot(a1,k))+ sin(np.dot(a2, k)) + sin(np.dot(a3, k))
    cosbsum = cos(np.dot(b1, k))+cos(np.dot(b2,k))+cos(np.dot(b3,k))
    sinbsum = sin(np.dot(b1,k))+ sin(np.dot(b2, k)) + sin(np.dot(b3, k))
    
    H = np.zeros([2,2], dtype=np.complex128)
    H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
    H[1,1] = -M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum
    H[0,1]= t1*(cosasum-1j*sinasum);
    H[1,0]=np.conj(H[0,1]);
    return H



phi = 3*pi/2
t1=1
t2=0.1
M = t2*3*sqrt(3) * sin(phi)-0.1


c1 = np.array([4*pi/3, 0])
c2 = np.array([2*pi/3, 2*pi/sqrt(3)])

#think u are qpoints?
dlt = 0.005
qpoints=201
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10

u1, u2 = np.meshgrid(u10, u20)

kx = u1*c1[0] + u2*c2[0]
ky = u1*c1[1] + u2*c2[1]

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi

lowerband = np.zeros([qpoints, qpoints], dtype=np.complex128)
upperband = np.zeros([qpoints, qpoints], dtype=np.complex128)
berrycurve = np.zeros([qpoints, qpoints], dtype=np.complex128)
berryconnect = np.zeros([qpoints, qpoints], dtype=np.complex128)

h = 0.0001;

for xcnt in range(qpoints):
    for ycnt in range(qpoints):
        k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
    
        H = hamiltonian(k, phi, M, t1, t2)
        
        d0,v0 = getevalsandevecs(H)
        
        #first eigenvector
        u0=v0[:,0]
        lowerband[xcnt,ycnt] = d0[0]
        upperband[xcnt,ycnt] = d0[1] 
        
        #dx direction
        kxx = k + np.array([h,0])
        H = hamiltonian(kxx, phi, M, t1, t2)
        dx,vx = getevalsandevecs(H)
        ux = vx[:,0]
        
        #dy direction
        kyy = k+np.array([0,h])
        H = hamiltonian(kyy, phi, M, t1, t2)
        dy,vy = getevalsandevecs(H)
        uy=vy[:,0]

        xder = (ux-u0)/h
        yder = (uy-u0)/h
        
        berrycurve[xcnt,ycnt] = -1j*(np.dot(np.conj(xder), yder) - np.dot(np.conj(yder), xder))
        berrycurve[xcnt,ycnt] = berrycurve[xcnt,ycnt]



sumchern = np.sum(berrycurve[:-1,:-1])*jacobian



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(15, 45)
ax.plot_surface(kx/pi, ky/pi, np.real(berrycurve), cmap=cmap)
ax.set_xticks([ -1,0, 1])
ax.set_xticklabels([ -1,0, r"$1$"])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([-1, 0, r"$1$"])
ax.set_title('Berry curvature where total chern number='+str(np.round(np.real(sumchern), 6)))
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
plt.show()       


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(15, 45)
ax.plot_surface(kx/pi, ky/pi, np.real(lowerband), cmap=cmap)
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels([1, 0, r"$1$"])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([-1, 0, r"$1$"])
ax.set_title('lowerband')
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
plt.show()    

#%%



normaliser = mpl.colors.Normalize(vmin=-110, vmax=110)

fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), norm=normaliser,cmap="RdBu", aspect="auto", interpolation='none', extent=[-pi,pi,-pi,pi])
ax.set_title(r"$\Omega_{-}$")
ax.set_xlabel(r"$k_x$")
label_list = [r'$-\pi$', r"$0$", r"$\pi$"]
ax.set_xticks([-pi,0,pi])
ax.set_yticks([-pi,0,pi])
ax.set_xticklabels(label_list)
ax.set_yticklabels(label_list)
ax.set_ylabel(r"$k_y$", rotation=0)
fig.colorbar(img)
#fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "
#             +str(t2)+r" \quad \phi = "+phistring(phi)+r"\quad \Delta / t_2 = "+str(np.round(delta/t2, 2))+r"$", y=0.99)

#plt.savefig(sh + "BerryCurvature1.pdf", format="pdf")

plt.show()