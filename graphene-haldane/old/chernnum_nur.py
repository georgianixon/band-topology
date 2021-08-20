# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:15:48 2021

@author: Georgia Nixon
"""

"""Nurs script equiv"""


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

phi = 3*pi/2
t1=1
t2=0.1
M = t2*3*sqrt(3) * sin(phi)-0.1

a1 = np.array([1/2, sqrt(3)/2])
a2 = np.array([-1, 0])
a3 = np.array([1/2, -sqrt(3)/2])

b1 = a2-a3
b2 = a3-a1
b3 = a1-a2

c1 = np.array([4*pi/3, 0])
c2 = np.array([2*pi/3, 2*pi/sqrt(3)])

#think u are qpoints?
dlt = 0.005
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10

u1, u2 = np.meshgrid(u10, u20)

kx = u1*c1[0] + u2*c2[0]
ky = u1*c1[1] + u2*c2[1]

lowerband = np.zeros([len(kx), len(kx)], dtype=np.complex128)
upperband = np.zeros([len(kx), len(kx)], dtype=np.complex128)
berrycurve = np.zeros([len(kx), len(kx)], dtype=np.complex128)

h = 0.0001;

for xcnt in range(len(u10)):
    for ycnt in range(len(u20)):
        k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
        
        cosasum = cos(np.dot(a1,k)) + cos(np.dot(a2,k))+cos(np.dot(a3,k))
        sinasum = sin(np.dot(a1,k))+ sin(np.dot(a2, k)) + sin(np.dot(a3, k))
        cosbsum = cos(np.dot(b1, k))+cos(np.dot(b2,k))+cos(np.dot(b3,k))
        sinbsum = sin(np.dot(b1,k))+ sin(np.dot(b2, k)) + sin(np.dot(b3, k))
    
        H = np.zeros([2,2], dtype=np.complex128)
        H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
        H[1,1] = -M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum
        H[0,1]= t1*(cosasum-1j*sinasum);
        H[1,0]=np.conj(H[0,1]);
        
        if xcnt == 139 and ycnt == 139:
            print(np.dot(np.conj(H.T),H))
        
        d0,v0 = getevalsandevecs(H)
        
        #first eigenvector
        u0=v0[:,0]
        u0 = np.conj(u0[1])/np.abs(u0[1])*u0 #makes second part real and normalises
        lowerband[xcnt,ycnt] = d0[0]
        upperband[xcnt,ycnt] = d0[1] 
        
        #x direction
        
        kxx = k + np.array([h,0])
        
        cosasum = cos(np.dot(a1,kxx)) + cos(np.dot(a2,kxx)) + cos(np.dot(a3,kxx))
        sinasum = sin(np.dot(a1,kxx)) + sin(np.dot(a2,kxx)) + sin(np.dot(a3,kxx))
        cosbsum = cos(np.dot(b1,kxx)) + cos(np.dot(b2,kxx)) + cos(np.dot(b3,kxx))
        sinbsum = sin(np.dot(b1,kxx)) + sin(np.dot(b2,kxx)) + sin(np.dot(b3,kxx))
        
        H = np.zeros([2,2], dtype=np.complex128)
        H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
        H[1,1] = -M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum
        H[0,1]= t1*(cosasum-1j*sinasum);
        H[1,0]=np.conj(H[0,1]);
        
        dx,vx = getevalsandevecs(H)
        
        ux = vx[:,0]
        ux = np.conj(ux[1])/np.abs(ux[1])*ux
        
        #y direction
        
        kyy = k+np.array([0,h])
        cosasum = cos(np.dot(a1,kyy)) + cos(np.dot(a2,kyy)) + cos(np.dot(a3,kyy))
        sinasum = sin(np.dot(a1,kyy)) + sin(np.dot(a2,kyy)) + sin(np.dot(a3,kyy))
        cosbsum = cos(np.dot(b1,kyy)) + cos(np.dot(b2,kyy)) + cos(np.dot(b3,kyy))
        sinbsum = sin(np.dot(b1,kyy)) + sin(np.dot(b2,kyy)) + sin(np.dot(b3,kyy))
        
        H = np.zeros([2,2], dtype=np.complex128)
        H[0,0] = M + 2*t2*cos(phi)*cosbsum - 2*t2*sin(phi)*sinbsum
        H[1,1] = -M+2*t2*cos(phi)*cosbsum+2*t2*sin(phi)*sinbsum
        H[0,1]= t1*(cosasum-1j*sinasum);
        H[1,0]=np.conj(H[0,1]);
        
        dy,vy = getevalsandevecs(H)
        
        uy=vy[:,0]
        uy=np.conj(uy[1])/np.abs(uy[1])*uy
        xder = (ux-u0)/h
        yder = (uy-u0)/h
        
        berrycurve[xcnt,ycnt] = -1j*(np.dot(np.conj(xder), yder) - np.dot(np.conj(yder), xder))




sumchern = np.sum(berrycurve[:-1,:-1])*dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(15, 0)
ax.plot_surface(kx/pi, ky/pi, np.real(berrycurve))
ax.set_xticks([ 0, 1, 2])
ax.set_xticklabels([ 0, r"$1$", 2])
ax.set_yticks([ 0, 1])
ax.set_yticklabels([ 0, r"$1$"])
ax.set_title('Berry curvature where total chern number='+str(np.round(np.real(sumchern), 8)))
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
plt.show()       


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(5, 45)
ax.plot_surface(kx/pi, ky/pi, np.real(lowerband), cmap=cmap)
plt.show()    

#%%


def phistring(phi):
    if phi == 0:
        return "0"
    else:
        return  r'\pi /' + str(int(1/(phi/pi)))

fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), cmap="RdBu", aspect="auto", interpolation='none', extent=[-pi,pi,-pi,pi])
ax.set_title(r"$\Omega_{-}$")
ax.set_xlabel(r"$k_x$")
label_list = [r'$-\pi$', r"$0$", r"$\pi$"]
ax.set_xticks([-pi,0,pi])
ax.set_yticks([-pi,0,pi])
ax.set_xticklabels(label_list)
ax.set_yticklabels(label_list)
ax.set_ylabel(r"$k_y$", rotation=0)
fig.colorbar(img)
fig.suptitle(r"$t="+str(t1)+r" \quad M ="+str(M) + r" \quad t_2 = "
             +str(t2)+r" \quad \phi = "+phistring(phi)+r"\quad M / t_2 = "+str(np.round(M/t2, 2))+r"$", y=0.99)

#plt.savefig(sh + "BerryCurvature1.pdf", format="pdf")

plt.show()


