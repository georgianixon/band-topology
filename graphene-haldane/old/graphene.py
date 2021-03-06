# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:33:14 2021

@author: Georgia Nixon
"""


import numpy as np
from numpy import sin, cos, pi, sqrt, exp
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib as mpl


sh = "/Users/Georgia Nixon/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

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


PauliX = np.array([[0,1], [1,0]])
PauliY = 1j*np.array([[0,-1], [1,0]])
PauliZ = np.array([[1,0], [0,-1]])
Identity = np.array([[1,0],[0,1]])


A = np.array([[1,0], [-0.5, sqrt(3)/2], [-0.5,-sqrt(3)/2]])
B = np.array([[0,sqrt(3)], [-1.5,-sqrt(3)/2], [1.5,-sqrt(3)/2]])
         
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


def HGraphene(t1,t2,phi, Delta, K):
    """
    Calculate Graphene matrix for variables tunneling (t1), NNN tunnelling (t2),
    energy offset (M), for particular quasimomentum (K)
    """
#    IdentityPart = Identity*2*t2*cos(phi)*np.sum([cos(np.dot(K, B[i])) for i in range(3)])
    PauliXPart = PauliX*t1*np.sum([cos(np.dot(K, A[i])) for i in range(3)])
    PauliYPart = -PauliY*t1*np.sum([sin(np.dot(K, A[i])) for i in range(3)])
    PauliZPart = PauliZ*(Delta + 2*t2*sin(phi)*np.sum([sin(np.dot(K, B[i])) 
                                                       for i in range(3)]))
    
    return PauliXPart + PauliYPart + PauliZPart


def phistring(phi):
    if phi == 0:
        return "0"
    else:
        return  r'\pi /' + str(int(1/(phi/pi)))
#%%

normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)
cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

t1 = 1
delta = 0.4
t2 = -0.075
phi = pi/2
qpoints = 1000
    
qlist = np.linspace(-pi,pi, qpoints, endpoint=True)

eiglist = np.zeros((qpoints,qpoints,2), dtype=np.complex128) # for both bands
eigveclist = np.zeros((qpoints, qpoints, 2), dtype=np.complex128) # for band n

for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        eigs, evecs = getevalsandevecs(HGraphene(t1, t2, phi, delta, np.array([qx, qy])))
#            eigs, evecs = getevalsandevecs(HGraphene(t1, t2, phi,  delta, np.array([qx, qy])))
        eiglist[xi,yi] = eigs
        eigveclist[xi,yi] = evecs[:,0] #only taking ground band

eiglist = np.real(eiglist)

normed = np.linalg.norm(eigveclist, axis=2)
#%%
"""Plot bands in 3D"""
X, Y = np.meshgrid(qlist, qlist)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(5, 45)
ax.set_zlabel("E")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels([r"$-\pi$", 0, r"$\pi$"])
groundband = ax.contour3D(X, Y, eiglist[:,:,0], 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, eiglist[:,:,1], 50,cmap=cmap, norm=normaliser)
ax.set_zlabel("E")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser))
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "+
             str(t2)+r" \quad \phi = "+phistring(phi)+r"$")
#plt.savefig(sh + "graphene_sideangle_m_t2.pdf", format="pdf")
plt.show()

#%%
"""Plot bands in 2D"""

fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8,4))
img = ax[0].imshow(np.real(np.flip(np.transpose(eiglist[:,:,0]), axis=0)), cmap=cmap, aspect="auto",
                   norm=normaliser,interpolation='none', extent=[-pi,pi,-pi,pi])
img1 = ax[1].imshow(np.real(np.flip(np.transpose(eiglist[:,:,1]), axis=0)), cmap=cmap, aspect="auto",
                    norm=normaliser, interpolation='none', extent=[-pi,pi,-pi,pi])
ax[0].set_title(r"Lowest Band")
ax[0].set_xlabel(r"$k_x$")
label_list = [r'$-\pi$', r"$0$", r"$\pi$"]
ax[0].set_xticks([-pi,0,pi])
ax[0].set_yticks([-pi,0,pi])
ax[0].set_xticklabels(label_list)
ax[0].set_yticklabels(label_list)
ax[0].set_ylabel(r"$k_y$")
ax[1].set_title(r"First Excited Band")
ax[1].set_xlabel(r"$k_x$")
ax[1].set_xticks([-pi,0,pi])
ax[1].set_yticks([-pi,0,pi])
ax[1].set_xticklabels(label_list)
ax[1].set_yticklabels(label_list)
fig.colorbar(img, cax = plt.axes([1.03, 0.155, 0.02, 0.66]))
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "+
             str(t2)+r" \quad \phi = "+phistring(phi)+r"$")
#plt.savefig(sh + "graphene_bands_m_-t2.pdf", format="pdf")
plt.show()



#%% 

"""Plot Berry curvature"""
dq = qlist[1] - qlist[0]

psi_A = eigveclist[:,:,0]
psi_B = eigveclist[:,:,1]

psi_A_dqx, psi_A_dqy = np.gradient(psi_A)/dq
psi_B_dqx, psi_B_dqy = np.gradient(psi_B)/dq

BerryCurvature = 2*np.imag(psi_A_dqx*psi_A_dqy + psi_B_dqx*psi_B_dqy)
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(BerryCurvature), axis=0)), cmap="RdBu", aspect="auto", interpolation='none', extent=[-pi,pi,-pi,pi])
ax.set_title(r"$\Omega_{-}$")
ax.set_xlabel(r"$k_x$")
label_list = [r'$-\pi$', r"$0$", r"$\pi$"]
ax.set_xticks([-pi,0,pi])
ax.set_yticks([-pi,0,pi])
ax.set_xticklabels(label_list)
ax.set_yticklabels(label_list)
ax.set_ylabel(r"$k_y$", rotation=0)
fig.colorbar(img)
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(delta) + r" \quad t_2 = "
             +str(t2)+r" \quad \phi = "+phistring(phi)+r"\quad \Delta / t_2 = "+str(np.round(delta/t2, 2))+r"$", y=0.99)

#plt.savefig(sh + "BerryCurvature1.pdf", format="pdf")

plt.show()

#%% rectangle


BZ = np.zeros((qpoints,qpoints))
qlist = np.linspace(-pi,pi, qpoints, endpoint=True)
for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        if (qx <= 2*pi/3 and qx >= -2*pi/3 and qy <= 4*pi/sqrt(3)/3 - 1/sqrt(3)*qx
            and qy>= -4*pi/3/sqrt(3) + 1/sqrt(3)*qx and qy>= -4*pi/3/sqrt(3) - 1/sqrt(3)*qx
            and qy <= 4*pi/3/sqrt(3) + 1/sqrt(3)*qx):        
            BZ[xi,yi] = 1

Chern = np.sum(BZ*BerryCurvature)
print(Chern)

#%%

""" Calculate chern num at different points """

t1 = 1
t2 = 0.1

qpoints = 200
qlist = np.linspace(-pi,pi, qpoints, endpoint=True)
dq = qlist[1] - qlist[0]
BZ = np.zeros((qpoints,qpoints))

for xi, qx in enumerate(qlist):
    for yi, qy in enumerate(qlist):
        if (qx <= 2*pi/3 and qx >= -2*pi/3 and qy <= 4*pi/sqrt(3)/3 - 1/sqrt(3)*qx
            and qy>= -4*pi/3/sqrt(3) + 1/sqrt(3)*qx and qy>= -4*pi/3/sqrt(3) - 1/sqrt(3)*qx
            and qy <= 4*pi/3/sqrt(3) + 1/sqrt(3)*qx):        
            BZ[xi,yi] = 1

nphis = 30; ndeltas=30
ChernNumbers = np.zeros((nphis, ndeltas), dtype=float)
            
for pn, phi in enumerate(np.linspace(0, 2*pi, nphis, endpoint=True)):
    for dn, delta in enumerate(np.linspace(-3*sqrt(3)*t2, 3*sqrt(3)*t2, ndeltas, endpoint=True)):
        print(pn,dn)
                
        #eiglist = np.zeros((qpoints,qpoints,2), dtype=np.complex128) # for both bands
        eigveclist = np.zeros((qpoints, qpoints, 2), dtype=np.complex128) # for band n
        
        for xi, qx in enumerate(qlist):
            for yi, qy in enumerate(qlist):
                eigs, evecs = getevalsandevecs(HGraphene(t1, t2, phi, delta, np.array([qx, qy])))
        #        eiglist[xi,yi] = eigs
                eigveclist[xi,yi] = evecs[:,0] #only taking ground band
                
        #eiglist = np.real(eiglist)
        
        psi_A = eigveclist[:,:,0]
        psi_B = eigveclist[:,:,1]
        
        psi_A_dqx, psi_A_dqy = np.gradient(psi_A)/dq
        psi_B_dqx, psi_B_dqy = np.gradient(psi_B)/dq
        
        BerryCurvature = 2*np.imag(psi_A_dqx*psi_A_dqy + psi_B_dqx*psi_B_dqy)
        
        ChernNumbers[pn,dn] = np.sum(BZ*BerryCurvature)


#%%
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(ChernNumbers), axis=0)), cmap="RdBu", aspect="auto", 
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

#%%



#%%
import plotly.graph_objects as go
# Download data set from plotly repo
pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
x, y, z = pts.T

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])
print(fig)
fig.show()