# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:23:03 2021

@author: Georgia Nixon
"""
import numpy as np
from numpy import sqrt, exp, pi
from numpy.linalg import eig
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits import mplot3d


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
 

t3 = np.array([
 [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025],
 [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
 [0.167664, -0.0425061, -0.0025, -0.0425061, 0.167664],
 [0.0833399, -0.0025, 0.0375061, -0.0025, 0.0833399],
 [-0.0025, -0.0883399, -0.172664, -0.0883399, -0.0025]
])

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

t8 = np.array([
 [0, -0.187857, -0.433013, -0.187857, 0],
 [-0.187857, 0, 0.30825, 0, -0.187857],
 [-0.433013, 0.30825, 0, 0.30825, -0.433013],
 [-0.187857, 0, 0.30825, 0, -0.187857],
 [0, -0.187857, -0.433013, -0.187857, 0]
        ])


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
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t1[i,j] for i in range(Nnx) for j in range(Nny)])
def h3N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t3[i,j] for i in range(Nnx) for j in range(Nny)])
def h4N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t4[i,j] for i in range(Nnx) for j in range(Nny)])
def h6N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t6[i,j] for i in range(Nnx) for j in range(Nny)])
def h8N(kx, ky):
    return np.sum([exp(1j*pi*(kx*lx[i] + ky*ly[j]))*t8[i,j] for i in range(Nnx) for j in range(Nny)])


def EulerHamiltonian(kx,ky):
    hjk = np.array([0, h1N(kx,ky), h3N(kx,ky), h4N(kx,ky), h6N(kx,ky), h8N(kx,ky)])
    gellManns = np.array([GM0, GM1, GM3, GM4, GM6, GM8])
    HFn = np.array([hjk[i]*gellManns[i] for i in range(len(hjk))])
    return np.sum(HFn, axis=0)

place = "Georgia Nixon"
sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"

size=25
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



#%%



kmin = -1
kmax = 3
qpoints = 201 # easier for meshgrid when this is odd
K1 = np.linspace(kmin, kmax, qpoints, endpoint=True)
K2 = np.linspace(kmin, kmax, qpoints, endpoint=True)

eiglist = np.zeros((qpoints,qpoints,3)) # for three bands

for xi, qx in enumerate(K1):
    for yi, qy in enumerate(K2):
        eigs, evecs = GetEvalsAndEvecs(EulerHamiltonian(qx,qy))
        eiglist[xi,yi] = eigs
        
        

#%%

from mpl_toolkits.axes_grid1 import make_axes_locatable
normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)
cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

X, Y = np.meshgrid(K1, K2)

sz = 15
fig, ax = plt.subplots(figsize=(15,10))
ax = plt.axes(projection='3d')
ax.view_init(0, 225)
groundband = ax.contour3D(X, Y, np.real(eiglist[:,:,0]), 50,cmap=cmap, norm=normaliser)
firstband = ax.contour3D(X, Y, np.real(eiglist[:,:,1]), 50,cmap=cmap, norm=normaliser)
secondband = ax.contour3D(X, Y, np.real(eiglist[:,:,2]), 50,cmap=cmap, norm=normaliser)
# fig.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser))
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zlabel("E")
ax.set_zlabel("E")
ax.set_xlabel(r"$k_x$", labelpad=25)
ax.set_ylabel(r"$k_y$", labelpad=25)
ax.set_title(r"Euler Hamiltonian $\xi = 2$ bandstructure",y=0.9)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(plt.cm.ScalarMappable(cmap=cmapstring, norm=normaliser), fraction=0.026, pad=0.04)
plt.savefig(sh + "Euler2BS.pdf", format="pdf")
plt.show()

#%%
"""
Find dirac points
"""
eigdiff = eiglist[:,:,1] - eiglist[:,:,0]
eigdiff = np.abs(eigdiff)
#align eigdiff with a usual plot, x alog bottom, y along LHS.
# =============================================================================
# eigdiff = [[{kx=min, ky=min},... , {kx=min,ky=max}],
#               ...
#             {kx=max, ky=min},... , {kx=max, ky=max}]]
# but, to plot accuratly in imshow with kx along horizontal axis, ky along vertical axis, we want
# np.flip(eigdiff.T, axis=0)
#         = [[{kx=min, ky=max},... , {kx=max,ky=max}],
#               ...
#             {kx=min, ky=min},... , {kx=max, ky=min}]]
# =============================================================================
eigdiff =  np.flip(eigdiff.T, axis=0)


# find min 4 values (dirac points)?
idx = np.argsort(eigdiff, axis=None)
idx = idx[:120]
verticalEl_upToDown, horizontalEl_leftToRight = np.unravel_index(idx, np.shape(eigdiff))
verticalEl_downToUp = len(eigdiff[:,0])-1-verticalEl_upToDown
diracPoints = list(zip( horizontalEl_leftToRight, verticalEl_upToDown))

# norm = mpl.colors.Normalize(vmin=eigdiff.min(), vmax=eigdiff.max())
norm=mpl.colors.LogNorm(vmin=eigdiff.min(), vmax=eigdiff.max())

sz = 15
fig, ax = plt.subplots(figsize=(sz,sz/2))
pos = plt.imshow(eigdiff, cmap='viridis', norm=norm)
ax.set_xticks([0, (qpoints-1)/2, qpoints-1])
ax.set_yticks([0, (qpoints-1)/2, qpoints-1])
ax.set_xticklabels([kmin, int((kmin+kmax)/2), kmax])
ax.set_yticklabels([kmax, int((kmin+kmax)/2), kmin])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)
for xd, yd in diracPoints:
    circ = mpl.patches.Circle((xd, yd), 2, fill=0, edgecolor='1')
    ax.add_patch(circ)
fig.colorbar(pos)
plt.savefig(sh + "EigDif-1,3Log120LowestPoints.pdf", format="pdf")
plt.show()

#%%



lst = np.array([[10, 12, 8], [50, 7, 15]])
idx = np.argsort(np.log(lst), axis=None)
idx = idx[:4]
verticalEl_uptodown, horizontalEl_lefttoright = np.unravel_index(idx, np.shape(lst))
diracPoints = list(zip(verticalEl_uptodown, horizontalEl_lefttoright))
print(lst)
print(diracPoints)


