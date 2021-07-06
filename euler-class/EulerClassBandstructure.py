# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:24:35 2021

@author: Georgia
"""

import numpy as np
from numpy import sqrt, exp, pi, cos, sin
import matplotlib.pyplot as plt 
import matplotlib as mpl
from numpy.linalg import eig
from mpl_toolkits import mplot3d

place = "Georgia Nixon"
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
from EulerClassHamiltonian import  EulerHamiltonian, GetEvalsAndEvecs

def CreateCircleLineIntVals(r, points, centre=[0,0]):
    CircleLine =  [(int(np.round(cos(2*pi/points*x)*r+centre[0])),int(np.round(sin(2*pi/points*x)*r+centre[1]))) for x in range(0,int(np.ceil(points+1)))]
    #get rid of duplicates
    CircleLine = list(dict.fromkeys(CircleLine) )
    return CircleLine

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints)
    kline = np.around(kline)
    kline = kline.astype(int)
    # kline = list(dict.fromkeys(kline) )
    return kline

# def CreateCircleLineIntVals(r, points, centre=[0,0]):
#     CircleLine =  [(int(np.round(cos(2*pi/points*x)*r+centre[0])),int(np.round(sin(2*pi/points*x)*r+centre[1]))) for x in range(0,int(np.ceil(points+1)))]
#     #get rid of duplicates
#     CircleLine = list(dict.fromkeys(CircleLine) )
#     return CircleLine

# def CreateRectLineIntVals(r, points, centre=[0,0]):
#     CircleLine =  [(int(np.round(cos(2*pi/points*x)*r+centre[0])),int(np.round(sin(2*pi/points*x)*r+centre[1]))) for x in range(0,int(np.ceil(points+1)))]
#     #get rid of duplicates
#     CircleLine = list(dict.fromkeys(CircleLine) )
#     return CircleLine



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


import time
start = time.time()


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
        
 
end = time.time()
print("Time consumed in working: ",end - start)       

#%%

"""
Calculate Bandstructure
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable
normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)
cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)

X, Y = np.meshgrid(K1, K2)

sz = 15
fig, ax = plt.subplots(figsize=(10,10))
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
plt.savefig(sh + "EulerBS.pdf", format="pdf")
plt.show()

#%%
"""
Find dirac points
"""
eigdiff = eiglist[:,:,1] - eiglist[:,:,0]
eigdiff = np.abs(eigdiff)
#align eigdiff with a usual plot, x along bottom, y along LHS.
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


# find min values (dirac points)?
idx = np.argsort(eigdiff, axis=None)
mins = np.sort(eigdiff, axis=None)
idx = idx[:10]
verticalEl_upToDown, horizontalEl_leftToRight = np.unravel_index(idx, np.shape(eigdiff))
verticalEl_downToUp = len(eigdiff[:,0])-1-verticalEl_upToDown
diracPoints = list(zip(horizontalEl_leftToRight, verticalEl_upToDown))

# norm = mpl.colors.Normalize(vmin=eigdiff.min(), vmax=eigdiff.max())
norm=mpl.colors.LogNorm(vmin=eigdiff.min(), vmax=eigdiff.max())

sz = 15
fig, ax = plt.subplots(figsize=(sz,sz/2))
pos = plt.imshow(eigdiff, cmap='viridis', norm=norm)
ax.set_xticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4,  qpoints-1])
ax.set_yticks([0, (qpoints-1)/4, (qpoints-1)/2, 3*(qpoints-1)/4, qpoints-1])
ax.set_xticklabels([kmin, int(round((kmin+kmax)/4)), int((kmin+kmax)/2), int(round(3*(kmin+kmax)/4)), kmax])
ax.set_yticklabels([kmax, int(round(3*(kmin+kmax)/4)), int((kmin+kmax)/2), int(round((kmin+kmax)/4)), kmin])
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$", rotation=0, labelpad=15)

# circle lowest points
# for i, (xd, yd) in enumerate(diracPoints):
#     circ = mpl.patches.Circle((xd, yd), 2, fill=0, edgecolor="1")
#     ax.add_patch(circ)

# add path part
npoints = 100
kline0 = CreateLinearLine(3*qpoints/8, qpoints/4, 3*qpoints/8, 3*qpoints/4,  npoints)
kline1 = CreateLinearLine(3*qpoints/8, 3*qpoints/4, 5*qpoints/8, 3*qpoints/4, npoints)
kline2 = CreateLinearLine(5*qpoints/8, 3*qpoints/4, 5*qpoints/8, qpoints/4, npoints)
kline3 = CreateLinearLine(5*qpoints/8, qpoints/4, 3*qpoints/8, qpoints/4, npoints)
kline =np.vstack((kline0,kline1,kline2, kline3))


# # klineCircle = CreateCircleLineIntVals(qpoints/8, 2*pi*qpoints/8, centre = [qpoints/4, qpoints/4])

for i, (xd, yd) in enumerate(kline):
    if i == 0 or i == len(kline):
        ec = '0'
    else:
        ec = '1'
    circ = mpl.patches.Circle((xd, -yd+qpoints), 2, fill=0, edgecolor=ec)
    ax.add_patch(circ)

fig.colorbar(pos)
plt.savefig(sh + "SquarePath.pdf", format="pdf")
plt.show()

#%%

x = np.linspace(0, 1, num=100)
plt.plot(x, np.arcsin(x))
plt.xlabel(r"$\sin ( \theta / 2)$")
plt.show()

theta = np.linspace(0, 2*pi, 100)
plt.plot(theta, np.tan(theta/2))
