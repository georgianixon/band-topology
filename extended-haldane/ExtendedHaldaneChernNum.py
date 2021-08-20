# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:11:25 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import sqrt, pi, sin
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/extended-haldane')
from ExtendedHaldaneModel import  ExtendedHaldaneHamiltonian, ExtendedHaldaneHamiltonian0
from ExtendedHaldaneModel import  HaldaneHamiltonian, BerryCurvature
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import eig

#params for ExtendedHaldane
# t1 = 1
# t2 = 0.6
# t3 = 0.9
# m = 1
# lambdaR = 0.3
# params = [m, lambdaR, t1, t2, t3]

# Params for Haldane
phi = pi/2
t1 = 1
t2 = 0.6
m = 1
params = [phi, m, t1, t2]

k = np.array([0.6,0.6])

H = HaldaneHamiltonian(k, params)
U = np.dot(H, np.conj(H.T))

apply = [
         np.abs, 
         np.real, np.imag]
# norm = mpl.colors.Normalize(vmin=-2, vmax=2)
sz = 20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
for n1, f in enumerate(apply):
    # pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr',  norm=norm)
    pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr')
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')
ax[0].set_ylabel('n', rotation=0, labelpad=10)
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(pcm)



#%%
""" Phase Diagram """

#fixed params
#Extended Haldane
M = 1
lambdaR = 0.3
t1 = 1
# Haldane
# t1=1
# t2=0.1

#reciprocal lattice vectors
r1 = (2*pi/3)*np.array([sqrt(3), 1])
r2 = (2*pi/3)*np.array([sqrt(3), -1])

#create meshgrid of momentum points
dlt = 0.005 #separation between momentum points
qpoints=201
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10
u1, u2 = np.meshgrid(u10, u20)
kx = u1*r1[0] + u2*r2[0]
ky = u1*r1[1] + u2*r2[1]

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi


# granularity of phase diagram
nt2s = 5; nt3s=5
chernnumbers = np.zeros((nt2s, nt3s), dtype=float)
    

#calculate chern num for various params
for pn, t2 in enumerate(np.linspace(0, 1, nt2s, endpoint=True)):
    for dn, t3 in enumerate(np.linspace(0, 1, nt3s, endpoint=True)):
        print(pn,dn)

        berrycurve = np.zeros([len(kx), len(kx)])
        
        for xcnt in range(len(u10)):
            for ycnt in range(len(u10)):
                
                #pick momentum point in meshgrid
                k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
                
                #calculate Berry Curvature at this point
                # params = [phi, M, t1, t2]
                params = [M, lambdaR, t1, t2, t3]
                berrycurve[xcnt, ycnt] = BerryCurvature(ExtendedHaldaneHamiltonian0, k, params)

        chernnumbers[pn,dn] = np.sum(berrycurve[:-1,:-1])*jacobian

# plot chern num phase diagram for different params
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(chernnumbers), axis=0)), cmap="RdBu", aspect="auto", 
                interpolation='none', extent=[0,1,0, 1], norm = mpl.colors.Normalize(vmin=-6, vmax=0))
ax.set_title(r"Chern Number")
ax.set_xlabel(r"$t_2$")
x_label_list = [r"$0$", r"$1$"]
y_label_list = [r"$0$", r"$1$"]
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
ax.set_ylabel(r"$t_3$",  rotation=0, fontsize = 23, labelpad=0)
fig.colorbar(img)
# fig.suptitle(r"$t="+str(t1) + r" M = "+str(m)+r" \lambda_R = "+str(lambdaR)+r"$", y=1.05)
plt.show()

#%%

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)
normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(35, -140)
ax.plot_surface(kx/pi, ky/pi, berrycurve, cmap=cmap)
#ax.set_xticks([ -1,0, 1])
#ax.set_xticklabels([ -1,0, r"$1$"])
#ax.set_yticks([-1, 0, 1])
#ax.set_yticklabels([-1, 0, r"$1$"])
# ax.set_title(r"$\Omega_{-}$" + " where total chern number="+str(np.round(np.real(sumchern), 6)))
ax.set_xlabel(r'$k_x/\pi$', labelpad=5)
ax.set_ylabel(r'$k_y/\pi$', labelpad=5)
plt.show()  


fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), cmap="RdBu",
                aspect="auto", interpolation='none')
ax.set_title(r"$\Omega_{-}$")
fig.colorbar(img)
plt.show()


