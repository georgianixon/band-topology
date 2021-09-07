# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:53:49 2021

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
sys.path.append("/Users/"+place+"/Code/MBQD/band-topology/graphene-haldane")
from hamiltonians import GetEvalsAndEvecs, PhiString, getevalsandevecs
from GrapheneFuncs import HaldaneHamiltonian, HaldaneHamiltonianNur, BerryCurvature, HaldaneHamiltonianPaulis

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)
# normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)

sh = "/Users/Georgia Nixon/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"


size=20
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
"""print Hamiltonian, sometimes handy"""

phi=pi/7;
t1=3;
t2=0.9;
M=0.5#t2*3*sqrt(3)*sin(phi)-0.1;
params = [phi, M, t1, t2]


k = np.array([0.6,0.6])

HN = HaldaneHamiltonianNur(k, params)
HM = HaldaneHamiltonian(k, params)
HP = HaldaneHamiltonianPaulis(k, params)
# U = np.dot(H, np.conj(H.T))

apply = [
         np.abs, 
         np.real, np.imag]


hMax = np.max(np.stack((np.real(HN), np.imag(HN), np.abs(HN), 
                        np.real(HM), np.imag(HM), np.abs(HM),
                        np.real(HP), np.imag(HP), np.abs(HP))))
hMin = np.min(np.stack((np.real(HN), np.imag(HN), np.abs(HN), 
                        np.real(HM), np.imag(HM), np.abs(HM),
                        np.real(HP), np.imag(HP), np.abs(HP))))


for H in [HN, HM, HP]:
    sz = 20
    fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                           figsize=(sz,sz/2))

    norm = mpl.colors.Normalize(vmin=hMin, vmax=hMax)
    for n1, f in enumerate(apply):
        pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr',  norm=norm)
        ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
          labeltop=False)  
        ax[n1].set_xlabel('m')
    ax[0].set_ylabel('n', rotation=0, labelpad=10)
    # cax = plt.axes([1.03, 0.1, 0.03, 0.8])
    fig.colorbar(pcm)
    plt.show()



#%%
"""Berry Curvature and Bands"""


phi=3*pi/2;
t1=1;
t2=0.1;
M=0.8#t2*3*sqrt(3)*sin(phi)-0.1;
params = [phi, M, t1, t2]


#reciprocal lattice vectors
r1 = (2*pi/(3))*np.array([1, sqrt(3)])
r2 = (2*pi/(3))*np.array([1, -sqrt(3)])



#think u are qpoints?
dlt = 0.005
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10
qpoints = len(u10)
u1, u2 = np.meshgrid(u10, u20)
kx = u1*r1[0] + u2*r2[0]
ky = u1*r1[1] + u2*r2[1]

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)

berrycurve = np.zeros([qpoints, qpoints], dtype=np.complex128)
lowerband = np.zeros([qpoints, qpoints], dtype=np.complex128)
upperband = np.zeros([qpoints, qpoints], dtype=np.complex128)


for xcnt in range(len(u10)):
    for ycnt in range(len(u10)):
        
        #pick momentum point in meshgrid
        k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
        
        bC, lB, uB = BerryCurvature(HaldaneHamiltonian, k, params)
        berrycurve[xcnt, ycnt] = bC
        lowerband[xcnt, ycnt] = lB
        upperband[xcnt, ycnt] = uB

sumchern = (1/2/pi)*np.sum(berrycurve[:-1,:-1])*jacobian


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(35, -140)
ax.plot_surface(kx/pi, ky/pi, np.real(berrycurve), cmap=cmap)
#ax.set_xticks([ -1,0, 1])
#ax.set_xticklabels([ -1,0, r"$1$"])
#ax.set_yticks([-1, 0, 1])
#ax.set_yticklabels([-1, 0, r"$1$"])
ax.set_zlim([-5, 15])
ax.set_title(r"$\Omega_{-}$" + " where total chern number="+str(np.round(np.real(sumchern), 6)))
ax.set_xlabel(r'$k_x/\pi$', labelpad=5)
ax.set_ylabel(r'$k_y/\pi$', labelpad=5)
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(np.round(M,2)) + r" \quad t_2 = "
             +str(t2)+r" \quad \phi = 0"+
             r"\quad \frac{\Delta}{ t_2 }\frac{1}{3 \sqrt{3}} = "+str(np.round(M/t2/(3*sqrt(3)),2))+r"$", 
             y=1.05)
# plt.savefig(sh + ".pdf", format="pdf")
plt.show()       



# normaliser = mpl.colors.Normalize(vmin=-110, vmax=110)
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), 
                cmap="RdBu", aspect="auto", #norm=normaliser,
                interpolation='none', extent=[-pi,pi,-pi,pi])
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
plt.show()


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(35, -140)
ax.plot_surface(kx/pi, ky/pi, np.real(lowerband), cmap=cmap)#, norm=normaliser)
#ax.set_xticks([-1, 0, 1])
#ax.set_xticklabels([1, 0, r"$1$"])
#ax.set_yticks([-1, 0, 1])
#ax.set_yticklabels([-1, 0, r"$1$"])
ax.set_title('lowerband')
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(np.round(M,2)) + r" \quad t_2 = "
              +str(t2)+r" \quad \phi = 0"+
              r"\quad \frac{\Delta}{ t_2 }\frac{1}{3 \sqrt{3}} = "+str(np.round(M/t2/(3*sqrt(3)),2))+r"$", y=0.99)
# plt.savefig(sh + "BerryCurvature3-lowerband.pdf", format="pdf")
plt.show()  


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(35, -140)
ax.plot_surface(kx/pi, ky/pi, np.real(upperband), cmap=cmap)#, norm=normaliser)
#ax.set_xticks([-1, 0, 1])
#ax.set_xticklabels([1, 0, r"$1$"])
#ax.set_yticks([-1, 0, 1])
#ax.set_yticklabels([-1, 0, r"$1$"])
ax.set_title('lowerband')
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(np.round(M,2)) + r" \quad t_2 = "
              +str(t2)+r" \quad \phi = 0"+
              r"\quad \frac{\Delta}{ t_2 }\frac{1}{3 \sqrt{3}} = "+str(np.round(M/t2/(3*sqrt(3)),2))+r"$", y=0.99)
# plt.savefig(sh + "BerryCurvature3-lowerband.pdf", format="pdf")
plt.show()  




#%%
""" Phase Diagram """

#fixed params
t1=1
t2=0.1


#reciprocal lattice vectors
r1 = (2*pi/(3))*np.array([1, sqrt(3)])
r2 = (2*pi/(3))*np.array([1, -sqrt(3)])


#create meshgrid of momentum points
dlt = 0.005 #separation between momentum points
u10 = np.linspace(0, 1, int(1/dlt + 1), endpoint=True)
u20=u10
u1, u2 = np.meshgrid(u10, u20)
kx = u1*r1[0] + u2*r2[0]
ky = u1*r1[1] + u2*r2[1]

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)/2/pi

# granularity of phase diagram
nphis = 5; nMs=5
chernnumbers = np.zeros((nphis, nMs), dtype=float)
    

#calculate chern num for various params

for pn, phi in enumerate(np.linspace(0, 2*pi, nphis, endpoint=True)):
    for dn, M in enumerate(np.linspace(-3*sqrt(3)*t2, 3*sqrt(3)*t2, nMs, endpoint=True)):
        print(pn,dn)

        berrycurve = np.zeros([len(kx), len(kx)])
        
        for xcnt in range(len(u10)):
            for ycnt in range(len(u10)):
                
                #pick momentum point in meshgrid
                k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
                
                #calculate Berry Curvature at this point
                # params = [phi, M, t1, t2]
                params = [phi, M,  t1, t2,]
                bC, _, _ = BerryCurvature(HaldaneHamiltonian, k, params)
                berrycurve[xcnt, ycnt] = bC

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
ax.set_ylabel(r"$\frac{\Delta}{ t_2}$",  rotation=0, labelpad=0)
fig.colorbar(img)
fig.suptitle(r"$t="+str(t1) + r" \quad t_2 = "
             +str(t2)+r"$", y=1.05)
#plt.savefig(sh + "chern_number.pdf", format="pdf")
plt.show()





#%%

