# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:11:25 2021

@author: Georgia Nixon
"""

place = "Georgia Nixon"
import numpy as np
from numpy import sqrt, pi, sin
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/')
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/graphene-haldane')
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/extended-haldane')
from ExtendedHaldaneModel import  ExtendedHaldaneHamiltonianSpins, ExtendedHaldaneHamiltonianRashbaCoupling 
from ExtendedHaldaneModel import  HaldaneHamiltonian
# from GrapheneFuncs import  HaldaneHamiltonian
from Funcs import  BerryCurvature
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import eig


#%%

#params for ExtendedHaldane
phi = 0
M = 0
t1 = 1
t2 = 0.6
t3 = 1
# lambdaR = 0.3
params = [phi, M, t1, t2, t3]
# params = [phi, M, t1, t2, t3, lambdaR]


"""Compute Hamiltonian and graph at some k point"""
k = np.array([0.6,0.6])

H = ExtendedHaldaneHamiltonian(k, params)
U = np.dot(H, np.conj(H.T))

apply = [
         np.abs, 
         np.real, 
         np.imag]


hMax = np.max(np.stack((np.real(H), np.imag(H), np.abs(H))))
hMin = np.min(np.stack((np.real(H), np.imag(H), np.abs(H))))
bound = np.max((np.abs(hMax), np.abs(hMin)))

norm = mpl.colors.Normalize(vmin=-bound, vmax=bound)

sz = 20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
for n1, f in enumerate(apply):
    # pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr',  norm=norm)
    pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr', norm=norm)
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')
ax[0].set_ylabel('n', rotation=0, labelpad=10)
# cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(pcm)



#%%
""" Phase Diagram for varying t2 and t3"""

#fixed params
#Extended Haldane
# phi = 0
# M = 0.1
t1 = 1
t2 = 1/3
# t3 = 0.35
# lambdaR = 0.3


#Haldane
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

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)


# granularity of phase diagram
# nt2s = 3; nt3s=3
# chernnumbers = np.zeros((nt2s, nt3s), dtype=float)

# granularity of phase diagram
nphis = 30; nMs=30
chernnumbers0 = np.zeros((nphis, nMs), dtype=float)
chernnumbers1 = np.zeros((nphis, nMs), dtype=float)


# calculate chern num for various params
# for pn, t2 in enumerate(np.linspace(0, 1, nt2s, endpoint=True)):
#     for dn, t3 in enumerate(np.linspace(0, 1, nt3s, endpoint=True)):
for pn, phi in enumerate(np.linspace(0, 2*pi, nphis, endpoint=True)):
    for dn, M in enumerate(np.linspace(-3*sqrt(3)*t2, 3*sqrt(3)*t2, nMs, endpoint=True)):
        print(pn,dn)

        berrycurve0 = np.zeros([len(kx), len(kx)])
        berrycurve1 = np.zeros([len(kx), len(kx)])


        for xcnt in range(len(u10)):
            for ycnt in range(len(u10)):
                
                #pick momentum point in meshgrid
                k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
                
                #calculate Berry Curvature at this point
                bC0, lB0, uB0 = BerryCurvature(HaldaneHamiltonian,
                                               k, [phi, M, t1, t2, 0])
                bC1, lB1, lB2 = BerryCurvature(HaldaneHamiltonian,
                                               k, [phi, M, t1, t2, 0.35])

                berrycurve0[xcnt, ycnt] = bC0
                berrycurve1[xcnt, ycnt] = bC1



        chernnumbers0[pn,dn] = (1/2/pi)*np.sum(berrycurve0[:-1,:-1])*jacobian
        chernnumbers1[pn,dn] = (1/2/pi)*np.sum(berrycurve1[:-1,:-1])*jacobian

# plot chern num phase diagram for different params


cn = chernnumbers0
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(cn), axis=0)), cmap="RdBu", aspect="auto", 
                interpolation='none', 
                # extent=[0,1,0, 1], 
                extent=[0, 2*pi, -3*sqrt(3)*t2,3*sqrt(3)*t2], 
                norm = mpl.colors.Normalize(vmin=np.min(cn), vmax=np.max(cn)))
ax.set_title(r"Chern Number,"+"\n"+r"Extended Haldane model, "+ r"$t_1="+str(t1) + r",  t_3=0.0$")
# ax.set_xlabel(r"$t_2$")
ax.set_xlabel(r"$\varphi$")
# x_label_list = [r"$0$", r"$1$"]
# y_label_list = [r"$0$", r"$1$"]
x_label_list = [r"$0$", r"$\pi$", r"$2\pi$"]
y_label_list = [r"$-3\sqrt{3}$", r"$0$", r"$3 \sqrt{3}$"]
ax.set_xticks([0, pi, 2*pi])
ax.set_yticks([-3*sqrt(3)*t2,0,3*sqrt(3)*t2])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
# ax.set_ylabel(r"$t_3$",  rotation=0, labelpad=0)
ax.set_ylabel(r"$\frac{\Delta}{ t_2}$",  rotation=0, fontsize = 16, labelpad=0)
fig.colorbar(img)
# fig.suptitle(r"$t="+str(t1) + r" M = "+str(m)+r" \lambda_R = "+str(lambdaR)+r"$", y=1.05)
plt.show()

cn = chernnumbers1
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(cn), axis=0)), cmap="RdBu", aspect="auto", 
                interpolation='none', 
                # extent=[0,1,0, 1], 
                extent=[0, 2*pi, -3*sqrt(3)*t2,3*sqrt(3)*t2], 
                norm = mpl.colors.Normalize(vmin=np.min(cn), vmax=np.max(cn)))
ax.set_title(r"Chern Number,"+"\n"+r"Extended Haldane model, "+ r"$t_1="+str(t1) + r",  t_3=0.35, \lambda_R = $"+str(lambdaR))
# ax.set_xlabel(r"$t_2$")
ax.set_xlabel(r"$\varphi$")
# x_label_list = [r"$0$", r"$1$"]
# y_label_list = [r"$0$", r"$1$"]
x_label_list = [r"$0$", r"$\pi$", r"$2\pi$"]
y_label_list = [r"$-3\sqrt{3}$", r"$0$", r"$3 \sqrt{3}$"]
ax.set_xticks([0, pi, 2*pi])
ax.set_yticks([-3*sqrt(3)*t2,0,3*sqrt(3)*t2])
ax.set_xticklabels(x_label_list)
ax.set_yticklabels(y_label_list)
# ax.set_ylabel(r"$t_3$",  rotation=0, labelpad=0)
ax.set_ylabel(r"$\frac{\Delta}{ t_2}$",  rotation=0, fontsize = 16, labelpad=0)
fig.colorbar(img)
# fig.suptitle(r"$t="+str(t1) + r" M = "+str(m)+r" \lambda_R = "+str(lambdaR)+r"$", y=1.05)
plt.show()
#%%

"""
Berry Curvature
"""

#fixed params
#Extended Haldane
phi = pi/4
M = 0.6
t1 = 1
t2 = 1/3
t3 = 0
# lambdaR = 0.3


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

jacobian = dlt**2*(4*pi/3)**2*sin(pi/3)


berrycurve0 = np.zeros([len(kx), len(kx)])
berrycurve1 = np.zeros([len(kx), len(kx)])

lowerband0 = np.zeros([len(kx), len(kx)])
lowerband1 = np.zeros([len(kx), len(kx)])

upperband0 = np.zeros([len(kx), len(kx)])
upperband1 = np.zeros([len(kx), len(kx)])

for xcnt in range(len(u10)):
    for ycnt in range(len(u10)):
        
        #pick momentum point in meshgrid
        k = np.array([kx[xcnt, ycnt], ky[xcnt,ycnt]])
        
        #calculate Berry Curvature at this point
        
        bC0, lB0, uB0 = BerryCurvature(HaldaneHamiltonian, k, [phi, M,  t1, t2, 0])
        bC1, lB1, uB1 = BerryCurvature(HaldaneHamiltonian, k, [phi, M, t1, t2, 0.35])


#        berrycurve0[xcnt, ycnt] = bC0
        berrycurve1[xcnt, ycnt] = bC1
        
        lowerband0[xcnt, ycnt] = lB0
        lowerband1[xcnt, ycnt] = lB1
        
        upperband0[xcnt, ycnt] = uB0
        upperband1[xcnt, ycnt] = uB1


chernnumber0 = (1/2/pi)*np.sum(berrycurve0[:-1,:-1])*jacobian
chernnumber1 = (1/2/pi)*np.sum(berrycurve1[:-1,:-1])*jacobian

#%%


cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)
normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)


"""#1"""


plt.plot(0,0)

lowerband = lowerband0
upperband = upperband0
berrycurve = berrycurve0
chernnumber = chernnumber0


#berry curve
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(25, -140)
ax.plot_surface(kx/pi, ky/pi, berrycurve, cmap=cmap)
ax.set_title(r"$\Omega_{-}$" + " where total chern number="+str(np.round(np.real(chernnumber), 6)))
ax.set_xlabel(r'$k_x/\pi$', labelpad=5)
ax.set_ylabel(r'$k_y/\pi$', labelpad=5)
plt.show()  

# berry curve 2
fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), cmap="RdBu",
                aspect="auto", interpolation='none')
ax.set_title(r"$\Omega_{-}$")
fig.colorbar(img)
plt.show()




"""Bands """

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







"""2"""

plt.plot(0,0)


lowerband = lowerband1
upperband = upperband1
berrycurve = berrycurve1
chernnumber = chernnumber1

cmapstring = 'twilight'
cmap = mpl.cm.get_cmap(cmapstring)
normaliser = mpl.colors.Normalize(vmin=-3, vmax=3)


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
ax.view_init(25, -140)
ax.plot_surface(kx/pi, ky/pi, berrycurve, cmap=cmap)
ax.set_title(r"$\Omega_{-}$" + " where total chern number="+str(np.round(np.real(chernnumber), 6)))
ax.set_xlabel(r'$k_x/\pi$', labelpad=5)
ax.set_ylabel(r'$k_y/\pi$', labelpad=5)
plt.show()  


fig, ax = plt.subplots()
img = ax.imshow(np.real(np.flip(np.transpose(berrycurve), axis=0)), cmap="RdBu",
                aspect="auto", interpolation='none')
ax.set_title(r"$\Omega_{-}$")
fig.colorbar(img)
plt.show()


"""Bands """

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
ax.set_title('upperband')
ax.set_xlabel(r'$k_x/\pi$')
ax.set_ylabel(r'$k_y/\pi$')
fig.suptitle(r"$t="+str(t1)+r" \quad \Delta ="+str(np.round(M,2)) + r" \quad t_2 = "
              +str(t2)+r" \quad \phi = 0"+
              r"\quad \frac{\Delta}{ t_2 }\frac{1}{3 \sqrt{3}} = "+str(np.round(M/t2/(3*sqrt(3)),2))+r"$", y=0.99)
# plt.savefig(sh + "BerryCurvature3-lowerband.pdf", format="pdf")
plt.show()  




