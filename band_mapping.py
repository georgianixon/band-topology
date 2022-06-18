# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:47:12 2022

@author: Georgia Nixon
"""

import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd
from math import floor
import seaborn as sns
import sys
from scipy.linalg import eigh 
place = "Georgia Nixon"
sys.path.append("/Users/"+place+"/Code/MBQD/floquet-simulations/src")
from hamiltonians import  GetEvalsAndEvecsGen

from scipy.integrate import solve_ivp

def Plot(size):    
    
    sns.set(style="darkgrid")
    sns.set(rc={'axes.facecolor':'0.96'})
    params = {
                'legend.fontsize': size*0.9,
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size*0.9,
              'ytick.labelsize': size*0.9,
              'font.size': size,
              'font.family': 'STIXGeneral',
    #          'axes.titlepad': 25,
              'mathtext.fontset': 'stix',
              
              # 'axes.facecolor': 'white',
              'axes.edgecolor': 'white',
              'axes.grid': True,
              'grid.alpha': 1,
              # 'grid.color': "0.9"
              "text.usetex": True
              }
    
    mpl.rcParams.update(params)
    mpl.rcParams["text.latex.preamble"] = mpl.rcParams["text.latex.preamble"] + r'\usepackage{xfrac}'
    
    # CB91_Blue = 'darkblue'#'#2CBDFE'
    # CB91_Green = '#47DBCD'
    # CB91_Pink = '#F3A0F2'
    # CB91_Purple = '#9D2EC5'
    # CB91_Violet = '#661D98'
    # CB91_Amber = '#F5B14C'
    # red = "#FC4445"
    # newred = "#FF1053"
    
    # color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
    #                CB91_Purple,
    #                 # CB91_Violet,
    #                 'dodgerblue',
    #                 'slategrey', newred]
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
    
Plot(12)


def FloatToStringSave(a):
    if type(a) != int and type(a) != np.int32:
        if a.is_integer():
            a = int(a)
    return str(a).replace(".", "p")


def V_func(V0, t_zero_potential, t):
    V = V0 - V0/t_zero_potential*t
    if V <0:
        return 0
    else:
        return V

def Ham_1(params, t_zero_potential,  q, lmax, t):
    """
    Hamiltonian in the plane wave basis
    """
    V0 = params
    H = np.diag([(2*i+q)**2 + V_func(V0, t_zero_potential, t)/2 for i in range(-lmax, lmax+1)],0)         
    H = H + np.diag([-V_func(V0, t_zero_potential, t)/4]*(2*lmax ), -1) + np.diag([-V_func(V0, t_zero_potential, t)/4]*(2*lmax), 1)
    return H

def Ham_2(params, t_zero_potential, q, lmax, t):
    """
    g*k_A = k_B : ie g is factor between wavelengths. 
    VA amplitude of first wavelength
    VB potential of second
    Ramp them both down in same time
    """
    VA = params[0]
    VB = params[1]
    g = params[2]
    H = np.diag([(2*i+q)**2 for i in range(-lmax, lmax+1)],0)         
    H = H + (np.diag([V_func(VA, t_zero_potential, t)/4]*(2*lmax), -1) 
             + np.diag([V_func(VA, t_zero_potential, t)/4]*(2*lmax), 1)
             + np.diag([V_func(VB, t_zero_potential, t)/4]*(2*lmax - g+1), g)
             + np.diag([V_func(VB, t_zero_potential, t)/4]*(2*lmax - g+1), -g))
    
    return H
    
    

def evals_df(Ham, params, lmax):
    df = pd.DataFrame(columns=['q']+['b'+str(i) for i in range(2*lmax + 1)])

    qlist = np.linspace(-1,1,201, endpoint=True)
    for i, q in enumerate(qlist):
        evals, _ = eig(Ham(params, 1,  q, lmax, t=0))
        evals = np.sort(evals)
        df.loc[i] = np.concatenate([np.array([q]), evals])
    return df

def F_Ham(t, psi, Ham, params, t_zero_potential, q, lmax):
    H =Ham(params, t_zero_potential,  q, lmax, t)
    return -1j*np.dot(H, psi)


def SolveSchrodinger(Ham, params, t_zero_potential, q, lmax, rtol, tspan, nTimesteps, psi0):
    """
    Solve Schrodinger Equation for oscilating Hamiltonian
    """
    # points to calculate the matter wave at
    t_eval = np.linspace(tspan[0], tspan[1], nTimesteps+1, endpoint=True)

    sol = solve_ivp(lambda t,psi: F_Ham(t, psi, Ham, params, t_zero_potential, q, lmax),
            t_span=tspan, y0=psi0, rtol=rtol, 
            atol=rtol, t_eval=t_eval,
            method='RK45')
    sol=sol.y
    return sol



#%%

"""
Bandmapping - simple 1D
"""
import matplotlib as mpl
V0 = 10
q0 = -0.5
# typically, (2*jmax+1) *\[HBar]\[Omega] should be chosen larger than the calculated trap depth.
# Lower trap depth, smaller required jMax. For example, jMax=2 will be too small for trap depth 10Er
# but it is fine for 5Er.
lmax = 5
band = 2

Ham = Ham_1
#find evecs at initial time t=0
evals_t0, evecs_t0 = eigh(Ham(V0, 1,  q0, lmax, t=0))

full_evals = evals_df(Ham, V0, lmax)
min_band_gap_below = 10#np.min(full_evals["b"+str(band)]) - np.max(full_evals["b"+str(band-1)])
min_band_gap_above = np.min(full_evals["b"+str(band+1)]) - np.max(full_evals["b"+str(band)])
min_surrounding_band_gap = np.min([min_band_gap_above, min_band_gap_below])

bandgap_above = evals_t0[band+1] - evals_t0[band]
bandgap_below = 10#evals_t0[band] - evals_t0[band-1]
bandgap = np.min([bandgap_above, bandgap_below])
adiabatic_criterion = 1/min_surrounding_band_gap# 1/min_surrounding_band_gap
#for adiabatic regime
t_zero_potential = adiabatic_criterion*50

# for fast regime
t_zero_potential = 1/bandgap/2

cmap = mpl.cm.get_cmap('hsv')
# fig, ax = plt.subplots()
# for i in range(2*lmax + 1):
#     ax.plot(range(2*lmax+1), np.abs(evecs_t0[:,i])**2, c=cmap(i/(2*lmax+1)), label=str(i))
# ax.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(range(2*lmax+1), evals_t0)
# plt.show()




psi0 = evecs_t0[:,band]
psi0 = psi0.astype(np.complex128)


rtol = 1e-11
tspan = (0,t_zero_potential)
nTimesteps = 1000

sol = SolveSchrodinger(Ham, V0, t_zero_potential, q0, lmax, rtol, tspan, nTimesteps, psi0)

groundstate_f = sol[:,-1]

#defone x axis, real momentum

#this a bit hacky
if q0 == 0:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j+1] = (np.abs(sol[:,i*100])**2)[j]
if q0==0.5:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j+2] = (np.abs(sol[:,i*100])**2)[j]
if q0==-0.5:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j] = (np.abs(sol[:,i*100])**2)[j]
        
notesLoc = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/Euler Class/"
fig, ax = plt.subplots()
for i in range(11):
    ax.plot(k,pk[i], c=cmap(i/(11)),  label=r"$\psi(t_{"+str(i)+r"})$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\psi(k,t)$", rotation=0, labelpad=20)
ax.legend(loc="upper right")
ax.set_ylim([-0.05, 1.05])
ax.set_xticks(np.linspace(-10,10,21))
plt.text(-10, .3, r"$\mathrm{band}="+str(band)+r"$"
         +"\n"+r"$q0="+str(q0)+r"$"
         +"\n"+r'$1/ \mathrm{bandgap}_{q0}='+str(round(1/bandgap,2))+r"$"
         +"\n"+r'$1/ \mathrm{bandgap} ='+str(round(1/min_surrounding_band_gap,2))+r"$"
         +"\n"+r"$t_{10}="+str(round(t_zero_potential,3))+r"$")
fig.savefig(notesLoc+"Bandmapping,1D,V0="+FloatToStringSave(V0)+",q0="+FloatToStringSave(q0)+",band="+str(band)+",NonAdiabatic.png", format='png', bbox_inches='tight')
plt.show()


#%%

"""
Bandmapping - sublattices
"""

VA = 10
VB = 5
q0 = 0.5
g = 2
params = [VA, VB, g]
# typically, (2*jmax+1) *\[HBar]\[Omega] should be chosen larger than the calculated trap depth.
# Lower trap depth, smaller required jMax. For example, jMax=2 will be too small for trap depth 10Er
# but it is fine for 5Er.
lmax = 5
band = 1
Ham = Ham_2
#find evecs at initial time t=0
evals_t0, evecs_t0 = eigh(Ham(params, 1,  q0, lmax, t=0))

full_evals = evals_df(Ham, params, lmax)
min_band_gap_below = 10#np.min(full_evals["b"+str(band)]) - np.max(full_evals["b"+str(band-1)])
min_band_gap_above = np.min(full_evals["b"+str(band+1)]) - np.max(full_evals["b"+str(band)])
min_surrounding_band_gap = np.min([min_band_gap_above, min_band_gap_below])

bandgap_above = evals_t0[band+1] - evals_t0[band]
bandgap_below = 10#evals_t0[band] - evals_t0[band-1]
bandgap = np.min([bandgap_above, bandgap_below])
adiabatic_criterion = 1/min_surrounding_band_gap# 1/min_surrounding_band_gap
#for adiabatic regime
t_zero_potential = adiabatic_criterion*50

# for fast regime
# t_zero_potential = 1/bandgap/2

cmap = mpl.cm.get_cmap('hsv')
# fig, ax = plt.subplots()
# for i in range(2*lmax + 1):
#     ax.plot(range(2*lmax+1), np.abs(evecs_t0[:,i])**2, c=cmap(i/(2*lmax+1)), label=str(i))
# ax.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(range(2*lmax+1), evals_t0)
# plt.show()




psi0 = evecs_t0[:,band]
psi0 = psi0.astype(np.complex128)


rtol = 1e-11
tspan = (0,t_zero_potential)
nTimesteps = 1000

sol = SolveSchrodinger(Ham, params, t_zero_potential, q0, lmax, rtol, tspan, nTimesteps, psi0)

groundstate_f = sol[:,-1]

#defone x axis, real momentum

#this a bit hacky
if q0 == 0:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j+1] = (np.abs(sol[:,i*100])**2)[j]
if q0==0.5:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j+2] = (np.abs(sol[:,i*100])**2)[j]
if q0==-0.5:
    k = np.linspace(-10.5, 10.5, 43)
    pk = np.zeros((11,43))
    for i in range(11):
        for j in range(2*lmax+1):
            pk[i,4*j] = (np.abs(sol[:,i*100])**2)[j]
        
notesLoc = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/Euler Class/"
fig, ax = plt.subplots()
for i in range(11):
    ax.plot(k,pk[i], c=cmap(i/(11)),  label=r"$\psi(t_{"+str(i)+r"})$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$\psi(k,t)$", rotation=0, labelpad=20)
ax.legend(loc="upper right")
ax.set_ylim([-0.05, 1.05])
ax.set_xticks(np.linspace(-10,10,21))
plt.text(-10, .3, r"$\mathrm{band}="+str(band)+r"$"
         +"\n"+r"$q0="+str(q0)+r"$"
         +"\n"+r'$1/ \mathrm{bandgap}_{q0}='+str(round(1/bandgap,2))+r"$"
         +"\n"+r'$1/ \mathrm{bandgap} ='+str(round(1/min_surrounding_band_gap,2))+r"$"
         +"\n"+r"$t_{10}="+str(round(t_zero_potential,3))+r"$")
# fig.savefig(notesLoc+"Bandmapping,1D,V0="+FloatToStringSave(V0)+",q0="+FloatToStringSave(q0)+",band="+str(band)+",NonAdiabatic.png", format='png', bbox_inches='tight')
plt.show()


