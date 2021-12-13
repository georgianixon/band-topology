# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:13:29 2021

@author: Georgia
"""
place = "Georgia"
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from EulerClass2Hamiltonian import  Euler2Hamiltonian, AlignGaugeBetweenVecs
# from EulerClass4Hamiltonian import  Euler4Hamiltonian


def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  np.array([[np.cos(x)*r+centre[0],np.sin(x)*r+centre[1]] for x in np.linspace(0, 2*np.pi, points, endpoint=True)])
    return CircleLine

def AlignGaugeBetweenVecs(vec1, vec2):
    """
    Make <vec1|vec2> real and positive by shifting overall phase of vec2
    Return phase shifted vec2
    """
    #overlap between vec1 and vec2
    c = np.vdot(vec1, vec2)
    #find conj phase of overlap
    conjPhase = np.conj(c)/np.abs(c)
    #remove phase, so overlap is real and positive
    vec2 = conjPhase*vec2
    
    # make sure vec1 is in the right gauge, to 20dp
    c = np.dot(np.conj(vec1), vec2)
    
    #try again if still not within..
    if round(np.imag(c), 30)!=0:
        conjPhase = np.conj(c)/np.abs(c)
        vec2 = conjPhase*vec2
        c = np.dot(np.conj(vec1), vec2)
        assert(round(np.imag(c), 20)==0)
    
    return vec2


def SetGaugeByBand(evecsInitial, evecsFinal):
    """overlap band gauges independently"""
    
    evec0I = evecsInitial[:,0]
    evec1I = evecsInitial[:,1]
    evec2I = evecsInitial[:,2]
    
    evecs0F = evecsFinal[:,0]
    evecs1F = evecsFinal[:,1]
    evecs2F = evecsFinal[:,2]
    
    evecs0F = AlignGaugeBetweenVecs(evec0I, evecs0F)
    evecs1F = AlignGaugeBetweenVecs(evec1I, evecs1F)
    evecs2F = AlignGaugeBetweenVecs(evec2I, evecs2F)
    
    evecsFinal[:,0] = evecs0F
    evecsFinal[:,1] = evecs1F
    evecsFinal[:,2] = evecs2F
    
    return evecsFinal



# orbital positions in the unit cell? Think this should be same for all Wilson Line calcs because we just change 
# things in reciprocal space for Wilson Line

		
nk = 101
nk_y = 101
k_start = [0,0.5] # z stays at 0.5, x stays at 0
num_bands = 3


nk = 101
radius=1.7
centre = [0,0]

# create circle line
k_path = CreateCircleLine(radius, nk, centre = centre)


debug = True
tol = 0.1

eigs = np.empty((len(k_path), num_bands))


P_tot = np.zeros((num_bands, num_bands),dtype=np.complex128)
U_0 = np.zeros((num_bands, num_bands),dtype=np.scomplex128)
U_current = np.zeros((num_bands, num_bands),dtype=np.complex128)


#find Hamiltonian at initial k point
H = Euler2Hamiltonian(k_path[0])
vals_0,vecs_0 = np.linalg.eigh(H)
vecs_0[np.abs(vecs_0)<1e-15] = 0
eigs[0] = vals_0


#store original evecs
for band in range(num_bands):
    U_0[:,band] = vecs_0[:,band]

P_tot =  np.copy(U_0)
vecsPrev = vecs_0

#go through integration path, each k represented by current_k
for int_k_passed, current_k in enumerate(k_path):
    
    if int_k_passed != 0: # if we not at the end or the beginning?
        U_current = np.zeros((num_bands, num_bands),dtype=np.complex128)
    
        #diagonalise Hamiltonian at this point
        H = Euler2Hamiltonian(current_k)
        vals,vecs = np.linalg.eigh(H)
        vecs[np.abs(vecs)<1e-15] = 0
        # idx_sorted = np.argsort(np.real(vals))
        # vecs = vecs[:,idx_sorted]
        # vals = vals[idx_sorted]
        
        vecs = SetGaugeByBand(vecsPrev, vecs)
        #store evecs
        for band in range(num_bands):
            U_current[:,band] = vecs[:,band]
        
        """ see what each UU^dagger is"""
        matrix = np.matmul(U_current, np.conjugate(U_current.T))
        matrix = np.round(matrix, 15)
        print(matrix)
    
        
        vecsPrev = vecs
				
       
            
            
        if debug:
            if np.sum(np.abs(np.matmul(np.conjugate(U_current).T, U_current))-np.eye(num_bands, num_bands))>1e-10:
                print("ERROR: Eigenvectors are not orthogonal!")
				
				
				# P_tot =  U * U^dagger * P_tot
        # Think this is making the Wilson Line Matrix
        P_tot = np.matmul(np.conjugate(U_current).T, P_tot)
        
        vals,vecs = np.linalg.eig(P_tot)
        eigs[int_k_passed, :] = np.sort(np.imag(np.log(vals)))
        
        if int_k_passed != nk-1:
            
            #for next one need to multiply again
            P_tot = np.matmul(U_current, P_tot)

    if debug:
        for val in vals:
            if np.abs(np.abs(val)-1)>tol:
                print("WARNING: Wilson eigenvalue %.3f is abs(%.3f) at k=[%.3f,%.3f] (number %i)!"%(np.imag(np.log(val)),np.abs(val),
                                                                                                    current_k[0],current_k[1],int_k_passed))


sz = 4
fig, ax = plt.subplots(figsize=(sz*1.3,sz))
multiplier = np.linspace(0,2*np.pi,nk)
ax.plot(multiplier, eigs[:,0], '.', label=r'$1^{\mathrm{st}}$ eigenvalue')
ax.plot(multiplier, eigs[:,1], '.', label=r'$2^{\mathrm{nd}}$ eigenvalue')
ax.plot(multiplier, eigs[:,2], '.', label=r'$3^{\mathrm{rd}}$ eigenvalue')
ax.set_xticks((0, np.pi/2,  np.pi, 3*np.pi/2,  2*np.pi))
ax.set_xticklabels((r'$0$', r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ))
ax.set_title( r'$\mathrm{Im} ( \mathrm{log}(e))$') 
plt.legend()
ax.set_yticks((-np.pi, -1,  0, 1,  np.pi))
ax.set_yticklabels((r'$-\pi$', "-1", r"$0$", "$1$", r"$\pi$" ))
fig.text(0.5, 0.03, 'final q-momentum', ha='center')
# plt.savefig(sh+ "WilsonLineEulerCircleEvals,r="+NumToString(radius)+",c=(1,1).pdf", format="pdf")
plt.show()


