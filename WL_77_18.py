import numpy as np
import matplotlib.pyplot as plt
from cmath import phase
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
plt.rcParams["text.usetex"] = True
from scipy.linalg import orth
import sys


####Model parameters for MSG77.18

t1 = 1.0
t2 = 1.0
t3 = 1.0

#lambdas
lm1 = 0.5*np.exp(1j*np.pi/5.0)
lm2 = 0.5*np.exp(1j*np.pi/5.0)

#rhos
rho1 = -1
rho2 = -2.0/5
epsilon_z = 0.0

a1 = 2*np.pi*np.array([1,0,0]) #Slightly unconvential definition, sorry
a2 = 2*np.pi*np.array([0,1,0])
a3 = 2*np.pi*np.array([0,0,1])


b1 = np.array([1,0,0])
b2 = np.array([0,1,0])
b3 = np.array([0,0,1])


# real space vecs
delta1 = 0.5*a1-0.5*a2+0.5*a3
delta2 = 0.5*a1+0.5*a2+0.5*a3
delta3 = -0.5*a1+0.5*a2+0.5*a3
delta4 = -0.5*a1-0.5*a2+0.5*a3

# seems like these are what we start on
rA = 0.5*a1
rB = 0.5*a2+0.5*a3
td = 0.5*(a1+a2+a3)



###Pauli matrices

I = np.array([[1,0],[0,1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1,0],[0,-1]])


sp = 0.5*(sx+1j*sy)
sm = 0.5*(sx-1j*sy)


## Functions to define H
def f1(k):
	return np.cos(np.dot(a1,k))-np.cos(np.dot(a2,k))

def f2(k):
	return 0.5*(np.cos(np.dot(delta1,k))-np.cos(np.dot(delta2,k))+np.cos(np.dot(delta3,k))-np.cos(np.dot(delta4,k)))

def f3(k):
	return 0.5*(np.cos(np.dot(delta1,k))+np.cos(np.dot(delta2,k))+np.cos(np.dot(delta3,k))+np.cos(np.dot(delta4,k)))

def g1(k):
	return np.sin(np.dot(a1,k))-1j*np.sin(np.dot(a2,k))

def g2(k):
	return 0.5*(np.sin(np.dot(delta1,k))-1j*np.sin(np.dot(delta2,k))-np.sin(np.dot(delta3,k))+1j*np.sin(np.dot(delta4,k)))

def h1(k):
	return 0.5*(np.sin(np.dot(delta1,k))+np.sin(np.dot(delta2,k))+np.sin(np.dot(delta3,k))+np.sin(np.dot(delta4,k)))

def h2(k):
	return 0.5*(np.sin(np.dot(delta1,k))-np.sin(np.dot(delta2,k))+np.sin(np.dot(delta3,k))-np.sin(np.dot(delta4,k)))

def make_H(k):
	a = t1*f1(k)*np.kron(sz,sz)
	b = t2*f2(k)*np.kron(sy,I)+t3*f3(k)*np.kron(sx,I)
	c = lm1*g1(k)*np.kron(I,sp)+np.conjugate(lm1)*np.conjugate(g1(k))*np.kron(I,sm)
	d = lm2*g2(k)*np.kron(sx,sp)+np.conjugate(lm2)*np.conjugate(g2(k))*np.kron(sx,sm)
	e = rho1*h1(k)*np.kron(sx,sz)
	f = rho2*h2(k)*np.kron(sy,sz)
	g = epsilon_z*np.kron(I,sz)

	H = 0.5*(a+b+c+d+e+f+g+np.conjugate((a+b+c+d+e+f+g).T))

	return H


def make_straight_path_x(k,nk):
	"""
	Returns a straight path along the kx direction (one lattice vector)
	k: Start point of path
	nk: Number of points along path
	"""

	k_out = np.zeros((nk,3))
	k_passed = 0
	for i in np.linspace(0,1,nk,endpoint=True):
        #add a little bit of b1
		k_out[k_passed,:] = k+np.array([1,0,0])*i
		k_passed += 1
	return k_out

# def make_straight_path_x(k,nk):
#     return np.linspace(k,k + np.array([1,0,0]),nk,endpoint=True)

def make_straight_path_y(k,nk):
	"""
	Returns a straight path along the ky direction (one lattice vector)
	k: Start point of path
	nk: Number of points along path
	"""

	k_out = np.zeros((nk,3))
	k_passed = 0
	for i in np.linspace(0,1,nk,endpoint=True):
        #add a bit of b2
		k_out[k_passed,:] = k+np.array([0,1,0])*i
		k_passed += 1
	return k_out

# def make_straight_path_y(k,nk):
#     return np.linspace(k, k+np.array([0,1,0]), nk, endpoint=True)


def make_straight_path_xy(k, nk):
     return np.linspace(k,k + np.array([1,1,0]),nk,endpoint=True)


def make_Wilson_loop(H_func,open_path, integration_path, bands,nk,orb_pos = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]],debug=True,tol=0.1):
    """
	This function computes the Wilson loop. It has been tested for four-band cases only so far.

	H_func -> Function generating the Hamiltonian. Should take only one parmeter (k), in scaled units e.g. k in [0,1]
	open path -> x-axis of Wilson loop plot, e.g. not integrated over. Should be array of base points to integrate over.
	integration_path -> The path to integrate over. Should be function taking a k-point (base point) and nk as input and returning an nk long array to integrate over
	bands -> Bands to include in the Wilson loop (e.g. "occupied bands")
	nk -> The number of k-points to integrate over (the resolution along x-axis is determined by the open path input parameter)
    
	orb_pos -> Position of orbitals in scaled units in the unit cell (needed for gauge matching condition)
	debug -> Boolean to control if eigenvalues are checked for consistency
	tol -> Tolerance. If debug=True, the code checks if abs(eigenvalus-1)<tol. Only used if debug = True
	"""
    
    num_bands = len(bands)
    num_states = len(orb_pos)

    eigs = np.zeros((len(open_path), num_bands))

    num_k_passed = 0
    
    # iterate through k points that designate the starting position of Wilson Loops
    for k in open_path:
        print(num_k_passed)
        P_tot = np.zeros((num_states,num_bands),dtype=np.complex128)
        U_0 = np.zeros((num_states,num_bands),dtype=np.complex128)
        U_current = np.zeros((num_states,num_bands),dtype=np.complex128)
        V_phase = np.zeros((num_states,num_states),dtype=np.complex128)
    
        #make k path for wilson loop given initial k value; k, and num of k points, nk
        int_k = integration_path(k,nk)
        #find Hamiltonian at initial k point
        H = H_func(int_k[0])
        #evals and evecs at initial point
        vals_0,vecs_0 = np.linalg.eigh(H)
        #remove small stuff
        vecs_0[np.abs(vecs_0)<1e-15] = 0
        #sort evecs
  		#idx_sorted = np.argsort(np.real(vals_0))
  		#vecs_0 = vecs_0[:,idx_sorted]
  		#vals_0 = vals_0[idx_sorted]
        
        #store original evecs
        for band in range(num_bands):
            U_0[:,band] = vecs_0[:,bands[band]]
        
    	#U_0 = orth(U_0)
        P_tot =  np.copy(U_0)
        int_k_passed = 0
        
        #go through integration path, each k represented by current_k
        for current_k in int_k:
            if int_k_passed != 0 and int_k_passed != nk-1: # if we not at the end or the beginning?
                U_current = np.zeros((num_states, num_bands),dtype=np.complex128)
            
                #diagonalise Hamiltonian at this point
                H = H_func(current_k)
                vals,vecs = np.linalg.eigh(H)
                vecs[np.abs(vecs)<1e-15] = 0
				#idx_sorted = np.argsort(np.real(vals))
				#vecs = vecs[:,idx_sorted]
				#vals = vals[idx_sorted]
				#print(vals)
    				
                #store evecs
                for band in range(num_bands):
                    U_current[:,band] = vecs[:,bands[band]]
                    
                    
                if debug:
                    if np.sum(np.abs(np.matmul(np.conjugate(U_current).T, U_current))-np.eye(num_bands, num_bands))>1e-10:
                        print("ERROR: Eigenvectors are not orthogonal!")
    				
    				
				# P_tot =  U * U^dagger * P_tot
                # Think this is making the Wilson Line Matrix
                P_tot = np.matmul(np.matmul(U_current, np.conjugate(U_current).T), P_tot)
            int_k_passed += 1
        
        #reciprocal lattice vec 
        G = int_k[-1]-int_k[0]
        
        #the phase you get per state from moving a reciprocal lattice vec
        for state in range(num_states):
            V_phase[state,state] = np.exp(1j*2.0*np.pi*np.dot(G, orb_pos[state]))
        
        P_tot = np.matmul(V_phase, P_tot)
        P_tot = np.matmul(np.conjugate(U_0.T),P_tot)

        # get evals and evecs of P_tot, this must be Wilson Line matrix
        vals,vecs = np.linalg.eig(P_tot)
        #store the phase of the eigs for this particular k start point
        eigs[num_k_passed,:] = np.sort(np.imag(np.log(vals)))
        
        if debug:
            for val in vals:
                if np.abs(np.abs(val)-1)>tol:
                    print("WARNING: Wilson eigenvalue %.3f is abs(%.3f) at k=[%.3f,%.3f] (number %i)!"%(np.imag(np.log(val)),np.abs(val),k[0],k[1],num_k_passed))
        num_k_passed += 1
    
    return eigs

				
		

# orbital positions in the unit cell? Think this should be same for all Wilson Line calcs because we just change 
# things in reciprocal space for Wilson Line
orb_pos = [[0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.5],[0.0,0.5,0.5]]

		
nk = 101
nk_y = 101
k_start = [0,0,0.5] # z stays at 0.5, x stays at 0
bands = [0,1]

# y_path = make_straight_path_y(k_start,nk_y)
y_path = np.linspace(k_start, k_start + np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]), nk, endpoint=True)
wilson_x = make_Wilson_loop(make_H, y_path, make_straight_path_xy, bands=bands,nk=nk,orb_pos = orb_pos)

#x_path = make_straight_path_x(k_start,nk_y)
#wilson_y = make_Wilson_loop(make_H, x_path, make_straight_path_y, bands=bands,nk=nk,orb_pos = orb_pos)

for i in range(len(bands)):
	plt.plot(y_path[:,1],wilson_x[:,i],'*')
plt.ylim([-np.pi-0.1,np.pi+0.1])
plt.xlabel(r"$k_y$")
plt.show()
