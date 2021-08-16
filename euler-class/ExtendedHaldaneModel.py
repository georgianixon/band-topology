# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:14:05 2021

@author: Georgia Nixon
"""


import numpy as np
from numpy import sqrt, sin, cos, exp
import matplotlib.pyplot as plt
import matplotlib as mpl

#pauli matrics
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])

#nearest neighbor vecs
a1 = np.array([sqrt(3)/2, -1/2])
a2 = np.array([-sqrt(3)/2, -1/2])
a3 = np.array([0, 1])
a = np.array([a3, a1, a2])

# 
Q6 = np.array([[1/2, -sqrt(3)/2], [sqrt(3)/2, 1/2]])
Q3 = np.array([[-1/2, -sqrt(3)/2], [sqrt(3)/2, -1/2]])

#n2 vec
b1 = np.array([sqrt(3)/2, -3/2])
b2 = np.array([-sqrt(3)/2, -3/2])
b3 = np.array([sqrt(3), 0])
b4 = -b1
b5 = -b2
b6 = -b3
# b1 = np.dot(Q6, b1)
# b2 = np.dot(Q6, b2)
# b3 = np.dot(Q6, b3)
# b4 = np.dot(Q6, b4)
# b5 = np.dot(Q6, b5)
# b6 = np.dot(Q6, b6)

b = np.array([b1, b2, b3, b4, b5, b6])


#n3 vecs
c1 = np.array([-sqrt(3), 1])
c2 = np.array([sqrt(3), 1])
c3 = np.array([0, -2])
# c1 = np.dot(Q3, c1)
# c2 = np.dot(Q3, c2)
# c3 = np.dot(Q3, c3)
c = np.array([c1, c3, c2])

#params
t1 = 1
t2 = 0.6
t3 = 0.6
m = 1
lambdaR = 0.3

k = np.array([0.4,0.1])


d1 = np.sum([t1*cos(np.dot(k, a[i]))+t3*cos(np.dot(k, c[i])) for i in range(3)])
d2 = np.sum([-t1*sin(np.dot(k, a[i])) - t3*sin(np.dot(k, c[i])) for i in range(3)])
d3p = m+np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])
d3m = m-np.sum([t2*sin(np.dot(k, b[i])) for i in range(6)])


H = np.zeros((4,4), dtype=np.complex128)
#H Hal
H[:2,:2] = d1*s1 + d2*s2 + d3p*s3
H[2:,2:] = d1*s1 + d2*s2 + d3m*s3


HR = 1j*lambdaR*np.sum(np.array([(c[i,0]*s2 - c[i,1]*s1)*exp(1j*np.dot(k, c[i])) for i in range(3)]), axis=0)

H[0,1] = HR[0,0]
H[1,0] = np.conj(HR[0,0])
H[0,3] = HR[0,1]
H[3,0] = np.conj(HR[0,1])
H[2,1] = HR[1,0]
H[1,2] = np.conj(HR[1,0])
H[2,3] = HR[1,1]
H[3,2] = np.conj(HR[1,1])

apply = [
         np.abs, 
         np.real, np.imag]
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sz = 20
fig, ax = plt.subplots(nrows=1, ncols=len(apply), sharey=True, constrained_layout=True, 
                       figsize=(sz,sz/2))
for n1, f in enumerate(apply):
    pcm = ax[n1].matshow(f(H), interpolation='none', cmap='PuOr',  norm=norm)
    ax[n1].tick_params(axis="x", bottom=True, top=False, labelbottom=True, 
      labeltop=False)  
    ax[n1].set_xlabel('m')
ax[0].set_ylabel('n', rotation=0, labelpad=10)
cax = plt.axes([1.03, 0.1, 0.03, 0.8])
fig.colorbar(plt.cm.ScalarMappable(cmap='PuOr', norm=norm), cax=cax)

#%%


    
    


