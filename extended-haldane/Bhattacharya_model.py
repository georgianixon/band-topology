# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:27:57 2021

@author: Georgia
"""

import numpy as np
from numpy import sqrt, pi, cos, sin, exp

def Haldane3_Bhattacharya(k, params):
    
    phi, M, t1, t2, t3 = params
    
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s2 = np.array([[0,-1j],[1j,0]])
    s3 = np.array([[1,0],[0,-1]])
    
    a1 = (1/2)*np.array([sqrt(3), 3])
    a2 = (1/2)*np.array([-sqrt(3), 3])
    
    h0 = 2*t2*cos(phi)*[cos(np.dot(k, a1)) + cos(np.dot(k, a2)) + cos(np.dot(k, a1-a2))]
    h1 = t1*(1 + cos(np.dot(k, a1)) +cos(np.dot(k, a2))) + t3*(cos(np.dot(k, a1 + a2)) + 2*cos(np.dot(k, a1-a2)))
    h2 = t1*(sin(np.dot(k, a1)) + sin(np.dot(k, a2))) - 2*t3*sin(np.dot(k, a1+a2))
    h3 = M + 2*t2*sin(phi)*(sin(np.dot(k, a2)) - sin(np.dot(k, a1)) + sin(np.dot(k, a1-a2)))
    
    H = h0*s0+ h1*s1 + h2*s2 + h3*s3
    
    return H