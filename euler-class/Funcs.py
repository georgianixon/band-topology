# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 20:29:18 2022

@author: Georgia
"""

import numpy as np
from numpy import cos, pi, sin

def VecToStringSave(q0):
    a = q0[0]
    b = q0[1]
    if type(a) != int and type(a) != np.int32:
        
        if a.is_integer():
            a = int(a)
    if type(b) != int and type(b) != np.int32:
        if b.is_integer():
            b = int(b)
        
    return "("+str(a).replace(".", "p")+","+str(b).replace(".", "p")+")"


def VecToString(q0):
    a = q0[0]
    b = q0[1]
    if type(a) != int and type(a) != np.int32:
        
        if a.is_integer():
            a = int(a)
        else:
            a = round(a, 2)
    if type(b) != int and type(b) != np.int32:
        if b.is_integer():
            b = int(b)
        else:
            b = round(b,2)

    return "("+str(a)+","+str(b)+")"

def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def CreateCircleLine(r, points, centre=[0,0]):
    CircleLine =  np.array([[cos(x)*r+centre[0],sin(x)*r+centre[1]] for x in np.linspace(0, 2*pi, points, endpoint=True)])
    return CircleLine

def CircleDiff(points, radius):
    """
    Gives vectors tangent to the circle for various theta between 0 and 2*pi
    Vector size = distance between points on the circle
    """
    circleDiffNormalised = np.array([[-sin(theta),cos(theta)] for theta in np.linspace(0, 2*pi, points, endpoint=True)])
    dtheta = 2*pi/(points-1)
    dqlength = 2*radius*sin(dtheta)
    circleDiff = circleDiffNormalised*dqlength
    return circleDiff

def CreateLinearLine(qBegin, qEnd,  qpoints):
    kline = np.linspace(qBegin, qEnd, qpoints)
    return kline





def SquareLine(q0, v1, v2, qpoints):
    kline = np.linspace(q0, q0+v1, qpoints)
    kline2 = np.linspace(q0+v1, q0+v1+v2, qpoints)[1:]
    kline3 = np.linspace(q0+v1+v2, q0+v2, qpoints)[1:]
    kline4 = np.linspace(q0+v2, q0, qpoints)[1:]
    return np.vstack([kline, kline2, kline3, kline4])

def FivePointLine(q0, q1, q2, q3, q4, qpoints):
    kline = np.linspace(q0, q1, qpoints)
    kline2 = np.linspace(q1, q2, qpoints)[1:]
    kline3 = np.linspace(q2, q3, qpoints)[1:]
    kline4 = np.linspace(q3, q4, qpoints)[1:]
    return np.vstack([kline, kline2, kline3, kline4])


def FindOverlapLine(line1, line2, line3, line4):
    thresh = 0.001
    result = np.empty(len(line1))
    for i, (a, b, c, d) in enumerate(zip(line1, line2, line3, line4)):
        #is a doubled
        if np.abs(a-b) < thresh:
            result[i] = a
        elif np.abs(a - c) < thresh:
            result[i] = a
        elif np.abs(a - d) < thresh:
            result[i] = a
        #is b doubled
        elif np.abs(b - c) < thresh:
            result[i] = b
        elif np.abs(b - d) < thresh:
            result[i] = b
        # is c doubled
        elif np.abs(c - d) < thresh:
            result[i] = c
        else:
            #all random cuz theta = 0
            result[i] = 0
    return result

def FindOverlap(a, b, c, d):
    thresh = 0.001

    #is a doubled
    if np.abs(a-b) < thresh:
        result = a
    elif np.abs(a - c) < thresh:
        result = a
    elif np.abs(a - d) < thresh:
        result= a
    #is b doubled
    elif np.abs(b - c) < thresh:
        result = b
    elif np.abs(b - d) < thresh:
        result = b
    # is c doubled
    elif np.abs(c - d) < thresh:
        result = c
    else:
        #all random cuz theta = 0
        result = 0
    return result

def InverseSin(val):
    """
    np.arcsin gives a value between -pi/2 and pi/2
    in fact we want two values between 0 and 2*pi
    """
    #get first val between -pi/2 and pi/2
    alpha1 = np.arcsin(val)
    #get second val which will be between pi/2 and 3 pi/2
    alpha2 = pi - alpha1 
    if alpha1 <0:
        #alpha1 is between -pi/2 and 0
        alpha1 = alpha1 + 2*pi
    return alpha1, alpha2
    

def InverseCos(val):
    """
    np.arccos gives a value beween 0 and pi
    in fact we want two values between 0 and 2 pi
    """
    # get first val between 0 and pi
    alpha1 = np.arccos(val)
    alpha2 = 2*pi - alpha1
    return alpha1, alpha2


def ProjReal(vec):
     assert(round(np.imag(vec), 26)==0)
     return np.real(vec)