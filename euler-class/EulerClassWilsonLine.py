# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:55:37 2021

@author: Georgia Nixon
"""


place = "Georgia Nixon"

import numpy as np
import sys
sys.path.append('/Users/'+place+'/Code/MBQD/band-topology/euler-class')
sys.path.append('/Users/'+place+'/Code/MBQD/floquet-simulations/src')
from EulerClass2Hamiltonian import  Euler2Hamiltonian, GetEvalsAndEvecsEuler, AlignGaugeBetweenVecs
from Kagome3Hamiltonian import Kagome3, GetEvalsAndEvecsKagome
from hamiltonians import GetEvalsAndEvecsGen
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import pi, cos, sin
import matplotlib as mpl
from scipy.linalg import expm
import numpy.linalg as la

size=25
params = {
        'legend.fontsize': size*0.8,
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'font.size': size,
          'font.family': 'STIXGeneral',
          'mathtext.fontset': 'stix',
          }
mpl.rcParams.update(params)
params = {
          'xtick.bottom':True,
          'xtick.top':False,
          'ytick.left': True,
          'ytick.right':False,
          'axes.edgecolor' :'white',
          'xtick.minor.visible': False,
          'axes.grid':True,
          "axes.facecolor":"0.9",
          "grid.color":"1"
          }


mpl.rcParams.update(params)


CB91_Blue = 'darkblue'#'#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
red = "#FC4445"

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
               CB91_Purple,
                # CB91_Violet,
                'dodgerblue',
                'slategrey']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sh = "/Users/"+place+"/OneDrive - University of Cambridge/MBQD/Notes/Topology Bloch Bands/"
def RoundComplex(num, dp):
    return np.round(num.real, dp) + np.round(num.imag, dp) * 1j

def Normalise(v):
    norm_=np.linalg.norm(v)
    return v/norm_

def DifferenceLine(array2D):
    X = np.append(np.append(array2D[[-2]], array2D, axis=0), array2D[[1]], axis=0)
    xDiff = np.zeros((len(array2D), 2))
    for i in range(len(array2D)):
        xDiff[i] = np.array([X[i+2,0] - X[i,0], X[i+2,1] - X[i,1]])
    return xDiff

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

def CreateLinearLine(qxBegin, qyBegin, qxEnd, qyEnd, qpoints):
    kline = np.linspace(np.array([qxBegin,qyBegin]), np.array([qxEnd,qyEnd]), qpoints, endpoint=False)
    return kline

def CalculateBerryConnect(k, n0, n1, EulerHamiltonian):
    
    h = 0.00001;
    
    H = EulerHamiltonian(k)
    evals, evecs = GetEvalsAndEvecsEuler(H)
    
    #first eigenvector
    u0 = evecs[:,n0]
    u1 = evecs[:,n1]
    
    #dx direction
    kxx = k + np.array([h,0])
    H = EulerHamiltonian(kxx)

    _,evecsX = GetEvalsAndEvecsEuler(H)
    ux1 = evecsX[:,n1]
 
    #dy direction
    kyy = k + np.array([0,h])
    H = EulerHamiltonian(kyy)
    _,evecsY = GetEvalsAndEvecsEuler(H)
    uy1=evecsY[:,n1]

    xdiff = (ux1-u1)/h
    ydiff = (uy1-u1)/h
    
    berryConnect = 1j*np.array([np.vdot(u0,xdiff),np.vdot(u0,ydiff)])

    return berryConnect

def CalculateBerryConnectMatrix(k, dgbands=3):
    berryConnect = np.zeros((dgbands,dgbands, 2), dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            berryConnect[n0,n1] = CalculateBerryConnect(k, n0, n1)
    return berryConnect
            
def AbelianCalcWilsonLine(evecsFinal, evecsInitial, dgbands=3):
    wilsonLineAbelian = np.zeros([dgbands, dgbands], dtype=np.complex128)
    
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            wilsonLineAbelian[n0,n1] = np.vdot(evecsFinal[:,n1], evecsInitial[:,n0])
        
    return wilsonLineAbelian



def CalcOverlapEvecs(evecsFinal, evecsInitial, dgbands=3):
    
    M = np.zeros([dgbands, dgbands], dtype=np.complex128)
    for n0 in range(dgbands):
        for n1 in range(dgbands):
            M[n0,n1] = np.vdot(evecsFinal[:,n1], evecsInitial[:,n0])
    return M

def SetGaugeSVG(evecsFinal, evecsInitial):
    """Set gauge of evecsFinal to ensure maximum overlap with evecsInitial 
    using SVG strategy
    Return new aligned evecsFinal"""
    M = CalcOverlapEvecs(evecsFinal, evecsInitial, dgbands = 3)
    v, s, wh = la.svd(M)
    Mrotate = np.matmul(np.transpose(np.conj(wh)),np.transpose(np.conj(v)))
    evecsFinalNewGauge = np.dot(Mrotate, evecsFinal)
    return evecsFinalNewGauge

def SetGaugeByBand(evecsFinal, evecsInitial):
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





#%%
"""
Define Wilson Line path (kline)
"""

#num of points to calculate the wilson Line of
qpoints = 1000
radius=0.5
centre = [0,0]

# create rectangle line
kline = CreateLinearLine(0, 0, 2*pi, 0,  qpoints)
dqs = np.full((qpoints,2), [2*pi/qpoints,0])
# kline1 = CreateLinearLine(0.5, 2, 1.5, 2, qpoints)
# kline2 = CreateLinearLine(1.5, 2, 1.5, 0, qpoints)
# kline3 = CreateLinearLine(1.5, 0, 0.5, 0, qpoints)
# kline =np.vstack((kline0,kline1,kline2, kline3))

# create circle line
# kline = CreateCircleLine(radius, qpoints, centre = centre)
# dqs = CircleDiff(qpoints, radius)
totalPoints = len(kline)



#%%
# '''
# Calculate Wilson Line - Abelian
# '''

# k0 = kline[0]
# H = EulerHamiltonian(k0)
# _, evecsInitial = GetEvalsAndEvecsEuler(H)

# wilsonLineAbelian = np.zeros([totalPoints, 3, 3], dtype=np.complex128)
# # go through possible end points for k
# for i, kpoint in enumerate(kline):
#     #Find evec at k point, calculate Wilson Line abelian method
#     H = EulerHamiltonian(kpoint[0], kpoint[1])
#     _, evecsFinal = GetEvalsAndEvecs(H)
#     wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsFinal, evecsInitial)

    

# #%%
# """
# Calculate Wilson Line - Non Abelian
# """

# wilsonLineNonAbelian = np.zeros([totalPoints, 3, 3], dtype=np.complex128)
# currentArgument = np.zeros([3,3], dtype=np.complex128)


# for i, kpoint in enumerate(kline):
#     berryConnect = CalculateBerryConnectMatrix(kpoint)
#     dq = dqs[i]
#     berryConnectAlongKLine =  1j*np.dot(berryConnect, dq)
#     currentArgument = currentArgument + berryConnectAlongKLine
#     wilsonLineNonAbelian[i] = expm(currentArgument)
    
    
    #%%
"""
Calculate Wilson Line - 
    Calculate fully closed wilson lines for some size parameter
    Plot first wilson line eval vs. the size of the loop parameter
"""


#num of points to calculate the wilson Line of
qpoints = 1000
Nw = 50
centre = [0,0]

wilsonLineEvalsByPathWidth = np.empty((Nw, 3), dtype=np.complex128)

for i, radius in enumerate(np.linspace(0,2,Nw)):


    # create circle line
    kline = CreateCircleLine(radius, qpoints, centre = centre)
    dqs = CircleDiff(qpoints, radius)
    totalPoints = len(kline)

    
    WilsonLine = np.zeros((3,3), dtype=np.complex128)
    
    
    #find starting vecs
    
    k_vec = kline[0]
    H = EulerHamiltonian(k_vec)
    _, evecs0 = GetEvalsAndEvecsEuler(H)
    evecsOld = evecs0
    
    # go through and align vecs until the end
    for k_vec in kline[1:]:
        H = EulerHamiltonian(k_vec)
        _, evecs = GetEvalsAndEvecsEuler(H)
        
        evecs = SetGaugeSVG(evecs, evecsOld)
        # evecs = SetGaugeByBand(evecs, evecsOld)

        evecsOld = evecs
    
    #calculate overlap
    WilsonLine = CalcOverlapEvecs(evecsOld, evecs0)
    WLevals, _ = GetEvalsAndEvecsGen(WilsonLine)
    
    wilsonLineEvalsByPathWidth[i]=WLevals

wilsonevals0 = np.angle(wilsonLineEvalsByPathWidth[:,0])

wilsonevals0newphase = np.where(wilsonevals0<0 , np.abs(wilsonevals0), wilsonevals0)


plt.plot(np.linspace(0,1,Nw), wilsonevals0)
plt.show()



#%%

"""
Calculate Wilson Line - 
    Calculate half baked wilson lines around a circle
    Plot first wilson line eval vs. loop parameter
"""


def NumToString(num):
    string = str(num)
    string= string.replace(".", "p")
    return string


for radius in np.linspace(0.1, 3, 30):
    #num of points to calculate the wilson Line of
    qpoints = 1000
    Nw = 50
    centre = [1,1]
    # radius = 1.1
    kline = CreateCircleLine(radius, qpoints, centre = centre)
    dqs = CircleDiff(qpoints, radius)
    totalPoints = len(kline)
    
    
    # initial conditions
    k0 = kline[0]
    H = EulerHamiltonian(k0)
    _, evecs0 = GetEvalsAndEvecsEuler(H)
    evecsOld = evecs0
    
    
    wilsonLineAbelian = np.empty([totalPoints, 3, 3], dtype=np.complex128)
    # go through possible end points for k
    
    for i, kpoint in enumerate(kline[1:]):
        #Find evec at k point, calculate Wilson Line abelian method
        H = EulerHamiltonian(kpoint)
        _, evecsP = GetEvalsAndEvecsEuler(H)
        
        #rotate evecs to be in aligning gauge with evecs before
        # evecsP = SetGaugeSVG(evecsP, evecsOld)
        evecsP = SetGaugeByBand(evecsP, evecsOld)
        
        #find abelian Wilson Line crossover
        
        wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsP, evecs0)
        
        evecsOld = evecsP
    
    
    
    # #%%
    # '''
    # Plot [2,2] Matrix element
    # '''
    
    # m1 = 2
    # m2 = 2
    # multiplier = np.linspace(0,2*pi,totalPoints, endpoint=True)
    # fig, ax = plt.subplots(figsize=(12,9))
    
    # # ax.plot(multiplier, np.square(np.abs(wilsonLineNonAbelian[:,m1,m2])), label="Non Abelian")
    # ax.plot(multiplier, np.real(wilsonLineAbelian[:,m1,m2]))
    
    # ax.set_ylabel(r"$|W["+str(m1) +","+str(m2)+"]|^2 = |<\Phi_{q_f}^"+str(m1)+" | \Phi_{q_i}^"+str(m2)+">|^2$")
    # ax.set_xticks([0, pi, 2*pi])
    # ax.set_xticklabels(['0',r"$\pi$", r"$2\pi$"])
    
    
    # # ax.set_ylim([0,1.01])
    # ax.set_xlabel(r"Final quasimomentum point, going around circle with centre (0,0)")
    # # plt.legend(loc="upper right")
    # # plt.savefig(sh+ "WilsonLineEulerRectangle00.pdf", format="pdf")
    # plt.savefig(sh+ "WilsonLineEulerCircleMatrixEl,"+str(m1)+str(m2)+".pdf", format="pdf")
    # plt.show()   
    
    
    """
    Eigenvalue flow
    """ 
    
    multiplier = np.linspace(0,2*pi,totalPoints, endpoint=True)
        
    # calc evals for Wilson Line
    evalsAbelian = np.empty([totalPoints, 3], dtype=np.complex128)
    for i in range(totalPoints):
        wL = wilsonLineAbelian[i]
        wLEvals, _ = GetEvalsAndEvecsGen(wL)
        evalsAbelian[i,:] = wLEvals
        
        
    #get rid of last value which can be funny..
    evalsAbelian = evalsAbelian[:-1]
    multiplier = multiplier[:-1]
    
    
    
    sz = 8
    fig, ax = plt.subplots(figsize=(sz*1.3,sz))
    
    ax.plot(multiplier, np.angle(evalsAbelian[:,0]), '.', label=r'$1^{\mathrm{st}}$ eigenvalue')
    ax.plot(multiplier, np.angle(evalsAbelian[:,1]), '.', label=r'$2^{\mathrm{nd}}$ eigenvalue')
    ax.plot(multiplier, np.angle(evalsAbelian[:,2]), '.', label=r'$3^{\mathrm{rd}}$ eigenvalue')
    ax.set_xticks((0, pi/2,  pi, 3*pi/2,  2*pi))
    ax.set_xticklabels((r'$0$', r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ))
    ax.set_title( r'$\mathrm{Im} ( \mathrm{log}(e))$') 
    plt.legend()
    ax.set_yticks((-pi, -1,  0, 1,  pi))
    ax.set_yticklabels((r'$-\pi$', "-1", r"$0$", "$1$", r"$\pi$" ))
    fig.text(0.5, 0.03, 'final q-momentum', ha='center')
    # plt.savefig(sh+ "WilsonLineEulerCircleEvals,r="+NumToString(radius)+",c=(1,1).pdf", format="pdf")
    plt.show()




# sz = 5
# fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, constrained_layout=True, 
#                         figsize=(sz*len(apply)*1.3,sz))

# funcs = [np.real, np.imag, np.angle]
# for i , f in enumerate(funcs):
#     ax[i].plot(multiplier, f(evalsAbelian[:,0]), '.', label=r'$1^{\mathrm{st}}$ eigenvalue')
#     ax[i].plot(multiplier, f(evalsAbelian[:,1]), '.', label=r'$2^{\mathrm{nd}}$ eigenvalue')
#     ax[i].plot(multiplier, f(evalsAbelian[:,2]), '.', label=r'$3^{\mathrm{rd}}$ eigenvalue')
#     # ax[i].plot(multiplier, np.imag(evalsAbelian[:,0]), label='first eigenvalue')
#     # ax[i].plot(multiplier, np.angle(evalsAbelian[:,0]), 'x', label='first eigenvalue')
#     ax[i].set_xticks((0, pi/2,  pi, 3*pi/2,  2*pi))
#     ax[i].set_xticklabels((r'$0$', r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ))
 
# for i, f in enumerate(apply):
#     # ax[i].plot(multiplier, f(evalstoPlot), label='first eigenvalue')
#     ax[i].set_title(labels[i],  fontfamily='STIXGeneral') 
# plt.legend()
# ax[0].set_yticks((-pi, -1,  0, 1,  pi))
# ax[0].set_yticklabels((r'$-\pi$', "-1", r"$0$", "$1$", r"$\pi$" ))
# fig.text(0.5, -0.08, 'position along circle path (parameterised by angle relative to horizontal)', ha='center')
# plt.savefig(sh+ "WilsonLineEulerCircleEvals,r=2,c=(0,0).pdf", format="pdf")
# plt.show()

# # fig, ax = plt.subplots(figsize=(12,9))
# # # ax.plot(multiplier, np.real(evalsNonAbelian[:,0]), label="Non Abelian, first eigenvalue")
# # ax.plot(multiplier, np.angle(evalsAbelian[:,0]), label="first eigenvalue")
# # plt.legend()
# # plt.show()

#%%

""" for Kagome"""


"""
Calculate Wilson Line - 
    Calculate half baked wilson lines around a circle
    Plot first wilson line eval vs. loop parameter
"""



#num of points to calculate the wilson Line of
qpoints = 1000
# create rectangle line
kline = CreateLinearLine(0, 0, 2*pi, 2*pi,  qpoints)

totalPoints = len(kline)

# initial conditions
k0 = kline[0]
H = Kagome3(k0)
_, evecs0 = GetEvalsAndEvecsKagome(H)
evecsOld = evecs0


wilsonLineAbelian = np.empty([totalPoints, 3, 3], dtype=np.complex128)
# go through possible end points for k

for i, kpoint in enumerate(kline[1:]):
    #Find evec at k point, calculate Wilson Line abelian method
    H = Kagome3(kpoint)
    _, evecsP = GetEvalsAndEvecsKagome(H)
    
    #rotate evecs to be in aligning gauge with evecs before
    # evecsP = SetGaugeSVG(evecsP, evecsOld)
    evecsP = SetGaugeByBand(evecsP, evecsOld)
    
    #find abelian Wilson Line crossover
    wilsonLineAbelian[i] = AbelianCalcWilsonLine(evecsP, evecs0)
    
    evecsOld = evecsP
    
    
    
"""
Eigenvalue flow
""" 

multiplier = np.linspace(0,2*pi,totalPoints, endpoint=True)
    
# calc evals for Wilson Line
evalsAbelian = np.empty([totalPoints, 3], dtype=np.complex128)
for i in range(totalPoints):
    wL = wilsonLineAbelian[i]
    wLEvals, _ = GetEvalsAndEvecsGen(wL)
    evalsAbelian[i,:] = wLEvals
    
    
#get rid of last value which can be funny..
evalsAbelian = evalsAbelian[:-1]
multiplier = multiplier[:-1]


sz = 8
fig, ax = plt.subplots(figsize=(sz*1.3,sz))

ax.plot(multiplier, np.angle(evalsAbelian[:,0]), '.', label=r'$1^{\mathrm{st}}$ eigenvalue')
ax.plot(multiplier, np.angle(evalsAbelian[:,1]), '.', label=r'$2^{\mathrm{nd}}$ eigenvalue')
ax.plot(multiplier, np.angle(evalsAbelian[:,2]), '.', label=r'$3^{\mathrm{rd}}$ eigenvalue')
ax.set_xticks((0, pi/2,  pi, 3*pi/2,  2*pi))
ax.set_xticklabels((r'$0$', r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ))
ax.set_title( r'$\mathrm{Im} ( \mathrm{log}(e))$') 
plt.legend()
ax.set_yticks((-pi, -1,  0, 1,  pi))
ax.set_yticklabels((r'$-\pi$', "-1", r"$0$", "$1$", r"$\pi$" ))
fig.text(0.5, 0.03, 'final q-momentum', ha='center')
plt.show()












