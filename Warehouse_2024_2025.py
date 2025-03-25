#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:28:21 2025

@author: phil
"""

import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from tabulate import tabulate

def dist_pt(x,y,point) :
    xpoint = point[0]
    ypoint = point[1]
    return ((x-xpoint)**2+(y-ypoint)**2)**0.5

def dist_tot(x,y) :
    dA = dist_pt(x, y, A)
    dB = dist_pt(x, y, B)
    dC = dist_pt(x, y, C)
    return dA+dB+dC

def vec_dist_tot(X) :
    x = X[0]
    y = X[1]
    return dist_tot(x, y)

def grad_dist(x,y) :
    xA,yA = A
    xB,yB = B
    xC,yC = C
    dA = dist_pt(x, y, A)
    dB = dist_pt(x, y, B)
    dC = dist_pt(x, y, C)
    
    dfdx = (x-xA)/dA + (x-xB)/dB + (x-xC)/dC
    dfdy = (y-yA)/dA + (y-yB)/dB + (y-yC)/dC

    return [dfdx,dfdy]

def vec_grad_dist(X) : 
    x = X[0]
    y = X[1]
    return grad_dist(x, y)


if __name__ == '__main__':
    
    # définition des points
    A = (0,0)
    B = (300,400)
    C = (700,300)
        
    # Tracé du graphique
    plt.close('all')
    plt.figure()
    
    x = np.linspace(0,800,151)
    y = np.linspace(0,500,81)
    X,Y = np.meshgrid(x,y)
    
    Dtot = dist_tot(X, Y)
    plt.contourf(X,Y,Dtot,100,cmap='jet')
    plt.colorbar()

    factories = np.array([A,B,C])
    plt.plot(factories[:,0],factories[:,1],'kh', 
             mec='k',mfc='w',markersize=12)
    offset = 20
    plt.annotate("A", xy = A, 
                 xytext=(A[0]+offset,A[1]+offset),
                 arrowprops=dict(facecolor='black',arrowstyle="->",))
    plt.annotate("B", xy = B, 
                 xytext=(B[0],B[1]+offset),
                 arrowprops=dict(facecolor='black',arrowstyle="->",))
    
    plt.annotate("C", xy = C,
                 xytext = (C[0]+offset,C[1]),
                 arrowprops=dict(facecolor='black',arrowstyle="->",))
    
    plt.axis([0,800,0,500])
    
    # Optimisation with simplex method
    print("SIMPLEX")
    Pinit = (200,200)
    Popt=scopt.fmin(vec_dist_tot,
                    Pinit,
                    full_output=True,
                    retall=True)
    
    xopt,fopt,niter,funcalls,warn,allvecs=Popt
    positions=np.array(allvecs)
    plt.plot(positions[:,0],positions[:,1],'wo-',mec='k',label='Simplex')
    plt.legend()
    print(tabulate(positions, disable_numparse=True,
                   floatfmt='5g',tablefmt='pipe',headers=("x","y")))
    
    # Optimisation with Newton method
    print('NEWTON')
    Pinit = (200.,200.)
    Popt=scopt.fmin_ncg(vec_dist_tot,
                        Pinit,
                        vec_grad_dist, 
                        full_output=True,
                        retall=True)
    
    xopt,fopt,niter,funcalls,gcalls,warn,allvecs=Popt
    positions=np.array(allvecs)
    plt.plot(positions[:,0],positions[:,1],'go-',mec='k',label='Newton')
    plt.legend()
    print(tabulate(positions, disable_numparse=True,
                   floatfmt='5g',tablefmt='pipe',headers=("x","y")))
    
    

    plt.tight_layout()
    
    
    
    
    
    
    
    
    