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
             markersize=12)
    offset = 20
    plt.annotate("A", (A[0]+offset,A[1]+offset))
    plt.annotate("B", (B[0],B[1]+offset))
    plt.annotate("C", (C[0]+offset,C[1]))
    
    plt.axis([0,800,0,500])
    
    # Optimisation
    P = (200,200)
    xinit, yinit = P  
    Popt=scopt.fmin(vec_dist_tot,
                    [xinit,yinit],
                    full_output=True,
                    retall=True)
    
    
    
    
    
    
    
    
    
    
    