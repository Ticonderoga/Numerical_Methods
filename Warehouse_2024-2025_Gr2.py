#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:10:26 2025

@author: phil
"""

import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from tabulate import tabulate

def dist_point(x,y,Point) :
    xP = Point[0]
    yP = Point[1]
    return ((x-xP)**2+(y-yP)**2)**0.5

def dist_tot(x,y) :
    dA = dist_point(x, y, A)
    dB = dist_point(x, y, B)
    dC = dist_point(x, y, C)
    return dA+dB+dC
    

if __name__ == '__main__':
    plt.close('all')
    A = (0,0)
    B = (300,400)
    C = (700,300)
    
    x = np.linspace(-50,750,101)
    y = np.linspace(-50,550,126)
    X,Y = np.meshgrid(x,y)
    
    plt.contourf(X,Y,dist_tot(X, Y),100)
    plt.colorbar()
    plt.contour(X,Y,dist_tot(X, Y),20,colors='w')
    plt.annotate('A', A)
    plt.annotate('B', B)
    plt.annotate('C', C)
    