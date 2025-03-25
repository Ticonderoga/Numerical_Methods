#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:34:35 2025

@author: phil
"""


import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from tabulate import tabulate

if __name__ == '__main__':
    A = 1
    E = 0.5
    C = 0.3
    
    D0 = [A ,  A,  A,  A,  A,  A,
          -4, -4, -4, -4, -4, -4, -4, -4,
           E,
           C,  C,  C,
          -4, -4,  C]
    D1 =  [2,1,1,1,1,0,2,1,1,1,1,0,2,1,1,1,1,0,2,1]
    Dm1 = [1,1,1,1,2,0,1,1,1,1,2,0,1,2,1,1,2,0,1,2]
    D6 =  [2,2,2,2,2,2,1,1,1,1,1,1,1,1,1]
    Dm6 = [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2] 
    
    Mat = np.diag(Dm6,-6)+np.diag(Dm1,-1)+\
            np.diag(D0,0)+\
            np.diag(D1,1)+np.diag(D6,6)
    