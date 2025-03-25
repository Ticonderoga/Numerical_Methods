#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:47:50 2025

@author: phil
"""

import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt
from tabulate import tabulate

def Cout_Total(x,y):
    return 800*x*y+2e4*(x+y)/x/y+50

def Vec_Ct(X) :
    return Cout_Total(X[0],X[1])

def grad_Ct(x,y):
    dCx=800*y-2e4/x**2
    dCy=800*x-2e4/y**2
    return np.array([dCx,dCy])

def Vec_grad_Ct(X) :
    return grad_Ct(X[0],X[1])


if __name__ == '__main__':
    plt.close('all')
    x=np.linspace(1,4,1000)
    y=x
    X,Y=np.meshgrid(x,y)
    Z=Cout_Total(X,Y)
    plt.contourf(X,Y,Z,60,cmap=plt.cm.jet)
    myaxis=plt.axis()
    plt.colorbar()
    CS=plt.contour(X,Y,Z,10,colors='k')
    plt.clabel(CS,fmt='%3g')
    plt.xlabel('x en [m]')
    plt.ylabel('y en [m]')
    plt.title('Coût total')

    xinit=1.5
    yinit=3.5  
   
    Popt=scopt.fmin(Vec_Ct,[xinit,yinit],full_output=True,retall=True)
    xopt,fopt,niter,funcalls,warn,allvecs=Popt
    positions=np.array(allvecs)
    plt.plot(positions[:,0],positions[:,1],'wo-')
    plt.axis(myaxis)
    print(tabulate(positions, disable_numparse=True,floatfmt='5g',tablefmt='pipe',headers=("x","y")))

    #%% Conjugate Gradient manually
     
    x0=np.array([xinit,yinit])
    positions = [x0]
    for i in range(10) :
        g0=grad_Ct(x0[0],x0[1])
        d0=-g0
        g=lambda gm:Cout_Total(x0[0]+gm*d0[0],x0[1]+gm*d0[1])
        gm0=scopt.fmin(g,0,disp=False)[0]
        x1=x0+gm0*d0
        positions.append(x1)
        x0=x1

    positions=np.array(positions)
    plt.plot(positions[:,0],positions[:,1],'go-')
    plt.axis(myaxis)
    print(tabulate(positions, disable_numparse=True,floatfmt='5g',tablefmt='pipe',headers=("x","y")))
    
    #%% Newton method
    
    x0=np.array([xinit,yinit])
    Popt=scopt.fmin_ncg(Vec_Ct,x0,fprime=Vec_grad_Ct,full_output=True,retall=True)
    xopt,fopt,fcalls,hcalls,niter,warn,allvecs=Popt
    positions=np.array(allvecs)
    plt.plot(positions[:,0],positions[:,1],'co-')
    plt.axis(myaxis)
    print(tabulate(positions, disable_numparse=True,floatfmt='5g',tablefmt='pipe',headers=("x","y")))

