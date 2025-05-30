#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:59:14 2025

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import scipy.sparse as scsp

def Tinf_sin(t,Tbase,DT,P) :
    """
    Function which computes the Tinf as a function of time

    Parameters
    ----------
    t : numpy array or float
        time.
    Tbase : scalar
        Value of the mean temperature.
    DT : float
        Amplitude in K or °C between Tmin and Tmax.
    P : float
        Period in seconds.

    Returns
    -------
    float
        External temperature at a spectific time.

    """
    return Tbase+DT*np.sin(2*np.pi*t/P)

def build_Matrix(df,mat,M) :
    """
    Function to contruct the Matrix and the vector add_rhs

    Parameters
    ----------
    df : dataframe
        dataframe with the mesh.
    mat : dictionnary
        dictionnary with the materials. Each key is a material
    Mstruct : numpy array
        Matrix with the structure i.e. 1 for a value and 2 for a symmetry
    ti : float
        time
    Returns
    -------
    Ms : Sparse array
        Sparse Matrix used for the computation.
    add_rhs : numpy array 
        Vector with all the additional values to the RHS (i.e. convection)

    """
    # M = deepcopy(Min)
    # df = deepcopy(dfin)
    # mat = deepcopy(matin)
    df = df.copy(deep=True)
    M = M.copy()
    
    add_rhs = np.zeros_like(M[0])
    
    df.dxw = df.dxw * 1e-3
    df.dxe = df.dxe * 1e-3
    df.dys = df.dys * 1e-3
    df.dyn = df.dyn * 1e-3
    df.SM = df.SM * 1e-6
    df.hint = df.hint * hint
    df.hext = df.hext * hext
    
    # indices convection
    left_CV = list(range(8,89,8)) + [104]
    right_CV = list(range(15,95,8)) 
    floor_CV = list(range(96,103))
    
    # corners
    corner = [95]
    top_left = [0]
    top_right = [7]
    bottom_left = [120]
    bottom_right = [135]
    floor_right = [103]
    
    # indices symmetry
    top_sym = list(range(1,7))
    right_sym = [119]
    bottom_sym = list(range(121,135))
    
    for direction in ['NW','NE','SW','SE'] :
        df['k_'+direction] = 0.0
        df['rho_'+direction] = 0.0
        df['cp_'+direction] = 0.0
        for materials in ['BLC','INS','PLA','CON'] :
            df.loc[df[direction]==materials,'k_'+direction] =  mat[materials]['k']
            df.loc[df[direction]==materials,'rho_'+direction] =  mat[materials]['rho']
            df.loc[df[direction]==materials,'cp_'+direction] =  mat[materials]['cp']
            
    df['rho_M'] = \
        df['rho_NW'] * df['rn'] * df['rw'] + \
        df['rho_NE'] * df['rn'] * df['re'] + \
        df['rho_SW'] * df['rs'] * df['rw'] + \
        df['rho_SE'] * df['rs'] * df['re']

    df['cp_M'] = \
        df['rho_NW'] * df['rn'] * df['rw'] * df['cp_NW'] + \
        df['rho_NE'] * df['rn'] * df['re'] * df['cp_NE'] + \
        df['rho_SW'] * df['rs'] * df['rw'] * df['cp_SW'] + \
        df['rho_SE'] * df['rs'] * df['re'] * df['cp_SE']
    df['cp_M'] = df['cp_M'] / df['rho_M']
    
    # Particular case for the corner
    corn_num = corner[0]
    df.at[corn_num,'rho_M'] = \
        df.at[corn_num,'rho_NW'] * df.at[corn_num,'rn'] * df.at[corn_num,'rw'] + \
        df.at[corn_num,'rho_SW'] * df.at[corn_num,'rs'] * df.at[corn_num,'rw'] + \
        df.at[corn_num,'rho_SE'] * df.at[corn_num,'rs'] * df.at[corn_num,'re']
    
    df.at[corn_num,'cp_M'] = (\
        df.at[corn_num,'rho_NW']*df.at[corn_num,'rn']*df.at[corn_num,'rw']*df.at[corn_num,'cp_NW'] + \
        df.at[corn_num,'rho_SW']*df.at[corn_num,'rs']*df.at[corn_num,'rw']*df.at[corn_num,'cp_SW'] + \
        df.at[corn_num,'rho_SE']*df.at[corn_num,'rs']*df.at[corn_num,'re']*df.at[corn_num,'cp_SE'] ) \
        / df.at[corn_num,'rho_M'] 
    
    # Compute the Fo numbers
    for direction in ['NW','NE','SW','SE'] :
        if direction[0] == 'N' :
            dy = df['dyn']
        elif direction[0] == 'S' :
            dy = df['dys']
        
        if direction[1] == 'W' :
            dx = df['dxw']
        elif direction[1] == 'E' :
            dx = df['dxe']
            
        df['Fo_xy_'+direction] = df['k_'+direction] * dt * dx / \
            (df['rho_M'] * df['cp_M'] * df['SM'] * dy)
        df['Fo_yx_'+direction] = df['k_'+direction] * dt * dy / \
            (df['rho_M'] * df['cp_M'] * df['SM'] * dx)
        
        df['Fo_xy_'+direction]=df['Fo_xy_'+direction].fillna(0)
        df['Fo_yx_'+direction]=df['Fo_yx_'+direction].fillna(0)
        
            
    # Compute the coefs    
    df['coef_N'] = - 0.5*(df['Fo_xy_NE'] + df['Fo_xy_NW'])
    df['coef_W'] = - 0.5*(df['Fo_yx_NW'] + df['Fo_yx_SW'])
    df['coef_E'] = - 0.5*(df['Fo_yx_NE'] + df['Fo_yx_SE'])
    df['coef_S'] = - 0.5*(df['Fo_xy_SE'] + df['Fo_xy_SW'])
    
    
    # Compute the Biot Numbers
    # direction is perpendicular to the convection
    df['Bi_SE'] = df['hext']*df['dxe']/df['k_SE'] 
    df['Bi_NE'] = df['hext']*df['dxe']/df['k_NE']
    
    df['Bi_SW'] = df['hint']*df['dxw']/df['k_SW']
    df['Bi_NW'] = df['hint']*df['dxw']/df['k_NW']
    
    # Biot number for the corner
    df.at[corner[0],'Bi_SE']=df.at[corner[0],'hint']*df.at[corner[0],'dys']/df.at[corner[0],'k_SE']
    df.at[corner[0],'Bi_NW']=df.at[corner[0],'hint']*df.at[corner[0],'dxw']/df.at[corner[0],'k_NW']
    
    
    # Biot numbers for inside floor
    for node in floor_CV :
        df.at[node,'Bi_SW']= df.at[node,'hint']*df.at[node,'dys']/df.at[node,'k_SW']
        df.at[node,'Bi_SE']= df.at[node,'hint']*df.at[node,'dys']/df.at[node,'k_SE']
    
    
    
    # Fill the matrix based on df and M 
    for node,row in zip(df.iterrows(),M):
        num,nd = node[0],dict(node[1]) # number and node info
        indices = np.argwhere(~np.isnan(row)).flatten()
        values =  row[indices]
        # print(indices)
        # print(num, " == ")
        # indices convection
        if num in left_CV :
            sumFo = 1-nd['coef_N']-nd['coef_E']-nd['coef_S']
            convec = 0.5*(nd['Bi_SE']*nd['Fo_yx_SE']+ nd['Bi_NE']*nd['Fo_yx_NE'])
            newrow = values * np.array(\
                [nd['coef_N'], sumFo+convec, nd['coef_E'], nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Text
        
        elif num in right_CV :
            sumFo = 1-nd['coef_N']-nd['coef_W']-nd['coef_S']
            convec = 0.5*(nd['Bi_SW']*nd['Fo_yx_SW']+nd['Bi_NW']*nd['Fo_yx_NW'])
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo+convec,  nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Tint
            
        elif num in floor_CV :
            sumFo = 1-nd['coef_W']-nd['coef_E']-nd['coef_S']
            convec = 0.5*(nd['Bi_SW']*nd['Fo_xy_SW']+nd['Bi_SE']*nd['Fo_xy_SE'])
            newrow = values * np.array(\
                [nd['coef_W'], sumFo+convec, nd['coef_E'],  nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Tint
        
        # corners
        elif num in corner :
            sumFo = 1-nd['coef_N']-nd['coef_W']-nd['coef_E']-nd['coef_S']
            convec = 0.5*(nd['Bi_SE']*nd['Fo_xy_SE']+nd['Bi_NW']*nd['Fo_xy_NW'])
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo+convec, nd['coef_E'],  nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Tint
        
        elif num in top_left : 
            sumFo = 1 -nd['coef_E']-2*nd['coef_S']
            convec = 0.5*(nd['Bi_SE']*nd['Fo_yx_SE']+ nd['Bi_NE']*nd['Fo_yx_NE'])
            newrow = values * np.array(\
                [sumFo+convec, nd['coef_E'], nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Text
            
        elif num in top_right :
            sumFo = 1 -nd['coef_W']-2*nd['coef_S']
            convec = 0.5*(nd['Bi_SW']*nd['Fo_yx_SW']+nd['Bi_NW']*nd['Fo_yx_NW'])
            newrow = values * np.array(\
                [nd['coef_W'], sumFo+convec,  nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Tint
            
        elif num in bottom_left :
            sumFo = 1-2*nd['coef_N']-nd['coef_E']
            convec = 0.5*(nd['Bi_SE']*nd['Fo_yx_SE']+ nd['Bi_NE']*nd['Fo_yx_NE'])
            newrow = values * np.array(\
                [nd['coef_N'], sumFo+convec, nd['coef_E']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Text
            
        elif num in bottom_right : 
            sumFo = 1-2*nd['coef_N']-2*nd['coef_W']
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo])
            M[num,indices] = newrow
            

        elif num in floor_right :
            sumFo = 1-2*nd['coef_W']-nd['coef_S']
            convec = 0.5*(nd['Bi_SW']*nd['Fo_xy_SW'])
            newrow = values * np.array(\
                [nd['coef_W'], sumFo+convec, nd['coef_S']])
            M[num,indices] = newrow
            add_rhs[num] = convec * Tint
            
        # indices symmetry
        elif num in top_sym :
            sumFo = 1-nd['coef_W']-nd['coef_E']-2*nd['coef_S']
            newrow = values * np.array(\
                [nd['coef_W'], sumFo, nd['coef_E'], nd['coef_S']])
            M[num,indices] = newrow
            
        elif num in right_sym :
            sumFo = 1-nd['coef_N']-2*nd['coef_W']-nd['coef_S']
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo, nd['coef_S']])
            M[num,indices] = newrow
            
        elif num in bottom_sym :
            sumFo = 1-2*nd['coef_N']-nd['coef_W']-nd['coef_E']
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo, nd['coef_E']])
            M[num,indices] = newrow
        
        else :
            sumFo = 1-nd['coef_N']-nd['coef_W']-nd['coef_E']-nd['coef_S']
            newrow = values * np.array(\
                [nd['coef_N'], nd['coef_W'], sumFo, nd['coef_E'], nd['coef_S']])
            M[num,indices] = newrow
         
    M[np.isnan(M)] = 0.0     
    Ms=scsp.csr_array(M)
    return Ms, add_rhs

def update_rhs(df,rhs,ti) :
    rhs = rhs.copy()
    df = df.copy(deep=True)
    indices_left_cv = np.argwhere(~np.isnan(df.hext.to_numpy())).flatten()
    rhs[indices_left_cv] = rhs_base[indices_left_cv] * Tinf_sin(ti, Tbase_ext, DT_Text, P_ext)
    return rhs

def contour_temp(df,stockT,index_time) :
    """
    Function to plot the field of temperature at a specific time

    Parameters
    ----------
    df : dataframe
        dataframe with the mesh.
    stockT : numpy array
        array with all the values of temperatures stored in columns .
    index_time : int
        index of the column i.e. a specific time 
        
        >>> savetime[index_time]
        
        gives the time

    Returns
    -------
    None.

    """    
    fig, ax = plt.subplots(layout='constrained')
    x_wall = np.unique(df['pos x'])
    y_wall = np.unique(df['pos y'])
    X,Y = np.meshgrid(x_wall,y_wall)
    mask = np.ones_like(X, dtype=bool)
    mask[:12,:8]=False
    mask[11:,:]=False
    T = stockT[:,index_time]
    Tup = np.c_[np.reshape(T[:88],(11,8)), Tint*np.ones((11,8))]
    Ttot = np.r_[Tup,np.reshape(T[88:],(3,16))]
    Tmask = np.ma.array(Ttot, mask=mask)
    cs = ax.contourf(X, Y, np.flipud(Tmask), 100, corner_mask=False)
    ax.contour(X, Y, np.flipud(Tmask), 10, corner_mask=False, colors='k')
   
    ax.axis('equal')
    ax.set_title('Temperature field at time : '+str(savetime[index_time])+' s')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    
    fig.colorbar(cs)

def temp_vs_time(time,stockT,pos) :
    """
    Function to plot the temperature vs time at a specific location. 
    The position is given by the index number

    Parameters
    ----------
    time : numpy array
        array with all the times stored.
    stockT : numpy array 
        array with all the values of temperatures stored in columns.
    pos : int
        index of the given point.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(layout='constrained')
    ax.set_title('Temperature vs time at position : '+str(pos))
    ax.set_ylabel('T [°C]')
    ax.set_xlabel('time [h]')
    
    ax.plot(time/3600,stockT[pos,:],label='Temperature at position : '+str(pos))
    plt.legend()    
    ax.grid(True)

if __name__ == '__main__' :

    plt.close('all')
    
    # Loading the materials properties
    Materials = pd.read_excel('Thermal_bridge.ods',sheet_name='Materials',index_col=0).to_dict()
   
    # Loading the mesh properties
    data = pd.read_excel('Thermal_bridge.ods',sheet_name='Nodes',index_col='node')
    
    # Loading the matrix with only 1 and 2
    Mstruct = pd.read_excel('Thermal_bridge.ods',sheet_name='Matrice',skiprows=1,
                      index_col=None, header=None)
    Mstruct = Mstruct.to_numpy()[:,1:]

    # Parameters
    dt = 10             # timestep
    tf = 5*24*3600      # final time
    dtsave = 300        # time step for saving results
    Tint = 20           # Int. Temperature
    Tinit = 20          # Init Temperature
    hint = 10           # Internal Heat transfer coefficient
    hext = 20           # External Heat transfer coefficient
    Tbase_ext = -5      # Mean Temperature outside
    DT_Text = 4         # Variation of the outside Temp. i.e. Tbase_ext +/- DT_Text
    P_ext = 24 * 3600   # Period in second
    
    # Build the matrix and the rhs_base vector
    Text = 1            # WARNING - Do not change
    Ms, rhs_base = build_Matrix(data,Materials,Mstruct)
    
    
    # update add_rhs with a real Text
    # Text_init = Tinf_sin(0, Tbase_ext, DT_Text , P_ext)
    add_rhs = update_rhs(data, rhs_base, 0)
    
    # Initial Temperature vector
    T = Tinit*np.ones(data.shape[0])
    saveT = T
    savetime = np.array([0])
    LU = scsp.linalg.splu(Ms)
    for ti in range(0,tf+1,dt) :
        # Build the matrix and the add_rhs vector
        
        add_rhs = update_rhs(data, rhs_base, ti)
        
        T = LU.solve(T + add_rhs)
        if ti>0 and (ti % dtsave == 0):
            saveT = np.c_[saveT, T]
            savetime = np.r_[savetime, ti]
            print("Time (s) : ",savetime[-1])
    
    # Graphs
    contour_temp(data, saveT, 60)
    temp_vs_time(savetime,saveT,95) 
    plt.figure(2)
    plt.plot(savetime/3600,Tinf_sin(savetime,Tbase_ext,DT_Text,P_ext),label=r'$T_\infty \left( t \right)$')
    plt.legend()