#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:59:14 2025

@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as scsp

def build_Matrix(data,mat,Mstruct) :
    """
    Function to contruct the Matrix and the vector add_rhs

    Parameters
    ----------
    data : dataframe
        dataframe with the mesh.
    mat : dictionnary
        dictionnary with the materials. Each key is a material
    Mstruct : numpy array
        Matrix with the structure i.e. 1 for a value and 2 for a symmetry

    Returns
    -------
    Ms : Sparse array
        Sparse Matrix used for the computation.
    add_rhs : numpy array 
        Vector with all the additional values to the RHS (i.e. convection)

    """
    M = Mstruct
    add_rhs = np.zeros_like(M[0])
    
    data.dxw = data.dxw * 1e-3
    data.dxe = data.dxe * 1e-3
    data.dys = data.dys * 1e-3
    data.dyn = data.dyn * 1e-3
    data.SM = data.SM * 1e-6
    data.hint = data.hint * hint
    data.hext = data.hext * hext
    
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
        data['k_'+direction] = 0.0
        data['rho_'+direction] = 0.0
        data['cp_'+direction] = 0.0
        for materials in ['BLC','INS','PLA','CON'] :
            data.loc[data[direction]==materials,'k_'+direction] =  mat[materials]['k']
            data.loc[data[direction]==materials,'rho_'+direction] =  mat[materials]['rho']
            data.loc[data[direction]==materials,'cp_'+direction] =  mat[materials]['cp']
            
    data['rho_M'] = \
        data['rho_NW'] * data['rn'] * data['rw'] + \
        data['rho_NE'] * data['rn'] * data['re'] + \
        data['rho_SW'] * data['rs'] * data['rw'] + \
        data['rho_SE'] * data['rs'] * data['re']

    data['cp_M'] = \
        data['rho_NW'] * data['rn'] * data['rw'] * data['cp_NW'] + \
        data['rho_NE'] * data['rn'] * data['re'] * data['cp_NE'] + \
        data['rho_SW'] * data['rs'] * data['rw'] * data['cp_SW'] + \
        data['rho_SE'] * data['rs'] * data['re'] * data['cp_SE']
    data['cp_M'] = data['cp_M'] / data['rho_M']
    
    # Particular case for the corner
    corn_num = corner[0]
    data.at[corn_num,'rho_M'] = \
        data.at[corn_num,'rho_NW'] * data.at[corn_num,'rn'] * data.at[corn_num,'rw'] + \
        data.at[corn_num,'rho_SW'] * data.at[corn_num,'rs'] * data.at[corn_num,'rw'] + \
        data.at[corn_num,'rho_SE'] * data.at[corn_num,'rs'] * data.at[corn_num,'re']
    
    data.at[corn_num,'cp_M'] = (\
        data.at[corn_num,'rho_NW']*data.at[corn_num,'rn']*data.at[corn_num,'rw']*data.at[corn_num,'cp_NW'] + \
        data.at[corn_num,'rho_SW']*data.at[corn_num,'rs']*data.at[corn_num,'rw']*data.at[corn_num,'cp_SW'] + \
        data.at[corn_num,'rho_SE']*data.at[corn_num,'rs']*data.at[corn_num,'re']*data.at[corn_num,'cp_SE'] ) \
        / data.at[corn_num,'rho_M'] 
    
    # Compute the Fo numbers
    for direction in ['NW','NE','SW','SE'] :
        if direction[0] == 'N' :
            dy = data['dyn']
        elif direction[0] == 'S' :
            dy = data['dys']
        
        if direction[1] == 'W' :
            dx = data['dxw']
        elif direction[1] == 'E' :
            dx = data['dxe']
            
        data['Fo_xy_'+direction] = data['k_'+direction] * dt * dx / \
            (data['rho_M'] * data['cp_M'] * data['SM'] * dy)
        data['Fo_yx_'+direction] = data['k_'+direction] * dt * dy / \
            (data['rho_M'] * data['cp_M'] * data['SM'] * dx)
        
        data['Fo_xy_'+direction]=data['Fo_xy_'+direction].fillna(0)
        data['Fo_yx_'+direction]=data['Fo_yx_'+direction].fillna(0)
        
            
    # Compute the coefs    
    data['coef_N'] = - 0.5*(data['Fo_xy_NE'] + data['Fo_xy_NW'])
    data['coef_W'] = - 0.5*(data['Fo_yx_NW'] + data['Fo_yx_SW'])
    data['coef_E'] = - 0.5*(data['Fo_yx_NE'] + data['Fo_yx_SE'])
    data['coef_S'] = - 0.5*(data['Fo_xy_SE'] + data['Fo_xy_SW'])
    
    
    # Compute the Biot Numbers
    # direction is perpendicular to the convection
    data['Bi_SE'] = data['hext']*data['dxe']/data['k_SE'] 
    data['Bi_NE'] = data['hext']*data['dxe']/data['k_NE']
    
    data['Bi_SW'] = data['hint']*data['dxw']/data['k_SW']
    data['Bi_NW'] = data['hint']*data['dxw']/data['k_NW']
    
    # Biot number for the corner
    data.at[corner[0],'Bi_SE']=data.at[corner[0],'hint']*data.at[corner[0],'dys']/data.at[corner[0],'k_SE']
    data.at[corner[0],'Bi_NW']=data.at[corner[0],'hint']*data.at[corner[0],'dxw']/data.at[corner[0],'k_NW']
    
    
    # Biot numbers for inside floor
    for node in floor_CV :
        data.at[node,'Bi_SW']= data.at[node,'hint']*data.at[node,'dys']/data.at[node,'k_SW']
        data.at[node,'Bi_SE']= data.at[node,'hint']*data.at[node,'dys']/data.at[node,'k_SE']
    
    
    
    # Fill the matrix based on data and M 
    for node,row in zip(data.iterrows(),M):
        num,nd = node[0],dict(node[1]) # number and node info
        indices = np.argwhere(~np.isnan(row)).flatten()
        values =  row[indices]
        
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

def contour_temp(data,stockT,index_time) :
    """
    Function to plot the field of temperature at a specific time

    Parameters
    ----------
    data : dataframe
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
    x_wall = np.unique(data['pos x'])
    y_wall = np.unique(data['pos y'])
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
    ax.set_xlabel('time [min]')
    
    ax.plot(time/60,stockT[pos,:])
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
    tf = 24*3600        # final time
    Text = -10          # Ext. Temperature
    Tint = 25           # Int. Temperature
    Tinit = 25          # Init Temperature
    hint = 10           # Internal Heat transfer coefficient
    hext = 20           # External Heat transfer coefficient

    # Build the matrix and the add_rhs vector
    Ms, add_rhs = build_Matrix(data,Materials,Mstruct)
    
    
    # Initial Temperature vector
    T = Tinit*np.ones_like(add_rhs)
    saveT = T
    savetime = np.array([0])
    
    for ti in range(0,tf+1,dt) :
        T = scsp.linalg.spsolve(Ms, T + add_rhs)
        if ti>0 and (ti % 300 == 0):
            saveT = np.c_[saveT, T]
            savetime = np.r_[savetime, ti]
            print("Time (s) : ",savetime[-1])
    
    # Graphs
    contour_temp(data, saveT, 60)
    temp_vs_time(savetime,saveT,95) 
  

