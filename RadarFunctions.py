#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
from scipy.io import loadmat
import sys

###############################################################################

### Load STO radar data ###

def loadStoMigData(fname,uice=168.,CReSIS=False):
    mfile = loadmat(fname)
    
    # data file variables
    migdata = mfile['migdata']              # the actual radar data
    surface = mfile['elev'][:,0]               # elevation of the surface
    time = mfile['travel_time'][0]          # travel times for any one trace
    # Coordinates
    lat = mfile['lat'][:,0]
    lon = mfile['long'][:,0]
    if CReSIS==True:
        time = mfile['travel_time'][:,0]          # travel times for any one trace
        lat = mfile['lat'][0]
        lon = mfile['long'][0]

    ### Calculations ###
    # Conversions
    vdist = time*uice/2.                  # distance to the top of the picked wavelet (m)
    # calculate distance using haversine formula
    dlat=lat*np.pi/180.
    dlon=lon*np.pi/180.
    R = 6371000.                            # Radius of the earth (m)
    a = np.sin((dlat-dlat[0])/2.)**2.+np.cos(dlat[0])*np.cos(dlat)*np.sin((dlon-dlon[0])/2.)**2.
    dist = 2.*R*np.arcsin(np.sqrt(a))
    
    return migdata,surface,time,dist,vdist


### Load STO pick data ###

def loadStoPickData(fname,uice=168.,CReSIS=False):
    pfile = loadmat(fname)
    
    # Pickfile data
    pnum = pfile['picks'][0][0][5]          # pick id
    ppower = pfile['picks'][0][0][4]        # power of picked layer
    psamp0 = pfile['picks'][0][0][0]-1      # sample number for top of Ricker Wavelet, subtract 1 for python indexing
    psamp1 = pfile['picks'][0][0][1]-1      # sample number for center of Ricker Wavelet, subtract 1 for python indexing
    psamp2 = pfile['picks'][0][0][2]-1      # sample number for bottom of Ricker Wavelet, subtract 1 for python indexing
    ptimes = pfile['picks'][0][0][3]        # pick time is to the top of the Ricker Wavelet
    
    # Coordinates
    lat = pfile['geocoords'][0][0][0].flatten()
    lon = pfile['geocoords'][0][0][1].flatten()
    x_coord = pfile['geocoords'][0][0][3].flatten()
    y_coord = pfile['geocoords'][0][0][4].flatten()
    
    ### Calculations ###
    # Conversions
    pdist = ptimes*uice/2.                  # distance to the top of the picked wavelet (m)
    ppower = 10.*np.log10(ppower)           # dB scale  
    # the cresis data was already power, then the picker squared it again
    if CReSIS == True:
        ppower = ppower/2.
        
    # calculate distance using haversine formula
    dlat=lat*np.pi/180.
    dlon=lon*np.pi/180.
    R = 6371000.                            # Radius of the earth (m)
    a = np.sin((dlat-dlat[0])/2.)**2.+np.cos(dlat[0])*np.cos(dlat)*np.sin((dlon-dlon[0])/2.)**2.
    dist = 2.*R*np.arcsin(np.sqrt(a))
    
    return ppower,psamp1,pdist,lat,lon,x_coord,y_coord,dist,pnum

###############################################################################    

### Fresnel's Equations ###
    
# Assuming that the materials are not magnetic
    
def Fresnel(n1,n2,theta_i):
    if np.all(theta_i > 2*np.pi) or isinstance(n1,int) or isinstance(n2,int):
        sys.exit("Input the incident angle in radians and the refractive indices as floating points.")
    # reflection angle is equal to incident angle
    theta_r = theta_i
    # Snell's law for the angle of refraction (transmission)
    theta_t = np.arcsin((n1/float(n2))*np.sin(theta_i))
    # P-Polarized
    Rs = (n1*np.cos(theta_i)-n2*np.cos(theta_t))/(n1*np.cos(theta_i)+n2*np.cos(theta_t))
    Ts = (2*n1*np.cos(theta_i))/(n1*np.cos(theta_i)+n2*np.cos(theta_t))
    # S-Polarized
    Rp = (n2*np.cos(theta_i)-n1*np.cos(theta_t))/(n2*np.cos(theta_i)+n1*np.cos(theta_t))
    Tp = (2*n1*np.cos(theta_i))/(n2*np.cos(theta_i)+n1*np.cos(theta_t))
    # Brewster's Angle
    theta_b = np.arctan(n2/n1)
    # Total Internal Reflection 
    theta_tir = np.arcsin(n2/n1)
    return theta_r,theta_t,theta_b,theta_tir,Rp,Tp,Rs,Ts

###############################################################################

### Attenuation rate to temperature ###

# See MacGregor et al. 2007
# TODO: move the constants to another script so that they are standardized
# TODO: This is for pure ice. I need to rethink for different ion concentrations

def attTemp(T,Hplus=1.3,Clminus=4.2):    
    eps0 = 8.85e-12     # permittivity of free space
    eps = 3.2           # relative permittivity of ice
    c = 3e8             # speed of light (m/s)
    k = 1.38e-23        # J K-1
    Tr = 251.           # reference temperature K
    # Activation Energies
    E0 = 0.55*1.6e-19   # J
    EH = 0.2*1.6e-19     # J
    ECl = 0.19*1.6e-19  # J
    # Pure ice conductivity
    sig0 = 7.2          # S m-1
    # Ion concentrations
    muH = 3.2           # S m-1 M-1
    muCl = 0.43          # S m-1 M-1
    # Convert to K
    T_K = T + 273.15
    # Calculate conductivity, MacGregor et al. (2007) eq. 1
    sig_pure = sig0*np.exp(E0/k*(1/Tr-1/T_K))
    sigH = muH*Hplus*np.exp(EH/k*(1/Tr-1/T_K))
    sigCl = muCl*Clminus*np.exp(ECl/k*(1/Tr-1/T_K))
    sig = sig_pure+sigH+sigCl
    # Convert to Attenuation Rate
    a = 10.*np.log10(np.exp(1))/(eps0*np.sqrt(eps)*c)   # MacGregor et al. (2007) eq. 10
    a *= 1e-3           # +3 for m-1 to km-1 and -6 for microSiemen
    att = sig*a
    return att

def pureTempAtt(att):
    eps0 = 8.85e-12     # permittivity of free space
    eps = 3.2           # relative permittivity of ice
    c = 3e8             # speed of light (m/s)
    k = 1.38e-23        # J K-1
    Tr = 251.           # reference temperature K
    # Activation Energies
    E0 = 0.55*1.6e-19   # J
    # Pure ice conductivity
    sig0 = 7.2          # S m-1
    # calculate the conductivity
    a = 10.*np.log10(np.exp(1.))/(eps0*np.sqrt(eps)*c)   # MacGregor et al. (2007) eq. 10
    a *= 1e-3           # +3 for m-1 to km-1 and -6 for microSiemen
    sig = att/a
    # calculate the temperature
    T = 1/(1/Tr-(k/E0)*np.log(sig/sig0))
    T -= 273.15
    return T

###############################################################################    

### Karlsson Continuity Method ###
    
# Based on Karlsson et al. (2012)
# This method gives a value for the continuity of radar layers
# P -- uncorrected power
# s_ind -- surface pick index
# b_ind -- bed pick index
# cutoff_ratio -- assigns the number of samples that are removed from top and bottom of the trace

def continuityKarlsson(P,s_ind,b_ind,lat,lon,cutoff_ratio,win=20,uice=168.,eps=3.2):
    # empty continuity index array     
    cont = np.empty_like(b_ind).astype(float)
    cont[:] = np.nan
    # calculate the continuity index for each trace
    for tr in range(len(P[0])):
        spick=int(s_ind[tr])
        bpick=int(b_ind[tr])
        if bpick-spick<10 or bpick>len(P[:,0]) or np.isnan(bpick-spick):
            continue
        else:
            # get data from between the surface and bed
            p_ext=P[spick:bpick,tr]
            # cutoff based on the assigned ratio
            cut=int(len(p_ext)/cutoff_ratio)
            p_ext=p_ext[cut:-cut]
            if np.any(~np.isfinite(p_ext)):
                continue
            # calculate the continuity index based on Karlsson et al. (2012) eq. 1
            cont[tr]=np.mean(abs(np.gradient(p_ext)))
    # smoother--simple moving boxcar
    cont_filt = np.convolve(cont, np.ones((win,))/win, mode='valid')

    return cont,cont_filt

###############################################################################    

### Slope Gradient ###
    
# The layer tilt will affect the returned power (e.g. Holschuh et al. 2014)
# This function calculates the gradient of slopes
# where the slope is (dz/dx)
# and the gradient of slopes is d(dz/dx)/dz

def layerSlopeGradient(x,z,win):
    # Calculate the Slope (dz/dx) of each line
    slope = np.gradient(z,x,axis=1)
    # create empty arrays for filling
    Dslope = np.array([])
    Derr = np.array([])
    # Calculate the change in slope with depth
    for tr in range(len(z[0])+1-win):
        # grab the data within the window
        Y = slope[:,tr:tr+win]
        X = z[:,tr:tr+win]
        # remove nan values
        idx = ~np.isnan(Y) & ~np.isnan(X)
        Y = Y[idx]
        X = X[idx]
        if len(Y)<5:
            Dslope = np.append(Dslope,np.nan)
            Derr = np.append(Derr,np.nan)
        else:
            # linear fit with depth
            p = np.polyfit(X,Y,1,cov=True)
            Dslope = np.append(Dslope,abs(p[0][0])*1000.)   # *1000. for m-1 to km-1
            Derr = np.append(Derr,np.sqrt(p[1][0,0])*1000.)   # *1000. for m-1 to km-1
    return Dslope,Derr