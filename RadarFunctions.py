#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
from scipy.io import loadmat
import sys

# ----------------------------------------------------------------------------

def loadStoMigData(fname,uice=168.,CReSIS=False,datatype='mig'):
    """
    ### Load STO radar data ###
    """

    mfile = loadmat(fname)

    # data file variables
    try:
        data = mfile['migdata']              # the actual radar data
    except:
        print('No migrated data, using input data type:',datatype)
        data = mfile[datatype+'data']
    
    # grab the migrated data from the first slot
    if np.shape(data)[1] == 2:
        data = data[0]['data'][0]

    if len(mfile['travel_time'])>1:
        surface = mfile['elev'][0]               # elevation of the surface
        time = mfile['travel_time'][:,0]          # travel times for any one trace
        # Coordinates
        lat = mfile['lat'][0]
        lon = mfile['long'][0]        
    else:
        surface = mfile['elev'][:,0]               # elevation of the surface
        time = mfile['travel_time'][0]          # travel times for any one trace
        # Coordinates
        lat = mfile['lat'][:,0]
        lon = mfile['long'][:,0]

    ### Calculations ###
    # Conversions
    vdist = time*uice/2.                  # distance to the top of the picked wavelet (m)
    # calculate distance using haversine formula
    dlat=lat*np.pi/180.
    dlon=lon*np.pi/180.
    R = 6371000.                            # Radius of the earth (m)
    a = np.sin((dlat-dlat[0])/2.)**2.+np.cos(dlat[0])*np.cos(dlat)*np.sin((dlon-dlon[0])/2.)**2.
    dist = 2.*R*np.arcsin(np.sqrt(a))

    return data,surface,time,dist,vdist



def loadStoPickData(fname,uice=168.,CReSIS=False):
    """
    ### Load STO pick data ###
    """
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

    return ppower,psamp0.astype(int),psamp1.astype(int),psamp2.astype(int),pdist,lat,lon,x_coord,y_coord,dist,pnum

# ----------------------------------------------------------------------------

def Fresnel(n1,n2,theta_i):
    """
    ### Fresnel's Equations ###

    # Assuming that the materials are not magnetic
    """

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

# ----------------------------------------------------------------------------

def attTemp(T,Hplus=1.3,Clminus=4.2):
    """
    ### Attenuation rate to temperature ###

    # See MacGregor et al. 2007
    # TODO: move the constants to another script so that they are standardized
    # TODO: This is for pure ice. I need to rethink for different ion concentrations
    """

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
    T = 1./(1./Tr-(k/E0)*np.log(sig/sig0))
    T -= 273.15
    return T

# ----------------------------------------------------------------------------

def continuityKarlsson(P,s_ind,b_ind,lat,lon,cutoff_ratio,win=20,uice=168.,eps=3.2):
    """
    Karlsson Continuity Method

    Based on Karlsson et al. (2012)
    This method gives a value for the continuity of radar layers

    Parameters
    ----------
    P:              uncorrected power
    s_ind:          surface pick index
    b_ind:          bed pick index
    cutoff_ratio:   assigns the number of samples that are removed from top and bottom of the trace

    Output
    ---------
    cont:
    cont_filt:

    """


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

# ----------------------------------------------------------------------------

def layerSlopeGradient(x,z,win):
    """
    ### Slope Gradient ###

    # The layer tilt will affect the returned power (e.g. Holschuh et al. 2014)
    # This function calculates the gradient of slopes
    # where the slope is (dz/dx)
    # and the gradient of slopes is d(dz/dx)/dz
    """

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


# ----------------------------------------------------------------------------
    
def snowPermittivity(rho,m=0.,fs=500e6,fw=9.07e9):
    """
    Calculate the dielectric permittivity of snow
    Kendra et al. (1998), IEEE
    
    Parameters
    ---------
    rho:        snow density       (g/cm3)
    m:          snow wetness        (%)
    fs:         radar frequency     (Hz)
    fw:         relaxation frequency of water at 0C
    
    Output
    ---------
    eps_ds:     snow permittivity
    """
    
    # permittivity of dry snow, Kendra eq. 13
    eps_ds = 1. + 1.7*rho + 0.7*rho**2.
    # added permittivity from wetness, Kendra eq. 14
    Deps_s = 0.02*m**1.015+(.073*m**1.31)/(1+(fs/fw))
    
    eps_ds+=Deps_s
    
    return eps_ds


def firnPermittivity(rhof,rhoi=917.,epsi_real=3.12,epsi_imag=-9.5):
    """
    Calculate the dielectric permittivity of firn with the DECOMP mixing model
    Wilhelms (2005), GRL
    
    Parameters
    ---------
    rhof:           firn density       (kg/m3)
    rhoi:           firn density       (kg/m3)
    epsi_real:      real permittivity of ice (relative)
    epsi_imag:      imaginary permittivity of ice (relative)

    Output
    ---------
    eps_f:          firn permittivity   (relative)
    """
    
    # Wilhelms (2005), end of section 2
    lhs = 1. + (rhof/rhoi)*((epsi_real-1j*epsi_imag)**(1/3.)-1)
    eps_f = lhs**3.

    return eps_f

# ----------------------------------------------------------------------------

def refractiveFocusing(z1,z2,eps1,eps2):
    """ 
    Refractive focusing at an interface
    Dowedswell and Evans eq. TODO: look this up
    
    Parameters
    ---------
    z1:     scalar      Thickness above interface (m)
    z2:     scalar      Thickness below interface (m)
    eps1:   scalar      Permittivity above interface (relative)
    eps2:   scalar      Permittivity below interface (relative)
    
    Output
    ---------
    q:      scalar              refractive focusing coefficient
    """
    q = ((z1+z2)/(z1+z2*np.sqrt(eps1/eps2)))**2.
    q[z2 <= z1] = 1.
    return q

def Spreading(z,eps=3.12,h=0.,refraction=False):
    """
    Geometrical spreading correction for radar power.
    Optionally includes refractive focusing 
    Dowedswell and Evans eq. TODO: look this up
    
    Parameters
    ---------
    z:      array or scalar     depth (m)
    eps:    array or scalar     permittivity (relative)
    h:      scalar              height of aircraft
    
    Output
    ---------
    loss:   array or scalar     spreading loss (dB)
    """
    spherical = (2.*z)**2.
    # refractive spreading correction
    if hasattr(eps,"__len__"):
        q = np.ones_like(z).astype(float)
        for i in range(1,len(eps)):
            qadd = refractiveFocusing(z[i],z-z[i],eps[i-1],eps[i])
            q*=qadd    
    else:
        q = refractiveFocusing(h,z,1.,eps)
    
    if refraction:
        # include refractive losses
        loss = 10.*np.log10(spherical/q)
    else:
        # purely spherical spreading
        loss = 10.*np.log10(spherical)
    
    return loss