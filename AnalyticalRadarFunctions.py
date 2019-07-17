#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
import sys

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
