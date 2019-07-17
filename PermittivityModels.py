#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

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
