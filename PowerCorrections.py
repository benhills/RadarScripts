#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------

def refractiveFocusing(z1,z2,eps1,eps2):
    """
    Refractive focusing at an interface
    Dowedswell and Evans eq. 5

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

def Spreading(z,eps=3.12,d_eps=0.,h=0.,refraction=False):
    """
    Geometrical spreading correction for radar power.
    Optionally includes refractive focusing
    Dowedswell and Evans eq. TODO: look this up

    Parameters
    ---------
    z:      array or scalar     depth of desired output locations (m)
    eps:    array or scalar     permittivity (relative)
    d_eps:    array or scalar     depths for permittivity boundaries
    h:      scalar              height of aircraft

    Output
    ---------
    loss:   array or scalar     spreading loss (dB)
    """

    # geometric spreading correction
    spherical = (2.*z)**2.

    if refraction:
        # permittivity profile from Kravchenko (2004) JGlac.
        d_eps = np.arange(0,round(np.nanmax(z))+1,.1)
        if eps=='SPL':
            n = 1.324+5.2854*(d_eps/1000.)-21.865*(d_eps/1000.)**2.+39.058*(d_eps/1000.)**3.
            eps = n**2.
            eps[eps>3.15]=3.15

        # refractive focusing correction
        if hasattr(eps,"__len__"):
            q = np.ones_like(d_eps).astype(float)
            for i in range(1,len(eps)):
                qadd = refractiveFocusing(d_eps[i],d_eps-d_eps[i],eps[i-1],eps[i])
                q*=qadd
            qinterp = interp1d(d_eps,q)
        else:
            q = refractiveFocusing(h,d_eps,1.,eps)
            qinterp = interp1d(d_eps,q)

        # include refractive losses
        loss = 10.*np.log10(spherical/qinterp(z))
    else:
        # purely spherical spreading
        loss = 10.*np.log10(spherical)

    return loss,qinterp
