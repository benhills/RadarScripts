#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 2020

@author: benhills
"""

import numpy as np

# ------------- From Jordan et al. 2019 ----------------------- #

def P(delta):
    """
    propagation/transmission matrix
    Jordan et al. (2019)

    Parameters
    ---------
    δ:     two-way birefringent phase shift
    """
    return np.array([[np.exp(1j*delta),0],
                     [0,1.]])

def Gamma(r):
    """
    Reflection matrix
    Jordan et al. (2019)

    Parameters
    ---------
    r:     ratio of reflection coefficients
    """
    return np.array([[r,0],
                     [0,1.]])

def R(alpha):
    """
    2-D rotation matrix
    Jordan et al. (2019)

    Parameters
    ---------
    alpha:     rotation angle between measurement axis and optic axis (ϵ)
    """
    return np.array([[np.cos(alpha),np.sin(alpha)],
                     [-np.sin(alpha),np.cos(alpha)]])

def S(alpha,delta,r):
    """
    Sinclair scattering matrix
    Jordan et al. (2019) eq. (7)

    Parameters
    ---------
    alpha:     rotation angle between measurement axis and optic axis
    delta:     two-way birefringent phase shift
    r:     ratio of reflection coefficients
    """

    M = np.matmul(R(alpha),P(delta/2.))
    M = np.matmul(M,Gamma(r))
    M = np.matmul(M,P(delta/2.))
    M = np.matmul(M,np.transpose(R(alpha)))

    return M

def rot_shift(S,theta):
    """
    Azimuthal (rotational) shift of principal axes
    at the transmitting and receiving antennas
    Mott, 2006

    Parameters
    --------
    S : array
        2-d array with [[shh,svh][shv,svv]] of complex numbers
    theta : complex
            rotational offset
    """

    shh = S[0,0]
    svh = S[0,1]
    shv = S[1,0]
    svv = S[1,1]

    S_ = np.empty_like(S)
    S_[0,0] = shh*np.cos(theta)**2.+(svh+shv)*np.sin(theta)*np.cos(theta)+svv*np.sin(theta)**2
    S_[0,1] = shv*np.cos(theta)**2.+(svv-shh)*np.sin(theta)*np.cos(theta)-svh*np.sin(theta)**2
    S_[1,0] = svh*np.cos(theta)**2.+(svv-shh)*np.sin(theta)*np.cos(theta)-shv*np.sin(theta)**2
    S_[1,1] = svv*np.cos(theta)**2.-(svh+shv)*np.sin(theta)*np.cos(theta)+shh*np.sin(theta)**2

    return S_

def phase_shift(z,freq=200e6,eps_bi=0.00354,eps=3.15,c=3e8):
    """
    Two-way phase shift
    Jordan et al. (2019)

    Parameters
    ---------
    z:
    freq:
    eps_bi:
    eps:
    c:

    """
    delta = 4.*np.pi*freq/c*(z*eps_bi/(2.*np.sqrt(eps)))
    return delta

def coherence(s1,s2):
    """
    phase correlation between two elements of the scattering matrix
    Jodan et al. (2019) eq. 13

    Parameters
    ---------
    s1:
    s2:
    """
    top = np.dot(s1,np.conj(s2))
    bottom = np.sqrt(np.abs(s1)**2.)*np.sqrt(np.abs(s2)**2.)
    c = top/bottom
    return c
