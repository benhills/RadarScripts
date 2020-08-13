#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 2020

@author: benhills
"""

import numpy as np

# ------------- From Doake et al. 2003 ----------------------- #

def P(δ):
    """
    phase-shift matrix
    Doake et al. (2003) pg. 6

    Parameters
    ---------
    δ:     two-way birefringent phase shift
    """
    return np.array([[1,0],
                     [0,np.exp(1j*δ)]])

def S(r):
    """
    target scattering matrix
    Doake et al. (2003) pg. 6

    Parameters
    ---------
    r:     ratio of reflection coefficients
    """
    return np.array([[1,0],
                     [0,r]])

def R(θ):
    """
    rotation matrix
    Doake et al. (2003) pg. 6

    Parameters
    ---------
    θ:     rotation angle
            can be between measurement axis and optic axis (ϵ)
            or between optic axis and reflection axis (γ)
    """
    return np.array([[np.cos(θ),np.sin(θ)],
                     [-np.sin(θ),np.cos(θ)]])

def Sm(ϵ,δ,γ,r):
    """
    Sinclair scattering matrix
    Doake et al. (2003) eq. (5)

    Parameters
    ---------
    ϵ:     rotation angle between measurement axis and optic axis
    δ:     two-way birefringent phase shift
    γ:     rotation angle between optic axis and reflection axis
    r:     ratio of reflection coefficients
    """

    M = np.matmul(R(ϵ),P(δ/2.))
    M = np.matmul(M,R(γ))
    M = np.matmul(M,S(r))
    M = np.matmul(M,R(-γ))
    M = np.matmul(M,P(δ/2.))
    M = np.matmul(M,R(-ϵ))

    return M

