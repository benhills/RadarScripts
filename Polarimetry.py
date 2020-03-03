#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue March 3 2020

@author: benhills
"""

import numpy as np

# ------------- From Matsuoka et al. 2009 ----------------------- #

def R(ψ):
    return np.array([[np.cos(ψ),np.sin(ψ)],[-np.sin(ψ),np.cos(ψ)]])

def T(m1,m2,l,f,c=3e8,ϵ0=8.85e-12,μ0=1.257e-6):
    k = c*np.sqrt(ϵ0*μ0)
    T1 = np.exp(2j*np.pi*m1*k*l*f/c)
    T2 = np.exp(2j*np.pi*m2*k*l*f/c)
    return np.array([[T1,np.zeros_like(l)],[np.zeros_like(l),T2]])

def single_pole_indicatrix(θ,ψ,θw,ψw,χ=[0,0,1],mr_perp=np.sqrt(3.152),mr_par=np.sqrt(3.189)):

    m1 = (1-χ[0])*mr_perp + χ[0]*mr_par
    m2 = (1-χ[1])*mr_perp + χ[1]*mr_par
    m3 = (1-χ[2])*mr_perp + χ[2]*mr_par
    m_par = m3
    m_perp = m1

    ψ_1 = np.arctan(np.sin(θ)*np.sin(ψ-ψw)/(np.cos(θw)*np.sin(θ)*np.cos(ψ-ψw)-np.sin(θw)*np.cos(θ)))
    θ_1 = np.arccos(np.cos(θw)*np.cos(θ)+np.sin(θw)*np.sin(θ)*np.cos(ψ-ψw))
    ψ_2 = np.arctan(np.sin(θ)*np.sin(ψ-ψw)/(np.cos(θw)*np.sin(θ)*np.cos(ψ-ψw)+np.sin(θw)*np.cos(θ)))
    θ_2 = np.arccos(np.cos(θw)*np.cos(θ)-np.sin(θw)*np.sin(θ)*np.cos(ψ-ψw))

    m1_1 = m_par*m_perp/np.sqrt(m_par**2.*np.cos(θ_1)**2.+m_perp**2.*np.sin(θ_1)**2.)
    m1_2 = m_par*m_perp/np.sqrt(m_par**2.*np.cos(θ_2)**2.+m_perp**2.*np.sin(θ_2)**2.)
    m2 = m_perp

    return ψ_1,θ_1,m1_1,ψ_2,θ_2,m1_2,m2

def vertical_girdle_indicatrix(θw,ψw,χ=[0,0,1],mr_perp=np.sqrt(3.152),mr_par=np.sqrt(3.189)):

    m1 = (1-χ[0])*mr_perp + χ[0]*mr_par
    m2 = (1-χ[1])*mr_perp + χ[1]*mr_par
    m3 = (1-χ[2])*mr_perp + χ[2]*mr_par

    A = (np.cos(ψw)**2./(m1**2.)+np.sin(ψw)**2./(m2**2.))*np.cos(θw)**2.+np.sin(θw)**2./(m3**2.)
    B = -(1./(m1**2.)-1./(m2**2.))*np.cos(θw)*np.sin(2.*ψw)
    C = np.sin(ψw)**2./(m1**2.)+np.cos(ψw)**2./(m2**2.)

    ψ_1 = 1/2.*np.arctan(B/(A-C))
    m1 = np.sqrt(2./(A+C+np.sqrt(B**2.+(A-C)**2.)))
    m2 = np.sqrt(2./(A+C-np.sqrt(B**2.+(A-C)**2.)))

    return ψ_1,m1,m2

def birefringent_losses(θw,ψw,d,f,
                        fabric='single-pole',
                        Wr = np.array([1,0]),
                        Wt = np.array([1,0]),
                        S = np.array([[-1,0],[0,-1]]),
                        θ=0.,ψ=0.,χ=[0,0,1],
                        c=3e8,ϵ0=8.85e-12,μ0=1.257e-6,
                        mr_perp=np.sqrt(3.152),mr_par=np.sqrt(3.189)):

    l = d/np.cos(θw)

    if fabric == 'single-pole':
        ψ_1,θ_1,m1_1,ψ_2,θ_2,m1_2,m2 = single_pole_indicatrix(θ,ψ,θw,ψw)
    elif fabric == 'vertical-girdle':
        ψ_1,m1_1,m2 = vertical_girdle_indicatrix(θw,ψw,χ)
        ψ_2 = ψ_1
        m1_2 = m1_1
    else:
        raise TypeError('Fabric not recognized; choose from single-pole or vertical-girdle.')

    Amp = np.matmul(Wr,R(-ψ_2))
    Amp = np.matmul(Amp,T(m1_2,m2,l,f))
    Amp = np.matmul(Amp,R(ψ_2))
    Amp = np.matmul(Amp,S)
    Amp = np.matmul(Amp,R(-ψ_1))
    Amp = np.matmul(Amp,T(m1_1,m2,l,f))
    Amp = np.matmul(Amp,R(ψ_1))
    Amp = np.matmul(Amp,np.transpose(Wt))

    return np.real(10.*np.log10(Amp**2.))

