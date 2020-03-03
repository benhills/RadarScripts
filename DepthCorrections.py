#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
from PermittivityModels import firnPermittivity

# ----------------------------------------------------------------------------

### Create a function for transfering time to depth

def timeToDepth(time,zs,rhofs,c=300.,dt=0.001,tol=1e-5,everyTime=False):
    # get the input velocity depth profile
    eps = np.real(firnPermittivity(rhofs))
    vels = c/np.sqrt(eps)
    # iterate over time, moving down according to the velocity at each step
    z_out = np.zeros_like(time)
    if np.mean(np.gradient(time)) > dt+tol or everyTime == True:
        zhold = np.array([])
        thold = np.array([])
        tf = np.nanmax(time)+dt
        t = 0.
        z = 0.
        i=0
        while t < tf-tol:
            zhold = np.append(zhold,z)
            thold = np.append(thold,t)
            v = vels[np.argmin(abs(zs-z))]
            z += v*dt/2.
            t += dt
            i += 1
        for i in range(len(time)):
            z_out[i] = zhold[np.argmin(abs(thold-time[i]))]
    elif not hasattr(time,"__len__"):
        for i in range(len(time)):
            tf = time[i]
            print(i,'of',len(time))
            t = 0.
            z = 0.
            while t < tf:
                v = vels[np.argmin(abs(zs-z))]
                z += v*dt/2.
                t += dt
            z_out = z
    elif np.mean(np.gradient(time)) <= dt+tol:
        tf = time[-1]
        t = 0.
        z = 0.
        i=0
        while t < tf-tol:
            z_out[i] = z
            v = vels[np.argmin(abs(zs-z))]
            z += v*dt/2.
            t += dt
            i += 1
    else:
        print('no good')

    return z_out
