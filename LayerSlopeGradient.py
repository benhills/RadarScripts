#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np

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

