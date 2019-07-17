#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:19:47 2018

@author: benhills
"""

import numpy as np
from scipy.io import loadmat

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
