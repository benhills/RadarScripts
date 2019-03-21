#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:15:22 2018

@author: benhills
"""

import numpy as np

###############################################################################

### Matsuoka Method ###

# Based on Matsuoka et al. (2010)
# This method fits a line to the measured power for internal reflectors (in log space)
# P, uncorrected power, 2-d array
# z, distance between the surface and the picked layer, 2-d array
# win, window size (i.e. using a bigger window will include more traces in the least-squares fit)
# Returns a 1-d array for attenuation rate in dB/km

def attenuationMatsuoka(P,z,win,eps=3.2):
    # correct for spherical spreading
    #Pc = P + 20.*np.log10(2.*z/np.sqrt(eps))               # Matsuoka et al. (2010) eq. 2 (changed by a factor of 2, maybe a typo?)
    Pc=P
    # create an empty array
    N_out = np.array([])
    Nerr_out = np.array([])
    # calculate the attenuation rate for each desired trace (or window)
    for tr in range(len(z[0])+1-win):
        # grab the data within the window
        y = Pc[:,tr:tr+win]
        x = z[:,tr:tr+win]
        # remove nan values
        idx = ~np.isnan(y) & ~np.isnan(x)
        y = y[idx]
        x = x[idx]
        if len(y)<5:
            N_out = np.append(N_out,np.nan)
            Nerr_out = np.append(Nerr_out,np.nan)
        else:
            try:
                # linear fit with depth
                p = np.polyfit(x,y,1,cov=True)
                N_out = np.append(N_out,-p[0][0]*1000./2.)   # *1000. for m-1 to km-1 and /2. for one-way attenuation 
                Nerr_out = np.append(Nerr_out,np.sqrt(p[1][0,0])*1000./2.)
            except:
                N_out = np.append(N_out,np.nan)
                Nerr_out = np.append(Nerr_out,np.nan)
    return N_out,Nerr_out

###############################################################################

### Jacobel Method ###
    
# Based on Jacobel et al. (2009)
# This method fits a line to the to the measured power from the basal reflector (in log space)
# P, uncorrected power, and H, thickness, are 1-d arrays
# returns a number for attenuation rate in dB/km
# for depth normalization see pg 12 first part of right column
    # normalized to a constant depth by multiplying by the square of the 
    # ratio of depth to the shallowest depth observed, 
    # effectively removing the inverse-square losses due to geometric spreading
 
def attenuationJacobel(P,H):
    # correct for geometric spreading (see description above)
    # TODO: Should there be a factor of two in the depth ratio?
    # Bob's language on this is confusing.
    Pc = P + 20*np.log10(2.*H/np.nanmin(H))
    # remove nan values
    idx = ~np.isnan(Pc) & ~np.isnan(H)
    y = Pc[idx]
    x = H[idx]  
    # fit a line for thickness and power
    p = np.polyfit(x,y,1,cov=True)
    N = -p[0][0]*1000./2.            # *1000./2. for m-1 to km-1 and for one-way attenuation 
    Nerr = np.sqrt(p[1][0,0])*1000./2.
    return N,Nerr    

###############################################################################

### Christianson Method ###
    
# Based on Christianson et al. (2016)
# This method fits a line to the to the measured power from the basal reflector (in log space)
# P, uncorrected power, and H, thickness, are 1-d arrays
# returns a number for attenuation rate in dB/km
 
def attenuationChristianson(power,power_mult,H,Risw,Rfa):
    # convert all terms out of log space in order to use eq. A4
    Rfa = 10**(Rfa/10.)
    Risw = 10**(Risw/10.)
    P_mult = 10**(power_mult/10.)
    P = 10**(power/10.)
    # Calculate the attenuation length scale with Christianson et al. (2016) eq. A4
    La = -2.*H/np.log((4./(Risw*Rfa))*(P_mult/P))
    # Then attenuation rate is (following Jacobel et al. (2009))
    Na = 1000.*10.*np.log10(np.exp(1))/La
    return np.nanmean(Na),np.nanstd(Na),Na

###############################################################################    

### Schroeder Method ###
    
# TODO: Error and test
    
# Based on Schroeder et al. (2016)
# This method minimizes the correlation coeffiecient between attenuation
# rate and ice thickness
# Assumes that the reflectivity of the bed is constant
# Pc, corrected power, and H, thickness, are 1-d arrays
# win is the window size as an integer
# returns a 1-d array for attenuation rate in dB/km
    
def attenuationSchroeder(P,H,N_max,N_step=1,Nh_target=1.,Cw=0.1,win_init=5,win_step=10,eps=3.2):
    # correct for spherical spreading, Schroeder et al. (2016) eq. 1
    Pg = P + 20.*np.log10(2.*H/np.sqrt(eps))
    # Create empty arrays to fill for the output attenuation rate and window size
    N_out = np.zeros_like(Pg).astype(float)
    win_out = np.zeros_like(Pg)
    # Possible values for the attenuation rate 
    N = np.arange(0,N_max,N_step)
    # Loop through all the traces
    for n in range(len(P)):    
        # Correlation Coefficient
        C = np.zeros_like(N)
        # Initial window size       
        win = win_init
        # Radiometric Resolution (needs to converge to Nh_target)
        Nh = Nh_target + 1.
        while Nh > Nh_target and win/2<=n and win/2<=(len(H)-n):
            # thickness and power in the window
            h = H[n-win/2:n+win/2]/1000.    # divide by 1000 for m-1 to km-1
            pg = Pg[n-win/2:n+win/2]
            # loop through all the possible attenuation rates
            for j in range(len(N)):
                # attenuation-corrected power, Schroeder et al. (2016) eq. 4
                pa = pg + 2.*h*N[j]
                # calculate the correlation coefficient, Schroeder et al. (2016) eq. 5
                sum1 = sum((h-np.mean(h))*(pa-np.mean(pa)))
                sum2 = np.sqrt(sum((h-np.mean(h))**2.))
                sum3 = np.sqrt(sum((pa-np.mean(pa))**2.))
                C[j] = abs(sum1/(sum2*sum3))
            # Whichever value has the lowest correlation coefficient is chosen
            Cm = np.min(C)
            Nm = N[C==Cm]
            C0 = C[N==0]
            if Cm < Cw and C0 >Cw:
                Nh = np.max(N[C<Cw])-np.min(N[C<Cw])
            #print win,C0,Cm,Nh,Nm
            win += win_step
        N_out[n] = Nm
        win_out[n] = win
    return N_out,win_out