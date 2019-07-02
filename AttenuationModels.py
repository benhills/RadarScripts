#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:15:22 2018

@author: benhills
"""

import numpy as np
from RadarFunctions import Spreading

# -----------------------------------------------------------------------------------------------------

def attenuationMatsuoka(P,z,win,eps=3.2,h=0.,refraction=False,spreading=True):
    """
    Matsuoka Method

    Based on Matsuoka et al. (2010)
    This method fits a line to the measured power for internal reflectors (in log space)

    Parameters
    ---------
    P:          uncorrected power (dB), 2-d array
    z:          distance between the surface and the picked layer, 2-d array
    win:        window size (i.e. using a bigger window will include more traces in the least-squares fit)

    Output
    ---------
    N_out:      a 1-d array for attenuation rate in dB/km

    """
    # correct for spreading
    if spreading:
        Pc = P + Spreading(z,eps=eps,h=h,refraction=refraction)
    else:
        Pc = P
    # create an empty array
    N_out = np.array([])
    Nerr_out = np.array([])
    b_out = np.array([])
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
                b_out = np.append(b_out,p[0][1])
            except:
                N_out = np.append(N_out,np.nan)
                Nerr_out = np.append(Nerr_out,np.nan)
                b_out = np.append(b_out,np.nan)
    return N_out,Nerr_out,b_out

# -----------------------------------------------------------------------------------------------------

def attenuationJacobel(P,H,eps=3.12,h=0.,refraction=False,spreading=True):
    """
    ### Jacobel Method ###

    # Based on Jacobel et al. (2009)
    # This method fits a line to the to the measured power from the basal reflector (in log space)
    # P, uncorrected power, and H, thickness, are 1-d arrays
    # returns a number for attenuation rate in dB/km
    # for depth normalization see pg 12 first part of right column
        # normalized to a constant depth by multiplying by the square of the
        # ratio of depth to the shallowest depth observed,
        # effectively removing the inverse-square losses due to geometric spreading

    """
    # correct for geometric spreading (see description above)
    # TODO: Should there be a factor of two in the depth ratio?
    # Bob's language on this is confusing.
    # correct for spreading
    if spreading:
        Pc = P + Spreading(H,eps=eps,h=h,refraction=refraction)
    else:
        Pc = P
    # remove nan values
    idx = ~np.isnan(Pc) & ~np.isnan(H)
    y = Pc[idx]
    x = H[idx]
    # fit a line for thickness and power
    p = np.polyfit(x,y,1,cov=True)
    N = -p[0][0]*1000./2.            # *1000./2. for m-1 to km-1 and for one-way attenuation
    Nerr = np.sqrt(p[1][0,0])*1000./2.
    return N,Nerr

# -----------------------------------------------------------------------------------------------------

def attenuationChristianson(power,power_mult,H,Risw,Rfa):
    """
    ### MacGregor Method ###

    # Based on MacGregor (2011) and Christianson et al. (2016)
    # This method fits a line to the to the measured power from the basal reflector (in log space)
    # P, uncorrected power, and H, thickness, are 1-d arrays
    # returns a number for attenuation rate in dB/km
    """
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

# -----------------------------------------------------------------------------------------------------

def attenuationSchroeder(P,H,Nmax,dN=1,Nh_target=1.,Cw=0.1,win_init=5,win_step=10,
                         eps=3.2,h_spread=0.,refraction=False,spreading=True):
    """
    Schroeder Method

    TODO: Error and test

    Based on Schroeder et al. (2016)
    This method minimizes the correlation coeffiecient between attenuation
    rate and ice thickness
    Assumes that the reflectivity of the bed is constant

    Parameters
    ----------
    P:      uncorrected power
    H:      thickness
    Nmax:   maximum attenuation
    dN:     step size for attenuation guesses
    Nh_target:
    Cw:
    win_init:
    win_step:
    eps:
    h_spread:
    refraction:
    spreading:

    Output
    ---------
    N_out:      1-d array for attenuation rate in dB/km
    win_out:    1-d array, resulting window size for optimized attenuation rate
    """

    # Load the power of picked reflectors
    if spreading:
        # correct for spherical spreading, Schroeder et al. (2016) eq. 1
        Pc = P + Spreading(H,eps=eps,h=h_spread,refraction=refraction)
    else:
        Pc = P

    # Create empty arrays to fill for the output attenuation rate and window size
    N_out = np.zeros_like(Pc).astype(float)
    win_out = np.zeros_like(Pc)

    # Possible values for the attenuation rate
    N = np.arange(0,Nmax,dN)

    # Loop through all the traces
    for n in range(len(P)):
        # zero out the correlation coefficient array
        C = np.zeros_like(N)
        # Initial window size
        win = win_init
        # Radiometric Resolution (needs to converge to Nh_target)
        Nh = Nh_target + 1.
        while Nh > Nh_target and win/2<=n and win/2<=(len(H)-n):
            # thickness and power in the window
            h = H[n-win/2:n+win/2]/1000.    # divide by 1000 for m-1 to km-1
            pc = Pc[n-win/2:n+win/2]
            # loop through all the possible attenuation rates
            for j in range(len(N)):
                # attenuation-corrected power, Schroeder et al. (2016) eq. 4
                pa = pc + 2.*h*N[j]
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

# -----------------------------------------------------------------------------------------------------


def attenuationSchroederVertical(H,P,att_ds,N_max=20,dN=.1,Nh_target=3.,win_init=100,win_step=20,
                        Cw=0.1,eps=3.2,h_spread=0.,refraction=False,spreading=True):
    """
    Schroeder Method

    TODO: Error and test

    Based on Schroeder et al. (2016)
    This method minimizes the correlation coeffiecient between attenuation
    rate and ice thickness
    Assumes that the reflectivity of the bed is constant

    Parameters
    ----------
    H:      thickness (or depth of internal layer)
    P:      uncorrected power
    att_ds: depths at which to center attenuation calculations
    Nmax:   maximum attenuation
    dN:     step size for attenuation guesses
    Nh_target:
    win_init:
    win_step:
    Cw:
    eps:
    h_spread:
    refraction:
    spreading:

    Output
    ---------
    N_out:      1-d array for attenuation rate in dB/km
    Nh_out:      1-d array for attenuation rate error in dB/km
    win_out:    1-d array, resulting window size for optimized attenuation rate
    """


    # Load the power of picked reflectors
    if spreading:
        # correct for spherical spreading, Schroeder et al. (2016) eq. 1
        Pc = P + Spreading(H,eps=eps,h=h_spread,refraction=refraction)
    else:
        Pc = P

    # Create empty arrays to fill for the output attenuation rate and window size
    N_out = np.zeros_like(att_ds).astype(float)
    Nh_out = np.zeros_like(att_ds).astype(float)
    win_out = np.zeros_like(att_ds)

    # possible attenuation rates
    N = np.arange(0,N_max+dN,dN).astype(float)

    # Loop through all the desired depths
    for i in range(len(att_ds)):
        # current depth for attenuation calc
        att_d = att_ds[i]
        # Correlation Coefficient (starts empty)
        C = np.empty_like(N)
        # Initial window size
        win = win_init
        # Radiometric Resolution (needs to converge to Nh_target)
        Nh = Nh_target + 1.
        while Nh > Nh_target and att_d-win>=np.min(H) and att_d+win<=np.max(H):
            # thickness and power in the window
            idx = np.argwhere(abs(H-att_d)<win)
            hi = H[idx]/1000.    # divide by 1000 for m-1 to km-1
            pi = Pc[idx]
            # loop through all the possible attenuation rates
            for n in range(len(N)):
                # attenuation-corrected power, Schroeder et al. (2016) eq. 4
                pa = pi + 2.*hi*N[n]
                # calculate the correlation coefficient, Schroeder et al. (2016) eq. 5
                sum1 = np.nansum((hi-np.nanmean(hi))*(pa-np.nanmean(pa)))
                sum2 = np.sqrt(np.nansum((hi-np.nanmean(hi))**2.))
                sum3 = np.sqrt(np.nansum((pa-np.nanmean(pa))**2.))
                if np.any(np.isnan([sum1,sum2,sum3])):
                    C[n] = np.nan
                else:
                    C[n] = abs(sum1/(sum2*sum3))
            # Whichever value has the lowest correlation coefficient is chosen
            Cm = np.min(C)
            Nm = N[C==Cm]
            C0 = C[N==0]
            if Cm < Cw and C0 > Cw:
                Nh = np.max(N[C<Cw])-np.min(N[C<Cw])
            else:
                Nh = np.nan
            # get ready for the next iteration
            win += win_step
        # output
        N_out[i] = Nm
        Nh_out[i] = Nh
        win_out[i] = win
    return N_out, Nh_out, win_out
