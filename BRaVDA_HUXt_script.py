# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:09:20 2023

@author: vy902033
"""

import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import os as os
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt

#from HUXt
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

#from HUXt_tools
#import huxt_ensembles as Hens

#from BRaVDA
import startBravda

#root directory for storing BRaVDA output
workingdir = os.environ['DBOX'] + 'Data\\BRaVDA\\'
runname = '7 June 2012 MAS'

#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'

#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
obs = 'ABC'

r_min_map = 30*u.solRad #BRaVDA and MAS inner boundary
r_min = 5*u.solRad #map data to this location

#target time. i.e. mid-point of the assimilation
target_time = datetime.datetime(2012,6,7)
starttime = target_time - datetime.timedelta(days=12)
endtime = starttime + datetime.timedelta(days=27.27)

outputpath = workingdir + runname + '\\'

# <codecell>

#helper functions
def _zerototwopi_(angles):
    """
    Function to constrain angles to the 0 - 2pi domain.

    Args:
        angles: a numpy array of angles.
    Returns:
            angles_out: a numpy array of angles constrained to 0 - 2pi domain.
    """
    twopi = 2.0 * np.pi
    angles_out = angles
    a = -np.floor_divide(angles_out, twopi)
    angles_out = angles_out + (a * twopi)
    return angles_out


# <codecell>

#==============================================================================
#==============================================================================
#Run startBravda
#==============================================================================
#==============================================================================


startBravda.bravdafunction(endtime, obsToUse = obs , usecustomens = False,
                           runoutputdir = outputpath, plottimeseries = True )


# <codecell>

#read in the posterior solution 
smjd = int(Time(starttime).mjd)
actual_startdate = Time(smjd, format = 'mjd').datetime
#post_filepath = glob.glob(outputpath + '\\posterior\\posterior_MJDstart*')
#posterior = np.loadtxt(post_filepath[0]) #THIS CAN FIND THE WRONG FILE. USE MJD START 
posterior = np.loadtxt(outputpath + '\\posterior\\posterior_MJDstart' + str(smjd) +'.txt')
prior = np.loadtxt(outputpath + '\\prior\\prior_MJDstart' + str(smjd) +'.txt')

#the inner boundary value is given as a function of time from the inition time
post_inner = posterior[0,:]
prior_inner = prior[0,:]
#convert from function of time to function of longitude
post_vlong = np.flipud(post_inner)
prior_vlong = np.flipud(prior_inner)

#find the associated carr_longs
cr, cr_lon_init = Hin.datetime2huxtinputs(actual_startdate)
#resample the ensemble to 128 longitude bins
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)

carrlongs = _zerototwopi_(phi128 + cr_lon_init.value)

#interpolate the posterior at the inner boundary to CarrLong grid
interp = interpolate.interp1d(carrlongs,
                              post_vlong, kind="nearest",
                              fill_value="extrapolate")

v_carrlong_post = uniform_filter1d(interp(phi128), size = 10) * u.km/u.s

interp = interpolate.interp1d(carrlongs,
                              prior_vlong, kind="nearest",
                              fill_value="extrapolate")
v_carrlong_prior = uniform_filter1d(interp(phi128), size = 10) * u.km/u.s


#map the DA values in to 5 rS
v_5_prior = Hin.map_v_boundary_inwards(v_carrlong_prior, r_min_map, r_min)
v_5_post = Hin.map_v_boundary_inwards(v_carrlong_post, r_min_map, r_min)

# <codecell> Solve HUXt for the posterior and prior



#trace a bunch of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad


plot_time = 12.8*u.day

#Now run HUXt with the posterior time series
model = H.HUXt(v_boundary=v_carrlong_post, cr_num=cr, cr_lon_init=cr_lon_init, latitude = 0*u.deg,
               simtime=27.27*u.day, dt_scale=4, r_min = r_min_map, frame = 'synodic')
model.solve([],streak_carr = lon_grid)
earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
#plot it
#fig, ax = plt.subplots(1)
#ax.plot(earth_series['time'],earth_series['vsw'])
HA.plot_earth_timeseries(model, plot_omni = True)

HA.plot(model, plot_time)


#and the inward-mapped v series
model = H.HUXt(v_boundary=v_5_post, cr_num=cr, cr_lon_init=cr_lon_init, latitude = 0*u.deg,
               simtime=27.27*u.day, dt_scale=4, r_min = r_min, frame = 'synodic')
model.solve([],streak_carr = lon_grid)
earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
#plot it
#fig, ax = plt.subplots(1)
#ax.plot(earth_series['time'],earth_series['vsw'])
HA.plot_earth_timeseries(model, plot_omni = True)

HA.plot(model, plot_time)


#get the time slice and export as csv
id_t = np.argmin(np.abs(model.time_out - plot_time))

lon_arr, dlon, nlon = H.longitude_grid()
lon, rad = np.meshgrid(lon_arr.value, model.r.value)
v_sub = model.v_grid.value[id_t, :, :].copy()


# Pad out to fill the full 2pi of contouring
pad = lon[:, 0].reshape((lon.shape[0], 1)) + model.twopi
lon = np.concatenate((lon, pad), axis=1)
pad = rad[:, 0].reshape((rad.shape[0], 1))
rad = np.concatenate((rad, pad), axis=1)
pad = v_sub[:, 0].reshape((v_sub.shape[0], 1))
v_sub = np.concatenate((v_sub, pad), axis=1)

#save the grid and data
np.savetxt(outputpath + "v_huxt.csv", v_sub, delimiter=",")
np.savetxt(outputpath + "lon_huxt.csv",lon, delimiter=",")
np.savetxt(outputpath + "r_huxt.csv",rad, delimiter=",")

#test plotting this
v_huxt = np.loadtxt(outputpath + "v_huxt.csv", delimiter = ',')
lon_huxt = np.loadtxt(outputpath + "lon_huxt.csv", delimiter = ',')
r_huxt = np.loadtxt(outputpath + "r_huxt.csv", delimiter = ',')

plotvmin = 250
plotvmax = 750
dv = 10
levels = np.arange(plotvmin, plotvmax + dv, dv)
 
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
cnt = ax.contourf(lon_huxt, r_huxt, v_huxt, levels=levels)




# #Now run HUXt with the prior time series
# model = H.HUXt(v_boundary=v_carrlong_prior, cr_num=cr, cr_lon_init=cr_lon_init, latitude = 0*u.deg,
#                simtime=27.27*u.day, dt_scale=4, r_min = r_min, frame = 'synodic')
# model.solve([],streak_carr = lon_grid)
# earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
# #plot it
# #fig, ax = plt.subplots(1)
# #ax.plot(earth_series['time'],earth_series['vsw'])
# HA.plot_earth_timeseries(model, plot_omni = True)

# HA.plot(model,13.7*u.day)