# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 08:54:19 2023

@author: mathewjowens

Some really ugly code to run BRaDVA with a user-defined WSA map, then run HUXt 
with a cone file, plus the original WSA map and teh BRaVDA modified WSA map

"""

import astropy.units as u
from astropy.time import Time, TimeDelta
import numpy as np
import datetime
import h5py
import os as os
import matplotlib.pyplot as plt
from astropy.io import fits
import shutil
from scipy import interpolate

#from HUXt
import huxt_inputs as Hin
import huxt as H
import huxt_analysis as HA

#from HUXt_tools
import huxt_ensembles as Hens

#from BRaVDA
import startBravda


forecasttime = datetime.datetime(2022,6,11, 12)

#input data
wsafilepath = os.environ['DBOX'] + 'python_repos\\HUXt_tools\\data\\models%2Fenlil%2F2022%2F6%2F11%2F12%2Fwsa.gong.fits'
conefilepath = os.environ['DBOX'] + 'python_repos\\HUXt_tools\\data\\models%2Fenlil%2F2022%2F6%2F11%2F12%2Fcone2bc.in'

#bravda ensemble directory
bravda_ens_dir = os.environ['DBOX'] + 'python_repos\\BRaVDA\\masEns\\'

#output directory
output_dir = os.environ['DBOX'] + 'python_repos\\HUXt_tools\\data\\test_bravda\\'

#which spacecraft to assimilate. A=STERA, B=STERB, C=ACE, All are currently assumed to be at Earth lat
run = 'AC'#, 'C', 'B', 'BC', 'ABC'
DAnow = True # run the DA now. Time consuming.
allplots = True #



#forecast tiem. i.e. time of last  in situ observation to be used.
buffertime_days = 5 # number of days ahead to start the run, to allow CME to propagate

#ensemble generation parameters
Nens = 500
lat_rot_sigma = 5*np.pi/180 *u.rad
lat_dev_sigma = 2*np.pi/180 *u.rad
long_dev_sigma = 2*np.pi/180 *u.rad

#HUXt run parameters
deacc = True # deaccelerate the WSA to 21.5 rS map prior to making the ensemble
r_min = 21.5*u.solRad
r_max = 240*u.solRad
simtime = 12*u.day

sigma = 10 #[10] width of the Gaussian for the Kalman filter function [degrees]


# <codecell> set up files, work out dates, load in WSA data, etc

#check the output directory exists
if not os.path.exists(output_dir):
    # Create the directory
    os.makedirs(output_dir)


# Extract the filename without extension
wsa_filename = os.path.splitext(os.path.basename(wsafilepath))[0]
cone_filename = os.path.basename(conefilepath)

#copy the original WSA FITS file to the output_dir
if not os.path.exists(output_dir + wsa_filename + '.fits'):
    shutil.copyfile(wsafilepath,
                    output_dir + wsa_filename + '.fits')

#copy the original CONE file FITS file to the output_dir
if not os.path.exists(output_dir +cone_filename):
    shutil.copyfile(conefilepath,
                    output_dir + cone_filename)

#start the HUXt runs a few days ahead of the forecast time
starttime = forecasttime  - datetime.timedelta(days=buffertime_days)
#BRaVDA rounds to nearest MJD
smjd = int(Time(starttime).mjd)
starttime = Time(smjd, format = 'mjd').datetime
fmjd = int(Time(forecasttime).mjd)
forecasttime = Time(fmjd, format = 'mjd').datetime

#get the huxt params for the start time
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)


#load the WSA data
wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(wsafilepath)
    
if deacc:
    #deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
    for nlat in range (1, len(vr_lats)):
        wsa_vr_map[nlat,:], lon_temp = Hin.map_v_inwards(wsa_vr_map[nlat,:], 215*u.solRad, 
                                                 vr_longs, r_min)
    


#Use the HUXt ephemeris data to get Earth lat over the CR
#========================================================
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   cr_num= cr, cr_lon_init = cr_lon_init, 
                   lon_out=0.0*u.deg,
                   r_min=r_min, r_max=r_max)

#retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
#get Earth lat as a function of longitude (not time)
E_lat = np.mean(earth.lat_c)
E_r = np.mean(earth.r)

reflats = np.interp(vr_longs,earth.lon_c,earth.lat_c)


# <codecell> generate WSA ensemble for the DA and put it in the BRaVDA dir
  
phi, theta = np.meshgrid(vr_longs, vr_lats, indexing = 'xy')

vr_ensemble = Hens.generate_input_ensemble(phi, theta, wsa_vr_map, 
                            reflats, Nens = Nens, 
                            lat_rot_sigma = lat_rot_sigma, 
                            lat_dev_sigma = lat_dev_sigma,
                            long_dev_sigma = long_dev_sigma)

#resample the ensemble to 128 longitude bins
vr128_ensemble = np.ones((Nens,128))  
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
for i in range(0, Nens):
    vr128_ensemble[i,:] = np.interp(phi128,
                  vr_longs.value,vr_ensemble[i,:])


#save the ensemble
var = 'vr'
if var == 'vr':
    h5f = h5py.File(output_dir + '\\WSA_CR' + str(cr) +'_vin_ensemble.h5', 'w')
    h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
elif var == 'br':
    h5f = h5py.File(output_dir + '\\WSA_CR' + str(cr) +'_bin_ensemble.h5', 'w')
    h5f.create_dataset('Bin_ensemble', data=vr128_ensemble)            
h5f.attrs['lat_rot_sigma'] = lat_rot_sigma
h5f.attrs['lat_dev_sigma'] = lat_dev_sigma
h5f.attrs['long_dev_sigma'] = long_dev_sigma
filepath = 'get_MAS_vrmap(cr)'  #this is used only to identify the source files. 
h5f.attrs['source_file'] = wsa_filename
h5f.attrs['r_in_rS'] = r_min
h5f.attrs['Carrington_rotation'] = cr
h5f.close()    


#also save vr128 to a .dat file for use in BRaVDA
outEnsTxtFile = open(f'{bravda_ens_dir}customensemble.dat', 'w')
np.savetxt(outEnsTxtFile, vr128_ensemble)
outEnsTxtFile.close()
x,y = np.meshgrid(vr_longs.value * 180/np.pi,vr_lats.value*180/np.pi)

if allplots:
    plt.figure()
    plt.pcolor(x, y, wsa_vr_map.value, vmin = 250, vmax=650)
    plt.title('Original WSA map')
    
# <codecell> Run BRaVDA

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



#output filepaths
outputpath = output_dir 
newfilename = wsa_filename + run + '.fits' 

#==============================================================================
#==============================================================================
#Run startBravda
#==============================================================================
#==============================================================================

if DAnow:
    startBravda.bravdafunction(forecasttime, obsToUse = run, usecustomens = True,
                               runoutputdir = outputpath, plottimeseries = True,
                               corona = 'WSA')


#read in the posterior solution to blend with the WSA map   
#BRaVDA files are labelled with teh start time, 27 days previous
posterior = np.loadtxt(outputpath + '\\posterior\\posterior_MJDstart' + str(fmjd-28) +'.txt')

#the inner boundary value is given as a function of time from the inition time
post_inner = posterior[0,:]
#convert to function of longitude
post_vlong = np.flipud(post_inner)
#post_vlong = post_inner
#find the associated carr_longs
cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
post_carrlongs = _zerototwopi_(phi128 + cr_lon_init.value)

#interpolate the posterior at the inner boundary to the WSA CarrLong grid
interp = interpolate.interp1d(post_carrlongs,
                              post_vlong, kind="nearest",
                              fill_value="extrapolate")
post_wsalongs = interp(vr_longs.value)

# Now distort the WSA solution on the basis of the DA at the SS using a Gaussain filter in lat
sigma_rad = sigma * np.pi/180
new_map = wsa_vr_map.value *np.nan
for nlong in range(0,len(vr_longs)):
    
    acelat = reflats[nlong]
    
    #compute the deltas at all latitudes
    delta_lat_ace = abs(vr_lats.value - acelat.value)
    
    #compute the guassian weighting at each lat
    weights_ace = np.exp( -delta_lat_ace*delta_lat_ace/(2*sigma_rad*sigma_rad))
    
    #merge the MAS and assimilated data
    new_map[:,nlong] = ((1-weights_ace)*wsa_vr_map[:,nlong].value +
        weights_ace*post_wsalongs[nlong])



#rotate the map back around to match the original WSA format
#===========================================================

#find the required angle of rotation
hdul = fits.open(output_dir + wsa_filename + '.fits')
cr_num = hdul[0].header['CARROT']
dgrid = hdul[0].header['GRID'] * np.pi / 180
carrlong = (hdul[0].header['CARRLONG']) * np.pi / 180
data = hdul[0].data
br_map_fits = data[0, :, :]
vr_map_fits = data[1, :, :]
hdul.flush() 
hdul.close()

# compute the Carrington map grids
vr_long_edges = np.arange(0, 2 * np.pi + 0.00001, dgrid)
vr_long_centres = (vr_long_edges[1:] + vr_long_edges[:-1]) / 2

vr_lat_edges = np.arange(-np.pi / 2, np.pi / 2 + 0.00001, dgrid)
vr_lat_centres = (vr_lat_edges[1:] + vr_lat_edges[:-1]) / 2

vr_longs = vr_long_centres * u.rad
vr_lats = vr_lat_centres * u.rad

# rotate the maps so they are in the Carrington frame
rot_vr_map = np.empty(vr_map_fits.shape)
for nlat in range(0, len(vr_lat_centres)):
    interp = interpolate.interp1d(_zerototwopi_(vr_long_centres - carrlong),
                                  new_map[nlat, :], kind="nearest",
                                  fill_value="extrapolate")
    rot_vr_map[nlat, :] = interp(vr_long_centres)

new_map = rot_vr_map



#make a copy of the original WSA FITS file
if not os.path.exists(output_dir + newfilename):
    shutil.copyfile(output_dir + wsa_filename + '.fits',
                    output_dir + newfilename)

#modify contents of the FITS file with the new map
hdul = fits.open(output_dir + newfilename, mode = 'update')
data = hdul[0].data
#paste in the new data
data[1, :, :] = new_map
hdul[0].data = data
hdul.flush() 
hdul.close()

#load it back in and plot
vr_map_fits, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
= Hin.get_WSA_maps(output_dir + newfilename )

if allplots:
    plt.figure()
    plt.pcolor(x,y,vr_map_fits.value , vmin = 250, vmax=650)
    plt.title('WSA + DA of ' + run)
    plt.colorbar()

# <codecell> Run HUXt with the no-DA and DA WSA maps


files = [wsa_filename + '.fits', newfilename]    
reflat = np.nanmean(reflats)


#read in and plot the HUXt input for each map  
#============================================ 
fig, (ax1, ax2) = plt.subplots(2)
for file in files:
    vr_in = Hin.get_WSA_long_profile(output_dir + file, lat = reflat.to(u.deg))
    dlong = 360/len(vr_in)
    longs = np.arange(0, 360, dlong)
    ax1.plot(longs, vr_in, label = file)
       
ymin, ymax = ax1.get_ylim()
cr_forecast, cr_lon_init_forecast = Hin.datetime2huxtinputs(forecasttime)
ax1.plot([cr_lon_init_forecast.to(u.deg).value, cr_lon_init_forecast.to(u.deg).value], [ymin,ymax],'r')
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)
ax1.plot([cr_lon_init.to(u.deg).value, cr_lon_init.to(u.deg).value], [ymin,ymax],'r--')
ax1.legend()
ax1.set_xlabel('Carrington longitude [deg]')
ax1.set_ylabel(r'$V_{SW}$ (21.5 rS) [km/s]') 
ax1.set_title('Earth lat = '+str(reflat.to(u.deg)))   

#get the HUXt parameters for the start time, allowing for prior CME propagation
cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)

#run the ambient solution for all runs
#============================================
for file in files:
    vr_in = Hin.get_WSA_long_profile(output_dir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=27.27*u.day, dt_scale=4, r_min = r_min, frame = 'sidereal')
    model.solve([])
    earth_series = HA.get_observer_timeseries(model, observer = 'Earth')
    ax2.plot(earth_series['time'],earth_series['vsw'], label = file)
    
    # if allplots:

    #     fig2, ax = HA.plot_earth_timeseries(model, plot_omni = False)
    #     ylims = ax[0].get_ylim()
    #     ax[0].plot([forecasttime, forecasttime], ylims, 'b')
        
ax2.legend()
ax2.set_ylabel(r'$V_{SW}$ (Earth) [km/s]') 


#Now add the CMEs and run HUXt
#=============================================

for file in files:
    vr_in = Hin.get_WSA_long_profile(output_dir + file, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                   simtime=simtime, dt_scale=4, r_min = r_min, frame = 'sidereal')
    cme_list = Hin.ConeFile_to_ConeCME_list(model, conefilepath)
        
    model.solve(cme_list) 
    fig3, ax = HA.plot_earth_timeseries(model, plot_omni = False)
    ylims = ax[0].get_ylim()
    ax[0].plot([forecasttime, forecasttime], ylims, 'b')
    
        
    cme_huxt = model.cmes[0]
    #  Print out some arrival stats
    print("**************************************")
    print(file)
    print(os.path.basename(conefilepath))
    
    stats = cme_huxt.compute_arrival_at_body('EARTH') 
    if stats['hit_id']:
        output_text = "Earth arrival:{},  Transit time:{:3.2f} days, Arrival longitude:{:3.2f}, Speed:{:3.2f}" 
        print(output_text.format(stats['t_arrive'].isot, stats['t_transit'], stats['lon'], stats['v']))

        #HA.plot(model, model.time_out[stats['hit_id']])
       
    