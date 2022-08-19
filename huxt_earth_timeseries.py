# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:01:10 2022

@author: mathewjowens
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pandas as pd

import huxt as H
import huxt_inputs as Hin



wsafilepath =  'D:\\Dropbox\\python_repos\\HUXt\\data\\example_inputs\\2022-02-24T22Z.wsa.gong.fits'
omnifilepath = 'D:\\Dropbox\\Data_hdf5\\omni_1hour.h5'

#simulate a whole CR in the synodic frame
simtime = 27.27*u.day




# load the data 
# =====================
wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(wsafilepath)
omni_1hour = pd.read_hdf(omnifilepath)

#Use the HUXt ephemeris data to get Earth lat and radius over the CR
#====================================================================

# set up a dummy model class to use for Earth ephemeris
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=simtime, 
                   cr_num= cr_fits, #cr_lon_init = cr_lon_init, 
                   lon_out=0.0*u.deg,  frame = 'synodic',
                   r_min=21.5*u.solRad, r_max=225*u.solRad)

#retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
#get average Earth lat and radius
E_lat = np.mean(earth.lat_c)
E_r = np.mean(earth.r)

#find the gridcell which corresponds to Earth for this period
iE_r = np.argmin(abs(dummymodel.r - E_r))
iE_lat = np.argmin(abs(vr_lats - E_lat))

#extract the time stemps from the dummy model and compute OMNI values
time = dummymodel.time_out + dummymodel.time_init
nt = len(time)
#convert huxt time to mjd
huxt_mjd = np.array([t.mjd for t in time])
#interpolate OMNI to HUXt time steps
omni_v = np.interp(huxt_mjd, omni_1hour['mjd'], -omni_1hour['Vx_gse'])






# ============================
# do the WSA speed loop here
# ===========================

N_wsa_params = 100
huxtoutput = np.ones((nt, N_wsa_params))
MAE = np.ones(N_wsa_params)

# run HUXT with these values
#================================================
for i in range(0,N_wsa_params):
    #perturb the speed map, in this case just subtract a constant
    this_v_map = wsa_vr_map - i*2*u.km/u.s
    
    #extract the longitudinal profile at the lat of interest
    v_long = this_v_map[iE_lat,:] 
    
    #run HUXt
    model = H.HUXt(v_boundary=v_long, simtime=simtime, 
                   cr_num= cr_fits,  frame = 'synodic',
                   lon_out=0.0*u.deg, latitude = E_lat,
                   #cr_lon_init = cr_lon_init,
                   r_min=21.5*u.solRad, r_max=225*u.solRad)
    model.solve([]) 
    
    #get value at Earth radius
    huxtoutput[:,i] = model.v_grid[:,iE_r,0]
    #compute MAE
    MAE[i] = np.nanmean(abs(omni_v - huxtoutput[:,i]))







#plot an example output
#=================
n = 50
plt.figure()
plt.plot(time.to_datetime(), huxtoutput[:,n], label = 'PFSS/HUXt')
plt.plot(time.to_datetime(), omni_v, label = 'OMNI')
plt.ylabel(r'$V_{SW}$ [km/s]')
plt.title('MAE: ' +str(MAE[n]) + ' km/s' )
plt.legend()
