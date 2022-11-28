# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 08:27:51 2022

@author: mathewjowens
"""
import requests
import os
import datetime

import huxt_inputs as Hin


forecasttime = datetime.datetime(2022,11,21,9)
ndays = 1 # download latest WSA and cone solution up to this many days prior

#UTC date format is "%Y-%m-%dT%H:%M:%S"
startdate = forecasttime - datetime.timedelta(days=ndays)
enddate = forecasttime

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(cwd, 'data')
savedir = os.path.join(cwd, 'output')

# startdate = '2022-05-25T10:10:00'
# enddate = '2022-05-27T10:10:00'




success, wsafilepath, conefilepath, model_time = Hin.getMetOfficeWSAandCone(
    startdate, enddate, datadir)

if success:
    #read in the files
    import huxt_inputs as Hin
    vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_num = Hin.get_WSA_maps(wsafilepath)
    cme_list = Hin.ConeFile_to_ConeCME_list_time(conefilepath, startdate)
    
    
    #run HUXt
    import huxt_inputs as Hin
    import huxt as H
    import huxt_analysis as HA
    import huxt_ensembles as Hens
    import astropy.units as u
    import numpy as np
    
    cr, cr_lon_init = Hin.datetime2huxtinputs(startdate)
    
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                       cr_num= cr, cr_lon_init = cr_lon_init, 
                       lon_out=0.0*u.deg,
                       r_min=21.5*u.solRad)
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    E_lat = np.mean(earth.lat_c)
    E_r = np.mean(earth.r)
    reflat = np.nanmean(np.interp(vr_longs,earth.lon_c,earth.lat_c))
    
    
    vr_in = Hin.get_WSA_long_profile(wsafilepath, lat = reflat.to(u.deg))
    model = H.HUXt(v_boundary=vr_in, cr_num=cr, cr_lon_init=cr_lon_init, latitude = reflat,
                       simtime=27*u.day, dt_scale=4, r_min = 21.5*u.solRad, frame = 'sidereal')
    model.solve(cme_list) 
    
    HA.plot(model, 1*u.day)
    HA.plot_earth_timeseries(model, plot_omni=False)
    
    
    #run the ensemble master script
    Hens.sweep_ensemble_run(forecasttime, savedir =savedir, 
                           wsafilepath = wsafilepath, conefilepath = conefilepath)
else: 
    print('no files for this date')
