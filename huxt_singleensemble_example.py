# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:00:53 2022

@author: mathewjowens
"""


import os
import datetime
import astropy.units as u
import numpy as np
import logging

import huxt_inputs as Hin
import huxt_ensembles as Hens



#==============================================================================
#forecasttime = datetime.datetime(2022,11,23,12,0,0)
forecasttime = datetime.datetime.now()

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))

#where to save the output images
savedir = os.path.join(cwd, 'output')
datadir = os.path.join(cwd, 'data')
logdir  = os.path.join(cwd, 'logs')

#==============================================================================

#create the log file
logfile = os.path.join(logdir, 'log_forecastdate_' +  forecasttime.strftime("%Y_%m_%dT%H_%M_%S") + '.log')

logger = logging.getLogger('WSA-HUXt')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


#==============================================================================
#get WSA data for this date - assumes API is an env vairable. 
#Can swap this section out for PFSS. Though would still need cone file.
#==============================================================================
ndays = 1 # download latest WSA and cone solution up to this many days prior
#UTC date format is "%Y-%m-%dT%H:%M:%S"
sdate = forecasttime - datetime.timedelta(days=ndays)
fdate = forecasttime
success, mapfilepath, conefilepath, model_time = Hens.getMetOfficeWSAandCone(sdate, fdate, datadir)

if success:
    logger.info('WSA and Cone files successfully downloaded from Met Office API')
    logger.info('WSA.FITS timestamp: ' + str(model_time))
    logger.info('WSA file path: ' + mapfilepath)
    logger.info('Cone file path: ' + conefilepath)
else:
    logger.error('no WSA or Cone data returned from Met Office API')
#===============================================================================
    

#run parameters
#==============================================================================

#ambient ensemble parameters
logging.info('==============Ambient Ensemble Parameters=========')
run_buffer_time = 5*u.day;           logging.info('run_buffer_time: ' + str(run_buffer_time))
N_ens_amb = 100;                     logging.info('N_ens_amb: ' + str(N_ens_amb))
simtime = 12*u.day;                  logging.info('N_ens_amb: ' + str(N_ens_amb)) 
lat_rot_sigma = 5*np.pi/180*u.rad;   logging.info('lat_rot_sigma: ' + str(lat_rot_sigma))
lat_dev_sigma = 2*np.pi/180*u.rad;   logging.info('lat_dev_sigma: ' + str(lat_dev_sigma))
long_dev_sigma = 2*np.pi/180*u.rad;  logging.info('long_dev_sigma: ' + str(long_dev_sigma))

#ambient ensemble parameters
logging.info('==============HUXt run Parameters=========')
r_in = 21.5*u.solRad;                logging.info('r_in: ' + str(r_in))    
r_out = 230*u.solRad;                logging.info('r_out: ' + str(r_out)) 
dt_scale = 4;                        logging.info('dt_scale: ' + str(dt_scale)) 

#CME ensemble parameters
logging.info('==============CME Ensemble Parameters=========')
N_ens_cme = 500;                     logging.info('N_ens_cme: ' + str(N_ens_cme))  
cme_v_sigma_frac = 0.1;              logging.info('cme_v_sigma_frac: ' + str(cme_v_sigma_frac))  
cme_width_sigma_frac = 0.1;          logging.info('cme_width_sigma_frac: ' + str(cme_width_sigma_frac))  
cme_thick_sigma_frac = 0.1;          logging.info('cme_thick_sigma_frac: ' + str(cme_thick_sigma_frac))  
cme_lon_sigma = 10*u.deg;            logging.info('cme_lon_sigma: ' + str(cme_lon_sigma))  
cme_lat_sigma = 10*u.deg;            logging.info('cme_lat_sigma: ' + str(cme_lat_sigma))  

#plotting parameters
confid_intervals = [5, 10, 32]
vlims = [300,850]
#compute the HUXt run start date, to allow for CMEs before the forecast date
starttime = forecasttime - datetime.timedelta(days=run_buffer_time.to(u.day).value) 


#Load the WSA data
if os.path.exists(mapfilepath):
    vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
        = Hin.get_WSA_maps(mapfilepath)
    runname = 'WSA'
    logger.info('WSA map loaded')
else:
    logger.error('no speed map found')
        
#Load the CME parameters
if os.path.exists(conefilepath):
    cme_list = Hin.ConeFile_to_ConeCME_list_time(conefilepath, starttime)
    logger.info('Cone file loaded')
else:
    logger.error('no cone file found')    
        
#run the ambient ensemble 
#==================================================================
#generate the ambient ensemble
logger.info('Running ambient ensemble')
ambient_time, huxtinput_ambient, huxtoutput_ambient = \
                     Hens.ambient_ensemble(vr_map, 
                                      vr_longs, 
                                      vr_lats,
                                      starttime,
                                      simtime = simtime,
                                      N_ens_amb = N_ens_amb,
                                      lat_rot_sigma = lat_rot_sigma,
                                      lat_dev_sigma = lat_dev_sigma,
                                      long_dev_sigma = long_dev_sigma,
                                      r_in = r_in, r_out = r_out,     
                                      dt_scale = dt_scale)

#generate the CME ensemble
logger.info('Running CME ensemble')
cme_time, huxtoutput_cme, cmearrivaltimes, cmearrivalspeeds  = \
    Hens.cme_ensemble(huxtinput_ambient, 
                 starttime, cme_list, 
                 simtime = simtime,
                 N_ens_cme = N_ens_cme,
                 cme_v_sigma_frac = cme_v_sigma_frac,
                 cme_width_sigma_frac = cme_width_sigma_frac,
                 cme_thick_sigma_frac = cme_thick_sigma_frac,
                 cme_lon_sigma = cme_lon_sigma,
                 cme_lat_sigma = cme_lat_sigma,
                 r_in = r_in, 
                 r_out = r_out,     
                 dt_scale = dt_scale) 
    
#plot the dashboard
logger.info('Generating output summary plot')
fig, axs = Hens.plot_ensemble_dashboard(ambient_time, vr_map,
                                        vr_longs, 
                                        vr_lats,
                                        cme_list,
                                        huxtoutput_ambient,
                                        huxtoutput_cme,
                                        cmearrivaltimes,
                                        cmearrivalspeeds, 
                                        forecasttime,
                                        filename = mapfilepath,
                                        runname = runname,
                                        confid_intervals = confid_intervals,
                                        vlims = vlims)

# save the summary figure, both with the forecast time and as the most recent forecast
fname = os.path.join(savedir, 'plot_forecast_' +  forecasttime.strftime("%Y_%m_%dT%H_%M_%S") + '.png')
fig.savefig(fname)
logger.info('Plot saved as ' + fname)
fname = os.path.join(savedir, 'plot_current.png')
fig.savefig(fname)
logger.info('Plot saved as ' + fname)
