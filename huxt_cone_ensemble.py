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
#forecasttime = datetime.datetime(2023,4,5,0,0,0)
forecasttime = datetime.datetime.now()

cor_inputs = ['WSA', 'PFSS']

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))

#where to save the output images
savedir = os.path.join(cwd, 'output')
datadir = os.path.join(cwd, 'data')
logdir  = os.path.join(cwd, 'logs')

deacc = True # whether to reduce WSA speeds from 1-AU calibrated values to 21.5 rS
det_viz = True # whether to generate the deterministic visualisations

#==============================================================================
for cor_input in cor_inputs:
    #create the log file
    logfile = os.path.join(logdir, 'log_forecastdate_' + forecasttime.strftime("%Y_%m_%dT%H_%M_%S") + '.log')
    logger = logging.getLogger(cor_input + '-HUXt_ensemble')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs everything
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
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
    vlims = [300,900]
    
    #==============================================================================
    #get Met Office data for this date - assumes API is an env vairable. 
    #==============================================================================
    ndays = 2 # download data solution up to this many days prior
    sdate = forecasttime - datetime.timedelta(days=ndays)
    fdate = forecasttime
    
    #get cone files for the given dates
    success, conefilepath,  model_time = Hens.getMetOfficeCone(sdate, fdate, datadir)
    if success:
        logger.info('Cone CME file successfully downloaded from Met Office API')
        logger.info('Cone2bc.in timestamp: ' + str(model_time))
        logger.info('Cone file path: ' + conefilepath)
    else:
        logger.error('no Cone CME data returned from Met Office API')
    
    
    ambient_success = False
    if cor_input == 'WSA':
        #get WSA solution
        success, mapfilepath,  model_time = Hens.getMetOfficeWSA(sdate, fdate, datadir)
        if success:
            logger.info('WSA file successfully downloaded from Met Office API')
            logger.info('WSA.FITS timestamp: ' + str(model_time))
            logger.info('WSA file path: ' + mapfilepath)
        else:
            logger.error('no WSA data returned from Met Office API')
           
        #Load the WSA data
        if os.path.exists(mapfilepath):
            vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
                = Hin.get_WSA_maps(mapfilepath)
            runname = 'WSA_' + model_time
            logger.info('WSA map loaded')
            ambient_success = True
            
            if deacc:
                #deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
                vr_map_deacc = vr_map.copy()
                for nlat in range (1, len(vr_lats)):
                    vr_map_deacc[nlat,:], lon_temp = Hin.map_v_inwards(vr_map[nlat,:], 215*u.solRad, 
                                                             vr_longs, r_in)
                runname = runname + '_deaccelerated'
                logger.info('WSA map deaccelerated from 1 AU to 21.5 rS')
                vr_map = vr_map_deacc
        
        else:
            logger.error('no WSA speed map found')   
    elif cor_input == 'PFSS':
        #get PFSS solution
        success, mapfilepath, model_time = Hens.getMetOfficePFSS(sdate, fdate, datadir)
        if success:
            logger.info('PFSS file successfully downloaded from Met Office API')
            logger.info('PFSS.nc timestamp: ' + str(model_time))
            logger.info('PFSS file path: ' + mapfilepath)
        else:
            logger.error('no PFSS data returned from Met Office API')
           
        #Load the WSA data
        if os.path.exists(mapfilepath):
            vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats \
                = Hin.get_PFSS_maps(mapfilepath)
            runname = 'PFSS_' + model_time
            logger.info('PFSS map loaded')
            ambient_success = True
            
        else:
            logger.error('no PFSS speed map found')   
    
    
    #compute the HUXt run start date, to allow for CMEs before the forecast date
    starttime = forecasttime - datetime.timedelta(days=run_buffer_time.to(u.day).value) 
           
    #Load the CME parameters
    cme_success = False
    if os.path.exists(conefilepath):
        cme_list = Hin.ConeFile_to_ConeCME_list_time(conefilepath, starttime)
        logger.info('Cone file loaded')
        cme_success = True
    else:
        logger.error('no cone file found')    
            
    #run the ensembles 
    #==================================================================
    #generate the ambient ensemble
    if ambient_success:
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
        if cme_success:
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
        else:
            huxtoutput_cme = []
            cmearrivaltimes = []
            cmearrivalspeeds = []
            
        #plot the dashboard
        logger.info('Generating output summary plot')
        time = ambient_time
        fig, axs = Hens.plot_ensemble_dashboard_V2(time, vr_map,
                                                vr_longs, 
                                                vr_lats,
                                                cme_list,
                                                huxtoutput_ambient,
                                                huxtoutput_cme,
                                                cmearrivaltimes,
                                                cmearrivalspeeds, 
                                                forecasttime, starttime,
                                                filename = mapfilepath,
                                                runname = runname,
                                                confid_intervals = confid_intervals,
                                                vlims = vlims)
        
        # save the summary figure, both with the forecast time and as the most recent forecast
        fname = os.path.join(savedir, 'plot_' + cor_input + '-HUXt_forecast_' +  forecasttime.strftime("%Y_%m_%dT%H_%M_%S") + '.png')
        fig.savefig(fname)
        logger.info('Plot saved as ' + fname)
        fname = os.path.join(savedir, 'plot_current.png')
        fig.savefig(fname)
        logger.info('Plot saved as ' + fname)
    
    
    # <codecell> Do a single deterministic run for visualisation purposes
        if det_viz:
            import huxt as H
            import huxt_analysis as HA
            
            
            
            cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)
            #Use the HUXt ephemeris data to get Earth lat over the CR
            #========================================================
            dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=simtime, 
                               dt_scale=dt_scale, cr_num= cr, cr_lon_init = cr_lon_init, 
                               lon_out=0.0*u.deg, r_min=r_in, r_max=r_out)
            
            #retrieve a bodies position at each model timestep:
            earth = dummymodel.get_observer('earth')
            #get Earth lat as a function of longitude (not time)
            E_lat = np.nanmean (np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c)))
            
            if cor_input == 'WSA':
                #get the WSA values at Earth lat.
                v_in = Hin.get_WSA_long_profile(mapfilepath, lat= E_lat)
                if deacc:
                    #deaccelerate them?
                    v_in, lon_temp = Hin.map_v_inwards(v_in, 215*u.solRad, 
                                                             vr_longs, r_in)
            elif cor_input == 'PFSS':
                v_in = Hin.get_PFSS_long_profile(mapfilepath, lat= E_lat)
            
            
            model = H.HUXt(v_boundary=v_in, simtime=simtime,
                           latitude = np.mean(E_lat), cr_lon_init = cr_lon_init,
                           dt_scale=dt_scale, cr_num= cr,
                           r_min=r_in, r_max= r_out, frame = 'sidereal')
            
            if cme_success:
                cmes = Hin.ConeFile_to_ConeCME_list(model, conefilepath)
            else:
                cmes = []
            
            model.solve(cmes)
            
            fname = os.path.join(savedir, 'plot_' + cor_input + '-HUXt_snapshot_' +  forecasttime.strftime("%Y_%m_%dT%H_%M_%S") + '.png')
            fig, ax = HA.plot(model,run_buffer_time)
            fig.savefig(fname)
    
    
    
