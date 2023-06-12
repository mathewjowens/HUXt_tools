# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:18:32 2021

@author: mathewjowens
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.spatial import cKDTree
from scipy import interpolate
import astropy.units as u
from astropy.time import Time, TimeDelta
import json
import pandas as pd
import os
import datetime
import h5py
import requests
import pickle



import huxt as H
import huxt_inputs as Hin


def interp2d(xi, yi, V, x, y, n_neighbour = 4):
    """
    Fast 3d interpolation on an irregular grid. Uses the K-Dimensional Tree
    implementation in SciPy. Neighbours are weighted by 1/d^2, where d is the 
    distance from the required point.
    
    Based on Earthpy exmaple: http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    
    Mathew Owens, 8/7/20
    
    Added check for infinite weights, resulting from the interpolated points 
    being identicial to original grid points. 26/11/20

    Parameters
    ----------
    xi, yi, zi :  Ni x Mi arrays of new positions at which to interpolate. 
    
    V : N x M array of the parameter field to be interpolated
    
    x, y, z: N x M arrays of the position of the parameter field, V
        
    n_neighbour : Number of neighbours to use in interpolation. The default is 4.

    Returns
    -------
    Vi : Ni x Mi array of the parameter at new positions.

    """
    
    #check that the dimensions of the coords and V are the same
    assert len(V) == len(x)
    assert len(x) == len(y)
    assert len(xi) == len(yi)

    
    
    #create a list of grid points
    gridpoints=np.ones((len(x.flatten()),3))
    gridpoints[:,0]=x.flatten()
    gridpoints[:,1]=y.flatten()
    gridpoints[:,2]=x.flatten()*0
    
    #create a list of densities
    V_list=V.flatten()

    #Create cKDTree object to represent source grid
    tree=cKDTree(gridpoints)
    
    #get the size of the new coords
    origsize=xi.shape

    newgridpoints=np.ones((len(xi.flatten()),3))
    newgridpoints[:,0]=xi.flatten()
    newgridpoints[:,1]=yi.flatten()
    newgridpoints[:,2]=xi.flatten()*0
    
    #nearest neighbour
    #d, inds = tree.query(newgridpoints, k = 1)
    #rho_ls[:,ie]=rholist[inds]
    
    #weighted sum of N nearest points
    distance, index = tree.query(newgridpoints, k = n_neighbour)
    #tree.query  will sometimes return an index past the end of the grid list?
    index[index>=len(gridpoints)]=len(gridpoints)-1
    
    #weight each point by 1/dist^2
    weights = 1.0 / distance**2
    
    #check for infinite weights (i.e., interp points identical to original grid)
    areinf=np.isinf(weights[:,0])
    weights[areinf,0]=1.0
    weights[areinf,1:]=0.0
    
    #generate the new value as the weighted average of the neighbours
    Vi_list = np.sum(weights * V_list[index], axis=1) / np.sum(weights, axis=1)
    
    #revert to original size
    Vi=Vi_list.reshape(origsize)
    
    return Vi



def generate_input_ensemble(phi, theta, vr_map, 
                            reflats, Nens = 500, 
                            lat_rot_sigma = 5*np.pi/180*u.rad, 
                            lat_dev_sigma = 2*np.pi/180*u.rad,
                            long_dev_sigma = 2*np.pi/180*u.rad):
    """
    a function generate an ensemble of solar wind speed HUXt inputs from a 
    V map such as provided by PFSS, DUMFRIC, HelioMAS. The first ensemble 
    member is always the unperturbed value

    Parameters
    ----------
    vr_map : float array, dimensions (nlong, nlat)
         The solar wind speed map
    phi : Float array, dimensions (nlong, nlat)
        The Carrington longitude in radians
    theta : Float array, dimensions (nlong, nlat)
        The heliographic longitude in radians (from equator)
    reflat : float array, dimesnions (nlong)
        The Earth's latitude in radians (from equator)
    Nens : Integer
        The number of ensemble members to generate
    lat_rot_sigma : float
        The standard deviation of the Gaussain from which the rotational 
        perturbation is drawn. In radians. 
    lat_dev_sigma: float
        The standard deviation of the Gaussain from which the linear 
        latitudinal perturbation is drawn. In radians
    long_dev_sigma: float
        The standard deviation of the Gaussain from which the linear 
        longitudinal perturbation is drawn. In radians
        
    Returns
    -------
    vr_ensmeble : NP ARRAY, dimensions (Nens, nlong)
        Solar wind speed longitudinal series   
    

    """
    assert((reflats.value.any() > -np.pi/2)  & (reflats.value.any() < np.pi/2))
    assert(len(reflats) == len(phi[0,:]))
    assert(Nens > 0)
    
    vr_longs = phi[0, :] 
    
    lats_E = reflats
    
    vr_E = interp2d(vr_longs, lats_E, vr_map, phi, theta)
    #br_E = interp2d(br_longs, lats_E, br_map, phi, theta)
    #vr_eq = interp2d(vr_longs, lats_E*0, vr_map, phi, theta)
    
    
    #rotation - latidunial amplitude - gaussian distribution
    lat_rots = np.random.normal(0.0, lat_rot_sigma.to(u.rad).value, Nens)  
    #rotation - longitude of node of rotation -uniform distribution
    long_rots = np.random.random_sample(Nens)*2*np.pi
    #deviation - latitude - gaussian distribution
    lat_devs = np.random.normal(0.0, lat_dev_sigma.to(u.rad).value, Nens)  
    #deviation - longitude - gaussian distribution
    long_devs = np.random.normal(0.0, long_dev_sigma.to(u.rad).value, Nens)  
    
    vr_ensemble = np.ones((Nens,len(vr_longs)))
    #br_ensemble = np.ones((Nens,len(br_longs)))
    #first ensemble member is the undeviated value
    vr_ensemble[0,:] = vr_E
    #br_ensemble[0,:] = br_E
    
    #for each set of random params, generate a V long series
    for i in range(1, Nens):
        this_lat = lats_E.value + lat_rots[i] * np.sin(vr_longs.value + long_rots[i]) +lat_devs[i]
        this_long = H._zerototwopi_(vr_longs.value + long_devs[i])
        
        #sort longitude into ascending order
        order = np.argsort(this_long)
        
        v = interp2d(this_long[order], this_lat[order], vr_map, phi, theta)
        vr_ensemble[i,:] = v
        
       # b = interp2d(this_long, this_lat, br_map, phi, theta)
       # br_ensemble[i,:] = b

       
    return vr_ensemble


def generate_HelioMAS_ensemble(cr, Nens = 500,
                               lat_rot_sigma = 5*np.pi/180*u.rad,
                               lat_dev_sigma = 2*np.pi/180*u.rad,
                               long_dev_sigma = 2*np.pi/180*u.rad):
    
    """
    A function to download, read and process MAS output to produce a solar wind
    speed ensmeble. File is saved to HUXt_tools/output

    Args:
        cr: Integer, Carrington rotation number
        Nens: Integer, number of ensemble members to produce

    Returns:
    """
    
    #MAS output is at 30 rS
    r_in = 30*u.solRad
    #get HelioMAS data
    v_map, v_longs, v_lats = Hin.get_MAS_vr_map(cr)
    #create the mesh grid
    phi, theta = np.meshgrid(v_longs, v_lats, indexing = 'xy')
    
    
    
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                      cr_num= cr, 
                       lon_out=0.0*u.deg,
                       r_min=r_in)
    
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    reflats = np.interp(v_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))
    
    vr_ensemble = generate_input_ensemble(phi, theta, v_map, 
                                reflats, Nens = Nens, 
                                lat_rot_sigma = lat_rot_sigma, 
                                lat_dev_sigma = lat_dev_sigma,
                                long_dev_sigma = long_dev_sigma)
    
    #interpolate to 128 long bins
    vr128_ensemble = np.ones((Nens,128))  
    dphi = 2*np.pi/128
    phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
    for i in range(0, Nens):
        vr128_ensemble[i,:] = np.interp(phi128,
                      v_longs.value,vr_ensemble[i,:])
    
    #save file
    outfilename =' HelioMAS_CR' + str(cr) +'_vin_ensemble.h5'
    cwd = os.path.abspath(os.path.dirname('__file__'))
    #the filenames should then be generated from the forecast date
    outfilepath =  os.path.join(cwd, 'output', outfilename)
    
    h5f = h5py.File(outfilepath, 'w')
    h5f.create_dataset('Vin_ensemble', data=vr128_ensemble)
           
    h5f.attrs['lat_rot_sigma'] = lat_rot_sigma
    h5f.attrs['lat_dev_sigma'] = lat_dev_sigma
    h5f.attrs['long_dev_sigma'] = long_dev_sigma
    infilepath = 'Hin.get_MAS_vrmap(cr)'  #this is used only to identify the source files. 
    h5f.attrs['source_file'] = infilepath
    h5f.attrs['r_in_rS'] = r_in
    h5f.attrs['Carrington_rotation'] = cr
    h5f.close()    
    
#compute the percentiles
def getconfidintervals(endata,confid_intervals):
    tmax = len(endata[0,:])
    n = len(confid_intervals)*2 + 1
    confid_ts = np.ones((tmax,n))*np.nan
    for t in range(0,tmax):
        dist = endata[:,t][~np.isnan(endata[:,t])]
        #median
        confid_ts[t,0] = np.percentile(dist,50)
        for nconfid in range(0,len(confid_intervals)):
            confid_ts[t,2*nconfid+1] = np.percentile(dist,confid_intervals[nconfid])
            confid_ts[t,2*nconfid+2] = np.percentile(dist,100-confid_intervals[nconfid])
    return confid_ts

#plot the percentiles
def plotconfidbands(tdata,endata,confid_intervals, plot_legend=False):
    
    n = len(confid_intervals)*2 + 1
    #get confid intervals
    confid_ts = getconfidintervals(endata,confid_intervals)
    
    #change the line colours to use the inferno colormap
    # nc = len(confid_intervals) + 1
    # color = plt.cm.cool(np.linspace(0, 1,nc))
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(confid_intervals))
    nplot = 1 
    nconfid = 0
    while (nplot < n):
        rgba = mpl.cm.cool(norm(nconfid))
        labeltxt = (str(confid_intervals[nconfid]) + '-' 
                    + str(100-confid_intervals[nconfid]) + 'th')
        plt.fill_between(tdata, confid_ts[:,nplot+1], confid_ts[:,nplot],
                         label= labeltxt, color = rgba, zorder = 0 ) 
        nconfid = nconfid + 1
        nplot = nplot + 2
    #plot the median
    plt.plot(tdata, confid_ts[:,0],'w', label = 'Median', zorder = 0)    
    if plot_legend:
        plt.legend(facecolor='silver')    

def ambient_ensemble(vr_map, lons, lats, starttime,
                     N_ens_amb = 100,
                     simtime = 12*u.day, 
                     lat_rot_sigma = 5*np.pi/180*u.rad,
                     lat_dev_sigma = 2*np.pi/180*u.rad,
                     long_dev_sigma = 2*np.pi/180*u.rad,
                     r_in = 21.5*u.solRad,      
                     r_out = 230*u.solRad,     
                     dt_scale = 4):               
    """
    A function to generate am ambient solar wind ensemble time series for a
    given speed and forecast time

    Parameters
    ----------
    vrmap : NxM numpy array containing Carrington map of solar wind speeds.
    lons : Nx1 numpy array of longitudes of vmap, in radians
    lats : Mx1 numpy array of latitudes of vmap, in radians
    starttime : Datetime. Sets the tiem through CR
    
    N_ens_amb: Number of ensemble members to produce
    simtime : HUXt simulation time, in days

    lat_rot_sigma: The standard deviation (in radians) of the Gaussain from which the rotational perturbation is drawn
    lat_dev_sigma : The standard deviation (in radians) of the Gaussain from which the linear latitudinal perturbation is drawn                          
    long_dev_sigma : The standard deviation (in radians) of the Gaussain from which the linear longitudinal perturbation is drawn

    r_in : HUXt inner boundary, in rS. NOTE: DEFAULTS TO 30 Rs
    r_out: HUXt outer boundary, in rS
    dt_scale : Frequency of HUXt timesteps to output
    

    Returns
    -------
    
    time : array of model output time steps, in Astropy Time format
    input_ensemble : Array of HUXt V input conditions, as v(CarrLong)
    output_ensemble: HUXt V time series at Earth
    

    """
    
    #create the mesh grid
    phi, theta = np.meshgrid(lons, lats, indexing = 'xy')
    
    #get the huxt params for the start time
    cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)
    
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=simtime, 
                       dt_scale=dt_scale, cr_num= cr, cr_lon_init = cr_lon_init, 
                       lon_out=0.0*u.deg,
                       r_min=r_in, r_max=r_out)
    
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    E_lat = np.interp(lons,np.flipud(earth.lon_c),np.flipud(earth.lat_c))
    E_r = np.mean(earth.r)
    
    #find the gridcell which corresponds to this.
    nE_r = np.argmin(abs(dummymodel.r - E_r))
    
        
    #==============================================================================
    #generate the ambient solar wind input ensemble
    #==============================================================================
    vr_ensemble = generate_input_ensemble(phi, theta, vr_map, 
                                          reflats = E_lat, Nens = N_ens_amb,
                                          lat_rot_sigma = lat_rot_sigma, 
                                          lat_dev_sigma = lat_dev_sigma,
                                          long_dev_sigma = long_dev_sigma)
        
    #resample the ensemble to 128 longitude bins
    vr128_ensemble = np.ones((N_ens_amb,128))  
    dphi = 2*np.pi/128
    phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
    for i in range(0, N_ens_amb):
        vr128_ensemble[i,:] = np.interp(phi128,
                      lons.value,vr_ensemble[i,:])
        
    #==============================================================================
    #run huxt with the ambient ensemble
    #==============================================================================
    nsteps = len(dummymodel.time_out)
    huxtoutput_ambient = np.ones((N_ens_amb,nsteps))
    #huxtoutput = []
    for i in range(0,N_ens_amb):
        model = H.HUXt(v_boundary=vr128_ensemble[i]* (u.km/u.s), simtime=simtime, 
                       dt_scale=dt_scale, cr_num= cr,
                       lon_out=0.0*u.deg, latitude = np.mean(E_lat),
                       cr_lon_init = cr_lon_init,
                       r_min=r_in, r_max= r_out)
        model.solve([]) 
        
        #find Earth location and extract the time series
        #huxtoutput.append(HA.get_earth_timeseries(model))
        
        #get value at Earth
        huxtoutput_ambient[i,:] = model.v_grid[:,nE_r,0]
        
        if i % 100 == 0:
            print('HUXt run (ambient) ' + str(i+1) + ' of ' + str(N_ens_amb))
            
    time = model.time_out + model.time_init
    
    return time, vr128_ensemble, huxtoutput_ambient

def cme_ensemble(huxtinput_ambient, starttime, cme_list,
                 N_ens_cme = 500,
                 simtime = 12*u.day,  
                 cme_v_sigma_frac = 0.1,
                 cme_width_sigma_frac = 0.1,
                 cme_thick_sigma_frac = 0.1,
                 cme_lon_sigma = 10*u.deg,
                 cme_lat_sigma = 10*u.deg,
                 r_in = 21.5*u.solRad,      
                 r_out = 230*u.solRad,
                 dt_scale = 4):
    """
    A function to generate a CME and ambient solar wind ensemble time series for a
    given speed and forecast time

    Parameters
    ----------
    huxtinput_ambient : speed as a function of CarrLong from ambient_ensemble (could be a single profile)
    starttime : Datetime. Sets the time through CR
    
    N_ens_cme: Number of ensemble members to produce
    simtime : HUXt simulation time, in days

    cme_v_sigma_frac : The standard deviation of the Gaussain from which the CME V [frac of value] perturbation is drawn
    cme_width_sigma_frac : The standard deviation of the Gaussain from which the CME width [frac of value] perturbation is drawn
    cme_thick_sigma_frac : The standard deviation of the Gaussain from which the CME thickness [frac of value] perturbation is drawn
    cme_lon_sigma  : The standard deviation [in deg] of the Gaussain from which the CME long [deg] perturbation is drawn
    cme_lat_sigma : The standard deviation [in deg] of the Gaussain from which the CME lat [deg] perturbation is drawn

    r_in : HUXt inner boundary, in rS. NOTE: DEFAULTS TO 30 Rs
    r_out: HUXt outer boundary, in rS
    dt_scale : Frequency of HUXt timesteps to output
    

    Returns
    -------
    
    time : array of model output time steps, in Astropy Time format
    huxtoutput_cme: HUXt V time series at Earth
    cmearrivaltimes : Time series of CME front arrivals, per CME
    cmearrivalspeeds : values of CME front arrival speeds, per CME
    

    """
    N_ens_amb = len(huxtinput_ambient[:,0])   
    
    cr, cr_lon_init = Hin.datetime2huxtinputs(starttime)
    
    #recreate the long grid
    nlong = len(huxtinput_ambient[0,:])
    dphi = 2*np.pi/nlong 
    vr_longs = np.arange(dphi/2, 2*np.pi - dphi/2 +0.0001, dphi) * u.rad
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=simtime, 
                       dt_scale=dt_scale, cr_num= cr, cr_lon_init = cr_lon_init, 
                       lon_out=0.0*u.deg,
                       r_min=r_in, r_max=r_out)
    
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    E_lat = np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))
    E_r = np.mean(earth.r)
    #find the gridcell which corresponds to this.
    nE_r = np.argmin(abs(dummymodel.r - E_r))

    
    #get the sorted CME list
    model = H.HUXt(v_boundary=huxtinput_ambient[0]* (u.km/u.s), simtime=simtime,
                   latitude = np.mean(E_lat), cr_lon_init = cr_lon_init,
                   dt_scale=dt_scale, cr_num= cr, lon_out=0.0*u.deg, 
                   r_min=r_in, r_max= r_out, track_cmes = False)
    model.solve(cme_list) 
    cme_list = model.cmes
    n_cme = len(cme_list)
    
    # Run HUXt with perturbed CMEs
    nsteps = len(model.time_out)
    huxtoutput_cme = np.ones((N_ens_cme,nsteps))
    cmearrivaltimes = np.zeros((nsteps,n_cme))
    cmearrivalspeeds = np.zeros((N_ens_cme,n_cme))*np.nan
    
    for i in range(0,N_ens_cme):
        cme_list_perturb = []
        for n in range(n_cme):
            cme = cme_list[n]
            v_perturb = cme.v.value 
            width_perturb = cme.width.to(u.deg).value 
            thick_perturb = cme.thickness.value 
            lon_perturb = cme.longitude.to(u.deg).value 
            lat_perturb = cme.latitude.to(u.deg).value 
            
            # don't perturb first ensemble member
            if i > 0: 
                v_perturb = v_perturb + np.random.normal(0.0, cme_v_sigma_frac* cme_list[n].v.value)
                width_perturb = width_perturb + np.random.normal(0.0, cme_width_sigma_frac* cme.width.value)
                thick_perturb = thick_perturb + np.random.normal(0.0, cme_thick_sigma_frac* cme.thickness.value)
                lon_perturb = lon_perturb + np.random.normal(0.0, cme_lon_sigma.to(u.deg).value)
                lat_perturb = lat_perturb + np.random.normal(0.0, cme_lat_sigma.to(u.deg).value)
            
            cme = H.ConeCME(cme.t_launch, 
                               longitude=lon_perturb*u.deg, latitude = lat_perturb*u.deg,
                               width=width_perturb*u.deg, 
                               v=v_perturb*u.km/u.s, thickness= thick_perturb*u.solRad,
                               initial_height = cme.initial_height)
            cme_list_perturb.append(cme)
            
        #set up huxt with a random ambient solution
        i_ambient = 0
        if i > 0: #use unperturbed solar wind for first ensemble member
            i_ambient = np.random.randint(0, N_ens_amb)
            
        model = H.HUXt(v_boundary=huxtinput_ambient[i_ambient]* (u.km/u.s), simtime=simtime,
                       latitude = np.mean(E_lat), cr_lon_init = cr_lon_init,
                       dt_scale=dt_scale, cr_num= cr, lon_out=0.0*u.deg, 
                       r_min=r_in, r_max= r_out, track_cmes = False)
        model.solve(cme_list_perturb) 
        
        #Extract the Earth V
        huxtoutput_cme[i,:] = model.v_grid[:,nE_r,0]
        
        #get arrival time amd speed at Earth for each CME
        for cme_id in range(0,n_cme):
            cme_r_field = model.cme_particles_r[cme_id, :, :, :]
            cme_v_field = model.cme_particles_v[cme_id, :, :, :]
            i_at_earth = np.where(abs(cme_r_field[:,0,0]*u.km - E_r.to(u.km)) < model.dr.to(u.km))[0]
            if i_at_earth.any():
                cmearrivaltimes[i_at_earth[0],cme_id] = cmearrivaltimes[i_at_earth[0],cme_id] + 1
                cmearrivalspeeds[i, cme_id] = cme_v_field[i_at_earth[0],0,0]
        
        if i % 100 == 0:
            print('HUXt run (CME) ' + str(i+1) + ' of ' + str(N_ens_cme))
            
    time = model.time_out + model.time_init
    
    return time, huxtoutput_cme, cmearrivaltimes, cmearrivalspeeds
        

# time = cme_time
# cme_list = cme_list
# vr_map = cortom_vr_map
# huxtoutput_ambient = cortom_huxtoutput_ambient
# huxtoutput_cme = cortom_huxtoutput_cme
# cmearrivaltimes = cortom_cmearrivaltimes
# cmearrivalspeeds = cortom_cmearrivalspeeds

def long_lat_of_circle_on_sphere(centre_lon, centre_lat, ang_radius, npoints = 100):
    # centre_long = longitude (in rad) of circle centre
    # centre_lat = latitude (in rad) of circle centre
    # ang_radius = angular half width of cicle, in rad
    
    # Generate angles along the circumference of the minor circle
    theta = np.linspace(0, 2*np.pi, npoints)
    
    # Calculate the spherical coordinates of the points on the circumference
    latitudes = np.arcsin(np.sin(centre_lat) * np.cos(ang_radius) + 
                          np.cos(centre_lat) * np.sin(ang_radius) * np.cos(theta))
    longitudes = centre_lon + np.arctan2(np.sin(theta) * np.sin(ang_radius) * np.cos(centre_lat),
                                          np.cos(ang_radius) - np.sin(centre_lat) * np.sin(latitudes))
    
    return longitudes, latitudes


def plot_ensemble_dashboard(time, vr_map, map_lon, map_lat, cme_list,
                            huxtoutput_ambient, huxtoutput_cme,
                            cmearrivaltimes, cmearrivalspeeds, 
                            forecasttime,
                            confid_intervals = [5, 10, 32],
                            vlims = [300,850],
                            filename = " ", runname = " "):
    """
    Function to plot the ensemble dashboard

    Parameters
    ----------
    time : times of HUXt output. output from ambient or CME ensemble
    vr_map : Carrington map of Vr
    cme_list : List of CME events used to generate CME ensemble
    huxtoutput_ambient : V times series from ambient ensemble
    huxtoutput_cme : V time series from cme_ensemble
    cmearrivaltimes : CME arrival time time series from cme_ensemble
    cmearrivalspeeds : CME arrival speeds from cme_ensemble
    forecasttime : Time fo forecast (datetime)
    confid_intervals : condifence intervals for ensemble plots [5, 10, 32].
    vlims : Plot limts for V timeseries[300,800].
    filename : Name of coronal run for plot annotation

    Returns
    -------
    fig, ax[array]: plot handles

    """
    
    colours = 'b','r','g','c','m','y','k','lime', 'indigo','orangered','b','r','k'
    
    mpl.rc("axes", labelsize=14)
    mpl.rc("ytick", labelsize=14)
    mpl.rc("xtick", labelsize=14)
    mpl.rc("legend", fontsize=14)
    
    #HUXt output
    startdate = time[0].to_datetime()
    enddate = time[-1].to_datetime()

    #get the array sizes
    N_ens_amb = len(huxtoutput_ambient[:,1])
    if cme_list:
        n_cme = len(cme_list) 
        N_ens_cme = len(huxtoutput_cme[:,1])
        #sort the CME list into chronological order
        launch_times = np.ones(n_cme)
        for i, cme in enumerate(cme_list):
            launch_times[i] = cme.t_launch.value
        id_sort = np.argsort(launch_times)
        cme_list = [cme_list[i] for i in id_sort]
    else: 
        n_cme = 0    
        
    # #work out the CME hit probability.
    # arrival_prob = 
    # if n_cme>0:
    #     for n in range(0,n_cme):
        
    
    
    #recreate the long and lat grid
    vr_longs = map_lon
    vr_lats = map_lat
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=1*u.day, 
                       cr_num= cr, cr_lon_init = cr_lon_init, 
                       lon_out=0.0*u.deg)
    
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    E_lat = np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))

    
    #compute the CME hit fraction
    cme_hit = np.ones((n_cme))
    for n in range(0,n_cme):
        cme_hit[n] = np.sum(~np.isnan(cmearrivalspeeds[:,n])) / N_ens_cme 
    
    # plot it
    #===========================================================
    fig = plt.figure(figsize = (17,10))
    gs = fig.add_gridspec(3, 3)
    
    
    #==================================V from ambient ensemble=================
    ax1 = fig.add_subplot(gs[0, :-1])
    plt.sca(ax1)
    plotconfidbands(time.to_datetime(),  huxtoutput_ambient, confid_intervals)
    ax1.plot(time.to_datetime(), huxtoutput_ambient[0,:], 'k--', zorder = 1)
    #plt.xlabel('Time through CR [days]')
    ax1.set_ylim(vlims)
    ax1.set_ylabel(r'V$_{SW}$ [km/s]')
    #ax1.set_title('Ambient solar wind (N = ' + str(N_ens_amb) + ')', fontsize = 14)
    ax1.text(0.02, 0.90,'Ambient solar wind (N = ' + str(N_ens_amb) + ')', fontsize = 14,
             transform=ax1.transAxes, backgroundcolor = 'silver')
    yy = plt.gca().get_ylim(); 
    ax1.set_ylim(yy); ax1.set_xlim([startdate, enddate]);
    ax1.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                     color = 'silver', zorder = 2, alpha = 0.7)
    ax1.legend(facecolor='silver', loc = 'lower left', framealpha=1, ncol = 2)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid(True, axis = 'x')
    ax1.grid(True, which = 'minor', axis = 'x')
    ax1.set_title(runname + '-HUXt', fontsize = 16)
    
    #==================================V from CME ensemble=====================
    ax2 = fig.add_subplot(gs[1, :-1])
    if n_cme >0:
        
        plt.sca(ax2)
        plotconfidbands(time.to_datetime(), huxtoutput_cme, confid_intervals)
        h1 = ax2.plot(time.to_datetime(), huxtoutput_cme[0,:],'k',label='Unperturbed CME', zorder = 1)
        h2 = ax2.plot(time.to_datetime(),huxtoutput_ambient[0,:],'k--',label='Unperturbed SW', zorder = 1)
        #plt.xlabel('Time through CR [days]')
        ax2.set_ylim(vlims)
        ax2.set_ylabel(r'V$_{SW}$ [km/s]')
        #ax2.set_title('Ambient + CMEs (N = ' + str(N_ens_cme) + ')', fontsize = 14)
        ax2.text(0.02, 0.90,'Ambient + CMEs (N = ' + str(N_ens_cme) + ')', fontsize = 14,
                 transform=ax2.transAxes, backgroundcolor = 'silver')
        yy = plt.gca().get_ylim(); 
        ax2.set_ylim(yy); ax2.set_xlim([startdate, enddate]); 
        ax2.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
        ax2.axes.xaxis.set_ticklabels([])
        ax2.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                         color = 'silver', zorder = 2, alpha = 0.7)
        ax2.legend(handles = [h2[0], h1[0]], facecolor='silver', loc = 'lower left', 
                   framealpha=1, ncol=1)
        ax2.xaxis.set_minor_locator(mdates.DayLocator())
        ax2.grid(True, axis = 'x')
        ax2.grid(True, which = 'minor', axis = 'x')
        
        #smooth the arrival time ensembles for plotting
        pd_cme_arrival = pd.DataFrame({ 'Date': time.to_datetime()}) 
        pd_cme_arrival.index = pd_cme_arrival['Date']
        for i in range(0,n_cme):
            pd_cme_arrival['CME'+str(i)] = cmearrivaltimes[:,i]
            pd_cme_arrival['CME'+str(i)+'_smooth'] = pd_cme_arrival['CME'+str(i)].rolling(10,center=True).mean()
        
    #============================CME arrival time distributions================
    ax3 = fig.add_subplot(gs[2, :-1])
    if n_cme >0:
        plt.sca(ax3)
        for n in range(0,n_cme):
            cme_label = 'CME ' +str(n+1) +': ' + str(int(cme_hit[n]*100)) + ' %'
            #ax3.plot(tdata_cme.to_datetime(), cmearrivaltimes[:,n],label='CME ' +str(n+1))
            h = ax3.plot( pd_cme_arrival['CME'+str(n)+'_smooth'] ,label=cme_label,
                         color = colours[n], zorder = 0)
        ax3.set_ylabel( 'Ensemble density')
        ax3.axes.yaxis.set_ticks([])
        yy = plt.gca().get_ylim(); 
        yy = [0,15]
        ax3.set_ylim(yy); ax3.set_xlim([startdate, enddate]);
        ax3.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
        #ax3.set_title('CME front arrival time (N = ' + str(N_ens_cme) + ')', fontsize = 14)
        ax3.text(0.02, 0.90,'CME front arrival time (N = ' + str(N_ens_cme) + ')', fontsize = 14,
                 transform=ax3.transAxes, backgroundcolor = 'silver')
        ax3.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                         color = 'silver', zorder = 2, alpha = 0.7)
        ax3.legend(facecolor='white', loc = 'upper right', framealpha=1)
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 2))
        ax3.xaxis.set_minor_locator(mdates.DayLocator())
        ax3.grid(True, axis = 'x')
        ax3.grid(True, which = 'minor', axis = 'x')
    
    #========================Vr map ===========================================
    ax4 = fig.add_subplot(gs[0, -1])
    pc = ax4.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value, 
               shading='auto',vmin=250, vmax=750)
    ax4.plot(vr_longs*180/np.pi,E_lat*180/np.pi,'k--',label = 'Earth')
    ax4.plot(vr_longs*180/np.pi,E_lat*0,'k')
    ax4.set_ylim([-90,90]); ax4.set_xlim([0,360])
    #ax4.xlabel('Carrington Longitude [deg]')
    #ax4.ylabel('Latitude [deg]')
    #ax4.set_title(filename, FontSize = 14)
    ax4.legend()
    ax4.axes.xaxis.set_ticks([0,90,180,270,360])
    ax4.axes.yaxis.set_ticks([-90,-45,0,45,90])
    ax4.axes.xaxis.set_ticklabels([])
    ax4.axes.yaxis.set_ticklabels([])
    plt.sca(ax4)
    #colorbar
    axins = inset_axes(ax4,
                        width="100%",  # width = 50% of parent_bbox width
                        height="10%",  # height : 5%
                        loc='upper right',
                        bbox_to_anchor=(0.28, 0.60, 0.72, 0.5),
                        bbox_transform=ax4.transAxes,
                        borderpad=0,)
    cb = fig.colorbar(pc, cax = axins, orientation = 'horizontal',  pad = -0.1)
    cb.ax.xaxis.set_ticks_position("top")
    ax4.text(0.02,1.05,r'$V_{SW}$ [km/s]' , 
            fontsize = 14, transform=ax4.transAxes, backgroundcolor = 'w')
    
    #========================Vr map: Add CMEs==================================
    #cbar = fig.colorbar(pc, ax=ax4, orientation = 'horizontal')
    #cbar.set_label(r'V$_{SW}$')
    #plot Earth at forecast time
    cr_f, cr_lon_init_f = Hin.datetime2huxtinputs(forecasttime)
    ax4.plot(cr_lon_init_f*180/np.pi, np.mean(E_lat)*180/np.pi, 'ko', zorder = 1)
    ax4.plot([cr_lon_init_f.value*180/np.pi, cr_lon_init_f.value*180/np.pi],
             [-90,90], 'k--')
    #plot CMEs
    if n_cme>0:
        for n, cme in enumerate(cme_list):
            #ax4.plot(H._zerototwopi_(cr_lon_init_f - cme.longitude)*180/np.pi, cme.latitude*180/np.pi, 'wo')
            stepSize = 0.1
            #Generated vertices
            positions = []  
            r = cme.width.to(u.rad)/2
            a = H._zerototwopi_(cr_lon_init_f - cme.longitude) *u.rad
            b = cme.latitude.to(u.rad)
            t = 0
            while t < 2 * np.pi:
                positions.append( ( (r * np.cos(t) + a).value,
                                  (r * np.sin(t) + b).value ))
                t += stepSize
            pos = np.asarray(positions)
            
            #wrap latitudes out of range
            mask = pos[:,1] > np.pi/2
            pos[mask,1] = -np.pi + pos[mask,1]
            mask = pos[:,1] < - np.pi/2
            pos[mask,1] = np.pi + pos[mask,1]
            
            h = ax4.plot(H._zerototwopi_(pos[:,0])*180/np.pi, pos[:,1]*180/np.pi,'.')
            h[0].set_color(colours[n])
            
    #========================CME arrival speeds================================
    ax5 = fig.add_subplot(gs[1:, -1])
    if n_cme >0:
        #plot the distributions of arrival speeds
        
        speeds = np.arange(200,1000,20)
        speeds_centres = speeds[0:-1]+10
        yspace = N_ens_cme/5
        for n in range(0,n_cme):
            hist, bin_edges = np.histogram(cmearrivalspeeds[:,n], speeds) 
            h = ax5.plot(speeds_centres,hist - yspace*n, label='CME ' +str(n+1),
                         color = colours[n])
            #axs[n].set_title('CME ' + str(n+1))
            #if n==0:
            #    axs[n].set_ylabel('# ensemble members')
        ax5.set_xlabel( 'CME arrival speed [km/s]')
        ax5.set_ylabel( 'Ensemble density')
        ax5.axes.yaxis.set_ticks([])
        #ax5.legend(facecolor='silver', loc = 'upper left', framealpha=1)
        ax5.grid(True, axis = 'x')
    
    
    fig.text(0.05,0.03,'HUXt1D using ' + filename, fontsize = 14)
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right =0.95, top =0.92,
                        wspace = 0.12, hspace =0.12)
    
    return fig, [ax1, ax2, ax3, ax4, ax5]


def plot_ensemble_dashboard_V2(time, vr_map, vr_longs, vr_lats, cme_list,
                            huxtoutput_ambient, huxtoutput_cme,
                            cmearrivaltimes, cmearrivalspeeds, 
                            forecasttime, starttime,
                            confid_intervals = [5, 10, 32],
                            vlims = [300,850],
                            filename = " ", runname = " ", ens_den_thresh = 0.0):
    """
    Function to plot the ensemble dashboard

    Parameters
    ----------
    time : times of HUXt output. output from ambient or CME ensemble
    vr_map : Carrington map of Vr
    cme_list : List of CME events used to generate CME ensemble
    huxtoutput_ambient : V times series from ambient ensemble
    huxtoutput_cme : V time series from cme_ensemble
    cmearrivaltimes : CME arrival time time series from cme_ensemble
    cmearrivalspeeds : CME arrival speeds from cme_ensemble
    forecasttime : Time fo forecast (datetime)
    confid_intervals : condifence intervals for ensemble plots [5, 10, 32].
    vlims : Plot limts for V timeseries[300,800].
    filename : Name of coronal run for plot annotation

    Returns
    -------
    fig, ax[array]: plot handles

    """

    colours = 'b','r','k','g','c','m','y','lime', 'indigo','orangered','b','r','k'
    labelcolour = 'white'
    
    mpl.rc("axes", labelsize=14)
    mpl.rc("ytick", labelsize=14)
    mpl.rc("xtick", labelsize=14)
    mpl.rc("legend", fontsize=14)
    
    #HUXt output
    startdate = time[0].to_datetime()
    enddate = time[-1].to_datetime()

    #get the array sizes
    N_ens_amb = len(huxtoutput_ambient[:,1])
    if cme_list:
        n_cme = len(cme_list) 
        N_ens_cme = len(huxtoutput_cme[:,1])
        #sort the CME list into chronological order
        launch_times = np.ones(n_cme)
        for i, cme in enumerate(cme_list):
            launch_times[i] = cme.t_launch.value
        id_sort = np.argsort(launch_times)
        cme_list = [cme_list[i] for i in id_sort]
    else: 
        n_cme = 0 
        N_ens_cme = 0
         
    
    #Use the HUXt ephemeris data to get Earth lat over the CR
    #========================================================
    cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
    dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=1*u.day, 
                       cr_num= cr, cr_lon_init = cr_lon_init, 
                       lon_out=0.0*u.deg)
    
    #retrieve a bodies position at each model timestep:
    earth = dummymodel.get_observer('earth')
    #get Earth lat as a function of longitude (not time)
    E_lat = np.interp(vr_longs,np.flipud(earth.lon_c),np.flipud(earth.lat_c))
    
    #rotate the speed map so that Earth is at the centre
    vr_map_rot = vr_map.copy()
    for nlat in range(0, len(vr_lats)):
        interp = interpolate.interp1d(H._zerototwopi_(vr_longs.to(u.rad)),
                                      vr_map[nlat, :].value, kind="nearest",
                                      fill_value="extrapolate")
        vr_map_rot[nlat, :] = interp(H._zerototwopi_(vr_longs.to(u.rad) 
                                                     - np.pi*u.rad + cr_lon_init.to(u.rad)))*u.km/u.s

    #smooth the arrival time ensembles for plotting
    pd_cme_arrival = pd.DataFrame({ 'Date': time.to_datetime()}) 
    pd_cme_arrival.index = pd_cme_arrival['Date']
    for i in range(0,n_cme):
        pd_cme_arrival['CME'+str(i)] = cmearrivaltimes[:,i]
        pd_cme_arrival['CME'+str(i)+'_smooth'] = pd_cme_arrival['CME'+str(i)].rolling(10,center=True).mean()
    
    #add the MJD to the arrival times, for computing median and confid intervals
    pd_cme_arrival['mjd'] = Time(pd_cme_arrival['Date']).mjd
    
    #compute the CME hit fraction
    cme_hit = np.ones((n_cme))
    n_cme_hit = 0
    for n in range(0,n_cme):
        cme_hit[n] = np.sum(~np.isnan(cmearrivalspeeds[:,n])) / N_ens_cme 
        if cme_hit[n] > ens_den_thresh:
           n_cme_hit = n_cme_hit + 1
        
    #compute the median arrival time, speed and spread 
    cme_arrival_median = []
    cme_arrival_std = []
    cme_v_median = []
    cme_v_std = []
    for n in range(0,n_cme):
        if cme_hit[n] > ens_den_thresh:
            #put together the distribution function of arrival MJD
            mjd_dist = []
            for t in range(0, len(pd_cme_arrival)):
                Ncmes_t = int(pd_cme_arrival['CME'+str(n)][t])
                if Ncmes_t > 0 :
                    #add the required number of CMEs
                    for j in range(0, Ncmes_t):
                        mjd_dist.append(pd_cme_arrival['mjd'][t])
            #compute the median and std values of this distribution
            cme_arrival_median.append(np.median(mjd_dist))
            cme_v_median.append(np.nanmean(cmearrivalspeeds[:,n]))
            if cme_hit[n] < 0.01:
                cme_arrival_std.append(np.nan)
                cme_v_std.append(np.nan)
            else:
                cme_arrival_std.append(np.nanstd(mjd_dist))
                cme_v_std.append(np.nanstd(cmearrivalspeeds[:,n]))
        else:  # no CME arrival
            cme_arrival_median.append(np.nan)
            cme_arrival_std.append(np.nan)
            cme_v_median.append(np.nan)
            cme_v_std.append(np.nan)
        
    # plot it
    #===========================================================
    fig = plt.figure(figsize = (17,10))
    gs = fig.add_gridspec(3, 3)
    
    
    #===================solar wind timeseries==================================
    ax2 = fig.add_subplot(gs[0, :-1])  
    plt.sca(ax2)
    
    yy = vlims
    ax2.set_ylim(yy)
    ax2.set_xlim([startdate, enddate]); 
  
    h2 = ax2.plot(time.to_datetime(), huxtoutput_ambient[0,:],'k--',label='Unperturbed SW', zorder = 1)
    
    if n_cme >1:
        plotconfidbands(time.to_datetime(), huxtoutput_cme, confid_intervals)
        h1 = ax2.plot(time.to_datetime(), huxtoutput_cme[0,:],'k',label='Unperturbed CME', zorder = 1)
        ax2.legend(handles = [h2[0], h1[0]], facecolor=labelcolour, loc = 'upper right', 
                   framealpha=1, ncol=1)
    else:
        plotconfidbands(time.to_datetime(), huxtoutput_ambient, confid_intervals)


    ax2.set_ylabel(r'V$_{SW}$ [km/s]')
    #ax2.set_title('Ambient + CMEs (N = ' + str(N_ens_cme) + ')', fontsize = 14)
    ax2.text(0.02, 0.90,'Ambient + CMEs (N = ' + str(N_ens_cme) + ')', fontsize = 14,
             transform=ax2.transAxes, backgroundcolor = labelcolour)
    
    ax2.plot([forecasttime, forecasttime], yy, 'silver', zorder = 2)
    ax2.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                     color = 'silver', zorder = 2, alpha = 0.7)
    


    ax2.xaxis.tick_top()
    #ax2.axes.xaxis.set_ticklabels([])
    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval = 2))
    ax2.grid(True, axis = 'x')
    ax2.grid(True, which = 'minor', axis = 'x')
    ax2.grid(True, axis = 'y')
    ax2.grid(True, which = 'minor', axis = 'y')    
        
    #============================CME arrival time distributions================
    ax3 = fig.add_subplot(gs[1, :-1])
    plt.sca(ax3)
    nplot = 0
    for n in range(0,n_cme):
        if cme_hit[n] > ens_den_thresh:
            cme_label = 'CME ' +str(n+1) +': ' + str(int(cme_hit[n]*100)) + ' %'
            #ax3.plot(tdata_cme.to_datetime(), cmearrivaltimes[:,n],label='CME ' +str(n+1))
            h = ax3.plot( pd_cme_arrival['CME'+str(n)+'_smooth'] / N_ens_cme,label=cme_label,
                         color = colours[nplot], zorder = 0)
            nplot = nplot + 1
    ax3.set_ylabel( 'Ensemble density')
    #ax3.axes.yaxis.set_ticks([])
    yy = [0,0.09]
    ax3.set_ylim(yy); ax3.set_xlim([startdate, enddate]);
    ax3.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
    #ax3.set_title('CME front arrival time (N = ' + str(N_ens_cme) + ')', fontsize = 14)
    ax3.text(0.02, 0.90,'CME front arrival time (N = ' + str(N_ens_cme) + ')', fontsize = 14,
             transform=ax3.transAxes, backgroundcolor = labelcolour)
    ax3.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                     color = 'silver', zorder = 2, alpha = 0.7)
    ax3.legend(facecolor='white', loc = 'upper right', framealpha=1)
    #ax3.axes.xaxis.set_ticklabels([])
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 2))
    ax3.xaxis.set_minor_locator(mdates.DayLocator())
    ax3.grid(True, axis = 'x')
    ax3.grid(True, which = 'minor', axis = 'x')
    #ax3.xaxis.tick_top()
        
    #==================================CME info=================
    ax1 = fig.add_subplot(gs[2, :-1])
    plt.sca(ax1)
    ax1.axis('off')
    #list Earth-impacting CMEs
    row_num = 0
    vspace = 0.12
    top_row = 0.65
    fsize = 13
    titlelabcol = 'silver'
    ax1.text(0, top_row + vspace, 'Cone\nID' , fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    
    ax1.text(0.537-0.457, top_row + vspace, 'Cone CME\nTime at 21.5 $r_S$', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.725-0.457, top_row + vspace, 'V\n[km/s]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.79-0.4572, top_row + vspace, 'Long\n[deg]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.851-0.457, top_row + vspace, 'Lat\n[deg]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.902-0.457, top_row + vspace, 'Width\n[deg]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    
    
    ax1.text(0.08+0.457, top_row + vspace, 'Hit\n %', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.129+0.457, top_row + vspace, 'Earth arrival time\n(median)', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.326+0.457, top_row + vspace, '+/-\n[hrs]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.387+0.457, top_row + vspace, 'V\n[km/s]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    ax1.text(0.445+0.457, top_row + vspace, '+/-\n[km/s]', fontsize = fsize,
             transform=ax1.transAxes, backgroundcolor = titlelabcol)
    

    
    for n in range(0,n_cme):
         if cme_hit[n] > ens_den_thresh:
             cme = cme_list[n]
             #work out CME insertion date
             cme_date = starttime + datetime.timedelta(days=cme.t_launch.to(u.day).value) 
             datestr = cme_date.strftime('%Y-%m-%d %H:%M')
             
             cme_arrival_median_dt = Time(cme_arrival_median[n], format = 'mjd').datetime
             arrivaldatestr = cme_arrival_median_dt.strftime('%Y-%m-%d %H:%M')
             arrivaluncertstr = str("{0:.1f}".format(cme_arrival_std[n]*24))
             if np.isnan(cme_arrival_std[n]):
                 arrivaluncertstr = '--'
             Vuncertstr = str("{0:.0f}".format(cme_v_std[n])) 
             if np.isnan(cme_v_std[n]):
                 Vuncertstr = '--'

             ax1.text(0, top_row - vspace*row_num, 'CME' + str(n+1) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             
             ax1.text(0.537-0.457, top_row - vspace*row_num, datestr , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.725-0.457, top_row - vspace*row_num, str(int(cme.v.value)) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.792-0.457, top_row - vspace*row_num, str(int(cme.longitude.to(u.deg).value)) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.851-0.457, top_row - vspace*row_num, str(int(cme.latitude.to(u.deg).value)) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.902-0.457, top_row - vspace*row_num, str(int(cme.width.to(u.deg).value)) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             
             ax1.text(0.08+0.457, top_row - vspace*row_num, str(int(cme_hit[n]*100)) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.129+0.457, top_row - vspace*row_num, arrivaldatestr , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.326+0.457, top_row - vspace*row_num, arrivaluncertstr , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.387+0.457, top_row - vspace*row_num, str(int(cme_v_median[n])) , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             ax1.text(0.450+0.457, top_row - vspace*row_num, Vuncertstr , fontsize = fsize,
                      transform=ax1.transAxes, backgroundcolor = labelcolour)
             

             row_num = row_num + 1
           
    
    #========================Vr map ===========================================
    ax4 = fig.add_subplot(gs[0, -1])
    pc = ax4.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi,
                    vr_map_rot.value, 
               shading='auto',vmin=250, vmax=650)
    ax4.plot(vr_longs*180/np.pi, E_lat*180/np.pi,'k--',label = 'Earth')
    ax4.plot(vr_longs*180/np.pi, E_lat*0,'k') #equator
    ax4.set_ylim([-90,90]); 
    ax4.set_xlim([0,360])
    ax4.set_xlabel('HEEQ longitude [deg]')
    ax4.set_ylabel('Latitude [deg]')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    #ax4.set_title(filename, FontSize = 14)
    ax4.legend()
    ax4.axes.xaxis.set_ticks([ 0,90,180, 270,360])
    ax4.axes.yaxis.set_ticks([-90,-45,0,45,90])
    ax4.axes.xaxis.set_ticklabels([-180, -90, 0, 90 ,180])
    #ax4.axes.yaxis.set_ticklabels([])
    plt.sca(ax4)
    #colorbar
    axins = inset_axes(ax4,
                        width="100%",  # width = 50% of parent_bbox width
                        height="10%",  # height : 5%
                        loc='upper right',
                        bbox_to_anchor=(0.28, 0.60, 0.72, 0.5),
                        bbox_transform=ax4.transAxes,
                        borderpad=0,)
    cb = fig.colorbar(pc, cax = axins, orientation = 'horizontal',  pad = -0.1)
    cb.ax.xaxis.set_ticks_position("top")
    ax4.text(0.02,1.05,r'$V_{SW}$ [km/s]' , 
            fontsize = 14, transform=ax4.transAxes, backgroundcolor = 'w')
    
    #========================Vr map: Add CMEs==================================
    #cbar = fig.colorbar(pc, ax=ax4, orientation = 'horizontal')
    #cbar.set_label(r'V$_{SW}$')
    #plot Earth at forecast time
    cr_f, cr_lon_init_f = Hin.datetime2huxtinputs(forecasttime)
    ax4.plot(180, np.mean(E_lat)*180/np.pi, 'ko', zorder = 1)
    ax4.plot([180, 180], [-90,90], 'k--')
    #plot CMEs
    if n_cme>0:
        nplot = 0
        for n, cme in enumerate(cme_list):
            
            longs, lats = long_lat_of_circle_on_sphere(cme.longitude.to(u.rad).value, 
                                                       cme.latitude.to(u.rad).value, 
                                                       cme.width.to(u.rad).value/2)
            
            # #ax4.plot(H._zerototwopi_(cr_lon_init_f - cme.longitude)*180/np.pi, cme.latitude*180/np.pi, 'wo')
            # stepSize = 0.1
            # #Generated vertices
            # positions = []  
            # r = cme.width.to(u.rad)/2
            # a = H._zerototwopi_(cme.longitude) *u.rad + np.pi*u.rad
            # b = cme.latitude.to(u.rad)
            # t = 0
            # while t < 2 * np.pi:
            #     positions.append( ( (r * np.cos(t) + a).value,
            #                       (r * np.sin(t) + b).value ))
            #     t += stepSize
            # pos = np.asarray(positions)
            
            # #wrap latitudes out of range
            # mask = pos[:,1] > np.pi/2
            # pos[mask,1] = -np.pi + pos[mask,1]
            # mask = pos[:,1] < - np.pi/2
            # pos[mask,1] = np.pi + pos[mask,1]
            
            h = ax4.plot(H._zerototwopi_(longs + np.pi)*180/np.pi, lats*180/np.pi,'.')
            if cme_hit[n] > ens_den_thresh:
                h[0].set_color(colours[nplot])
                nplot = nplot + 1
            else:
                h[0].set_color('silver') 

            
    #========================CME arrival speeds================================
    ax5 = fig.add_subplot(gs[1:, -1])
    #if n_cme >0:
        #plot the distributions of arrival speeds
        
    speeds = np.arange(200,1000,20)
    speeds_centres = speeds[0:-1]+10
    yspace = 1/2
    ax5.set_ylim((-(n_cme_hit-1)*yspace -0.01, yspace))
    
    nplot = 0
    for n in range(0,n_cme):
        if cme_hit[n] > ens_den_thresh:
            hist, bin_edges = np.histogram(cmearrivalspeeds[:,n], speeds) 
            h = ax5.plot(speeds_centres, hist/N_ens_cme - yspace*nplot, label='CME ' +str(n+1),
                         color = colours[nplot])
            nplot = nplot + 1
        #axs[n].set_title('CME ' + str(n+1))
        #if n==0:
        #    axs[n].set_ylabel('# ensemble members')
    ax5.set_xlabel( 'CME arrival speed [km/s]')
    ax5.set_ylabel( 'Ensemble density')
    ax5.axes.yaxis.set_ticks([])
    #ax5.legend(facecolor='silver', loc = 'upper left', framealpha=1)
    ax5.grid(True, axis = 'x')
    ax5.yaxis.set_label_position("right")
    
    
    fig.text(0.05,0.03,'HUXt1D using ' + runname, fontsize = 14)
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right =0.95, top =0.92,
                        wspace = 0.12, hspace =0.12)
    
    #make the CME speed plot a little smaller, to fit in the Vr map labels
    pos1 = ax5.get_position()
    pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height / 1.1] 
    ax5.set_position(pos2)
    
    return fig, [ax1, ax2, ax3, ax4, ax5]


def plot_multimodel_ensemble_dashboard(time,
                            ambient_ensemble_list, 
                            cme_list, cme_ensemble_list,
                            cme_arrival_list, cme_speed_list, 
                            filenames,
                            forecasttime, runnames,
                            confid_intervals = [5, 10, 32],
                            vlims = [300,850]):
    """
    Function to plot the mulit-model ensemble dashboard

    Parameters
    ----------
    time : times of HUXt output. output from ambient or CME ensemble
    forecasttime : Time fo forecast (datetime)
    confid_intervals : condifence intervals for ensemble plots [5, 10, 32].
    vlims : Plot limts for V timeseries[300,800].

    Returns
    -------
    fig, ax[array]: plot handles

    """
    
    colours = 'b','r','g','c','m','y','k','lime', 'indigo','orangered'
    
    linestyles = '-', '--', ':','-.'
    
    mpl.rc("axes", labelsize=14)
    mpl.rc("ytick", labelsize=14)
    mpl.rc("xtick", labelsize=14)
    mpl.rc("legend", fontsize=14)
    
    #HUXt output
    startdate = time[0].to_datetime()
    enddate = time[-1].to_datetime()
    
    #get properties from the array sizes
    nmodels = len(filenames)
    nt = len(time)
    
    if cme_list:
        n_cme = len(cme_arrival_list[0][0,:])  
        N_ens_cme = len(cme_ensemble_list[0][:,1]) 
        #create the CME multimodel ensemble
        huxtoutput_cme = np.concatenate(cme_ensemble_list, axis=0 )
        cmearrivalspeeds = np.concatenate(cme_speed_list, axis=0 )
        cmearrivaltimes = cme_arrival_list[0]
        #cmearrivalspeeds = cme_speeds[0]
        for i in range(1,nmodels):
            cmearrivaltimes = cmearrivaltimes + cme_arrival_list[i]
    else:
        n_cme = 0
    N_ens_amb = len(ambient_ensemble_list[0][:,1])
    
    
    #create the ambient multimodel ensemble
    huxtoutput_ambient = np.concatenate(ambient_ensemble_list, axis=0 )

    
    # plot it
    #===========================================================
    fig = plt.figure(figsize = (17,10))
    gs = fig.add_gridspec(3, 3)
    
    
    ax1 = fig.add_subplot(gs[0, :-1])
    plt.sca(ax1)
    plotconfidbands(time.to_datetime(),  huxtoutput_ambient, confid_intervals)
    #plot the individual unperturbed values
    for n in range(0, nmodels):
        ax1.plot(time.to_datetime(), ambient_ensemble_list[n][0,:], color = 'k', 
                 linestyle=linestyles[n], zorder = 1)
    #plt.xlabel('Time through CR [days]')
    ax1.set_ylim(vlims)
    ax1.set_ylabel(r'V$_{SW}$ [km/s]')
    #ax1.set_title('Ambient solar wind (N = ' + str(N_ens_amb*nmodels) + ')', fontsize = 14)
    ax1.text(0.02, 0.90,'Ambient solar wind (N = ' + str(N_ens_amb*nmodels) + ')', fontsize = 14,
             transform=ax1.transAxes, backgroundcolor = 'silver')
    yy = plt.gca().get_ylim(); 
    ax1.set_ylim(yy); ax1.set_xlim([startdate, enddate]);
    ax1.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                     color = 'silver', zorder = 2, alpha = 0.7)
    ax1.legend(facecolor='silver', loc = 'lower left', framealpha=1, ncol = 2)
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid(True, axis = 'x')
    ax1.grid(True, which = 'minor', axis = 'x')
    
    
    ax2 = fig.add_subplot(gs[1, :-1])
    if n_cme >0:
        plt.sca(ax2)
        plotconfidbands(time.to_datetime(), huxtoutput_cme, confid_intervals)
        #plot the individual unperturbed values
        h=[]
        for n in range(0, nmodels):
            temp = ax2.plot(time.to_datetime(), cme_ensemble_list[n][0,:], 
                               color = 'k', linestyle=linestyles[n],
                               label = 'Unperturbed ' + runnames[n], zorder = 1) 
            h.append(temp[0])
        #plt.xlabel('Time through CR [days]')
        ax2.set_ylim(vlims)
        ax2.set_ylabel(r'V$_{SW}$ [km/s]')
        #ax2.set_title('Ambient + CMEs (N = ' + str(N_ens_cme) + ')', fontsize = 14)
        ax2.text(0.02, 0.90,'Ambient + CMEs (N = ' + str(N_ens_cme*nmodels) + ')', fontsize = 14,
                 transform=ax2.transAxes, backgroundcolor = 'silver')
        yy = plt.gca().get_ylim(); 
        ax2.set_ylim(yy); ax2.set_xlim([startdate, enddate]); 
        ax2.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
        ax2.axes.xaxis.set_ticklabels([])
        ax2.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                         color = 'silver', zorder = 2, alpha = 0.7)
        ax2.legend(handles = h, facecolor='silver', loc = 'lower left', 
                    framealpha=1, ncol=1)
        ax2.xaxis.set_minor_locator(mdates.DayLocator())
        ax2.grid(True, axis = 'x')
        ax2.grid(True, which = 'minor', axis = 'x')
    
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5,'No CMEs or no Cone CME file', fontsize = 14,
                 transform=ax1.transAxes, backgroundcolor = 'silver')
    
    ax3 = fig.add_subplot(gs[2, :-1])
    if n_cme > 0:
        #smooth the arrival time ensembles for plotting
        pd_cme_arrival = pd.DataFrame({ 'Date': time.to_datetime()}) 
        pd_cme_arrival.index = pd_cme_arrival['Date']
        for i in range(0,n_cme):
            pd_cme_arrival['CME'+str(i)] = cmearrivaltimes[:,i]
            pd_cme_arrival['CME'+str(i)+'_smooth'] = pd_cme_arrival['CME'+str(i)].rolling(10,center=True).mean()
            #smooth each individual model
            for n in range(0, nmodels):
                pd_cme_arrival['CME'+str(i)+'_model'+str(n)] = cme_arrival_list[n][:,i]
                pd_cme_arrival['CME'+str(i)+'_model'+str(n)+'_smooth'] = \
                    pd_cme_arrival['CME'+str(i)+'_model'+str(n)].rolling(10,center=True).mean()
       
        
       
        
        #plot the arrival time distributions
        plt.sca(ax3)
        for n in range(0,n_cme):
            #ax3.plot(tdata_cme.to_datetime(), cmearrivaltimes[:,n],label='CME ' +str(n+1))
            h = ax3.plot( pd_cme_arrival['CME'+str(n)+'_smooth'] ,label='CME ' +str(n+1),
                         color = colours[n], zorder = 1)
        ax3.set_ylabel( 'Ensemble density')
        ax3.axes.yaxis.set_ticks([])
        yy = plt.gca().get_ylim(); 
        ax3.set_ylim(yy); ax3.set_xlim([startdate, enddate]);
        ax3.plot([forecasttime,forecasttime],yy,'silver',zorder = 2)
        #ax3.set_title('CME front arrival time (N = ' + str(N_ens_cme*nmodels) + ')', fontsize = 14)
        ax3.text(0.02, 0.90,'CME front arrival time (N = ' + str(N_ens_cme*nmodels) + ')', fontsize = 14,
                 transform=ax3.transAxes, backgroundcolor = 'silver')
        ax3.fill_between([startdate, forecasttime], [yy[0], yy[0]], [yy[1], yy[1]], 
                         color = 'silver',zorder=2, alpha=0.7)
        ax3.legend(facecolor='white', loc = 'upper right', framealpha=1)
        #find and plot the modal arrival times
        for n in range(0, nmodels):
            for i in range(0,n_cme):
                tmode = pd_cme_arrival['CME'+str(i)+'_model'+str(n)].idxmax()
                ax3.plot([tmode, tmode],yy, color = colours[i], 
                         linestyle=linestyles[n], zorder = 1) 
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval = 2))
        ax3.xaxis.set_minor_locator(mdates.DayLocator())
        ax3.grid(True, axis = 'x')
        ax3.grid(True, which = 'minor', axis = 'x')

    
    
    ax4 = fig.add_subplot(gs[0, -1])
    ax4.set_xlim([0,1]); ax4.set_ylim([0,1])
    dy = 0.2
    ax4.text(0.03, 1 - dy, 'HUXt1D multi-model ensemble', fontsize = 14)
    for n in range(0,nmodels):
        ax4.plot([0.05, 0.15], [1 - dy*(n+2), 1- dy*(n+2)] , color = 'k', linestyle=linestyles[n])
        ax4.text(0.2, 1- dy*(n+2), filenames[n], fontsize = 14)
    ax4.set_axis_off()    
    
    ax5 = fig.add_subplot(gs[1:, -1])
    if n_cme > 0:
        #plot the distributions of arrival speeds
        speeds = np.arange(200,1000,20)
        speeds_centres = speeds[0:-1]+10
        yspace = N_ens_cme/5
        for n in range(0,n_cme):
            hist, bin_edges = np.histogram(cmearrivalspeeds[:,n], speeds) 
            ax5.plot(speeds_centres,hist - yspace*n, label='CME ' +str(n+1),
                         color = colours[n])
        #find and plot the modal speeds for each individual model
        for n in range(0, nmodels):
            for i in range(0,n_cme):
                hist, bin_edges = np.histogram(cme_speed_list[n][:,i], speeds) 
                imax  = np.argmax(hist)
                if (hist[imax] > 0):
                    ax5.plot([speeds_centres[imax], speeds_centres[imax]],
                             [- yspace*i, - yspace*(i-1)], 
                              color = colours[i], linestyle=linestyles[n])
        
        
        ax5.set_xlabel( 'CME arrival speed [km/s]')
        ax5.set_ylabel( 'Ensemble density')
        ax5.axes.yaxis.set_ticks([])
        #ax5.legend(facecolor='silver', loc = 'upper left', framealpha=1)
        ax5.grid(True, axis = 'x')
    
    
    #fig.text(0.05,0.03,'HUXt1D using ' + filename, fontsize = 14)
    fig.subplots_adjust(left = 0.05, bottom = 0.1, right =0.95, top =0.92,
                        wspace = 0.12, hspace =0.12)
    
    return fig, [ax1, ax2, ax3, ax4, ax5]

def sweep_ensemble_run(forecasttime, savedir =[], 
                       wsafilepath = "", pfssfilepath = "", 
                       cortomfilepath = "", dumfricfilepath = "",
                       conefilepath = "",
                       cme_buffer_time = 5*u.day,
                       N_ens_amb = 100,
                       N_ens_cme = 500,
                       simtime = 12*u.day, 
                       lat_rot_sigma = 5*np.pi/180*u.rad,
                       lat_dev_sigma = 2*np.pi/180*u.rad,
                       long_dev_sigma = 2*np.pi/180*u.rad,
                       cme_v_sigma_frac = 0.1,
                       cme_width_sigma_frac = 0.1,
                       cme_thick_sigma_frac = 0.1,
                       cme_lon_sigma = 10*u.deg,
                       cme_lat_sigma = 10*u.deg,
                       r_in = 21.5*u.solRad,      
                       r_out = 230*u.solRad ,    
                       dt_scale = 4,
                       confid_intervals = [5, 10, 32],
                       vlims = [300,850]):
    
    """
    A function to generate all the ambient and CME ensembles and produce the 
    multi-model ensembles
    """
    
    #start the run a few days early, to allow CMEs to reach Earth
    starttime = forecasttime - datetime.timedelta(days=cme_buffer_time.to(u.day).value) 
    
    
    #set up the parameter lists
    vr_map_list = []
    lon_list = []
    lat_list = []
    filename_list = []
    run_list = []
    
    
    #load the solar wind speed maps with the various bespoke readers
    #================================================================
    if os.path.exists(wsafilepath):
        wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
            = Hin.get_WSA_maps(wsafilepath)
        vr_map_list.append(wsa_vr_map)
        lon_list.append(vr_longs)
        lat_list.append(vr_lats)
        filename_list.append(os.path.basename(wsafilepath))
        run_list.append('WSA')
        
    if os.path.exists(pfssfilepath):
         pfss_vr_map, vr_longs, vr_lats, br_map, br_lats, br_longs \
             = Hin.get_PFSS_maps(pfssfilepath)
         vr_map_list.append(pfss_vr_map)
         lon_list.append(vr_longs)
         lat_list.append(vr_lats)
         filename_list.append(os.path.basename(pfssfilepath))
         run_list.append('PFSS')  
    
    if os.path.exists(dumfricfilepath):
         dumfric_vr_map, vr_longs, vr_lats, br_map, br_lats, br_longs \
             = Hin.get_PFSS_maps(dumfricfilepath)
         vr_map_list.append(dumfric_vr_map)
         lon_list.append(vr_longs)
         lat_list.append(vr_lats)
         filename_list.append(os.path.basename(dumfricfilepath))
         run_list.append('DUMFRIC')  
         
         
    if os.path.exists(cortomfilepath):
         cortom_vr_map,  vr_longs, vr_lats \
             = Hin. get_CorTom_vr_map(cortomfilepath)
             
         # CorTOm is at 8 rS. Map this out to 21.5 rS
         cortom_vr_map_21 = Hin.map_vmap_inwards(cortom_vr_map, vr_lats, vr_longs, 
                                            8*u.solRad, r_in)
         vr_map_list.append(cortom_vr_map_21)
         lon_list.append(vr_longs)
         lat_list.append(vr_lats)
         filename_list.append(os.path.basename(cortomfilepath))
         run_list.append('CorTom') 
    #need to add dumfric reader here
    
         
    #run the ambient ensemble 
    #==================================================================
    huxtinput_ambient_list = []
    huxtoutput_ambient_list = []
    if run_list: # check if there's any data been loaded in
        for listno in range(0, len(run_list)):
            #generate the ambient ensemble
            ambient_time, this_huxtinput_ambient, this_huxtoutput_ambient = \
                                 ambient_ensemble(vr_map_list[listno], 
                                                  lon_list[listno], 
                                                  lat_list[listno],
                                                  starttime,
                                                  simtime = simtime,
                                                  N_ens_amb = N_ens_amb,
                                                  lat_rot_sigma = lat_rot_sigma,
                                                  lat_dev_sigma = lat_dev_sigma,
                                                  long_dev_sigma = long_dev_sigma,
                                                  r_in = r_in, r_out = r_out,     
                                                  dt_scale = dt_scale)
            #store the data                     
            huxtinput_ambient_list.append(this_huxtinput_ambient)
            huxtoutput_ambient_list.append(this_huxtoutput_ambient)
    else:
        return 0
            
        

    # CME ensembles
    #=========================
    cme_list = []
    #Load the CME parameters
    if os.path.exists(conefilepath):
        cme_list = Hin.ConeFile_to_ConeCME_list_time(conefilepath, starttime)
    
    huxtoutput_cme_list = []
    cmearrivaltimes_list = []
    cmearrivalspeeds_list = []
    if cme_list:
        #generate the CME ensembles
        for listno in range(0, len(run_list)):
            
            #create the CME ensemble
            cme_time, this_huxtoutput_cme, this_cmearrivaltimes, this_cmearrivalspeeds  = \
                cme_ensemble(huxtinput_ambient_list[listno], 
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
            #store the data
            huxtoutput_cme_list.append(this_huxtoutput_cme)
            cmearrivaltimes_list.append(this_cmearrivaltimes)
            cmearrivalspeeds_list.append(this_cmearrivalspeeds)
    else:
        print('No CMEs or no Cone File found. Ambient ensembles only.')
        for listno in range(0, len(run_list)):
            huxtoutput_cme_list.append([])
            cmearrivaltimes_list.append([])
            cmearrivalspeeds_list.append([])
             
        
    # Plot and save the individual ensemble dashboards
    for listno in range(0, len(run_list)):
        fig, axs = plot_ensemble_dashboard(ambient_time, vr_map_list[listno],
                                           lon_list[listno], 
                                           lat_list[listno],
                                           cme_list,
                                    huxtoutput_ambient_list[listno],
                                    huxtoutput_cme_list[listno],
                                    cmearrivaltimes_list[listno],
                                    cmearrivalspeeds_list[listno], 
                                    forecasttime,
                                    filename = filename_list[listno],
                                    runname = run_list[listno],
                                    confid_intervals = confid_intervals,
                                    vlims = vlims)
        if savedir:
            fname = os.path.join(savedir, 'dashboard_' + run_list[listno] + '.png')
            fig.savefig(fname)


    # Produce and save the multi-model ensemble plot
    fig, axs = plot_multimodel_ensemble_dashboard(ambient_time,
                                huxtoutput_ambient_list, 
                                cme_list,
                                huxtoutput_cme_list,
                                cmearrivaltimes_list, cmearrivalspeeds_list, 
                                filename_list, forecasttime, run_list)
    if savedir:
        fname = os.path.join(savedir, 'dashboard_multimodel.png')
        plt.savefig(fname)
    
    return 1


def getMetOfficeWSA(startdate, enddate, datadir = ''):
    #downloads the most recent WSA output for a given time 
    #window from the Met Office system. Requires an API key to be set as
    #a system environment variable
    #saves WSA files to datadir, which defaults tot he current directory

    #outputs the filepaths to the  WSA files
    
    version = 'v1'
    api_key = os.getenv("API_KEY")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    
    startdatestr = startdate.strftime("%Y-%m-%dT%H:%M:%S")
    enddatestr = enddate.strftime("%Y-%m-%dT%H:%M:%S")
    
    request_url = url_base + "/" + version + "/data/swc-enlil-wsa?from=" + startdatestr + "&to=" + enddatestr
    response = requests.get(request_url,  headers={"accept" : "application/json", "apikey" : api_key })
    
    success = False
    wsafilepath = ''
    model_time = ''
    if response.status_code == 200:
    
        #Convert to json
        js = response.json()
        nfiles=len(js['data'])
        #print('Found: ' + str(nfiles))
                
        #get the latest file
        i = nfiles - 1
        found_wsa = False
    
        #start with the most recent file and work back in time
        while i > 0:
            model_time = js['data'][i]['model_run_time']
            wsa_file_name = js['data'][i]['gong_file']
            
            wsa_file_url = url_base + "/" + version + "/" + wsa_file_name

            
            if not found_wsa:
                response_wsa = requests.get(wsa_file_url,  headers={ "apikey" : api_key })
                if response_wsa.status_code == 200:
                    wsafilepath = os.path.join(datadir, wsa_file_name)
                    open(wsafilepath,"wb").write(response_wsa.content)
                    found_wsa = True
            i = i - 1
            if found_wsa:
                success = True
                break
    #else: 
        #print('Found: 0')
        
    return success, wsafilepath, model_time

def getMetOfficeCone(startdate, enddate, datadir = ''):
    #downloads the most recent coneCME files for a given time range
    #window from the Met Office system. Requires an API key to be set as
    #a system environment variable
    #saves wsa and cone files to datadir, which defaults tot he current directory
    #outputs the filepath to the cone files
    
    version = 'v1'
    api_key = os.getenv("API_KEY")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    
    startdatestr = startdate.strftime("%Y-%m-%dT%H:%M:%S")
    enddatestr = enddate.strftime("%Y-%m-%dT%H:%M:%S")
    
    request_url = url_base + "/" + version + "/data/swc-enlil-wsa?from=" + startdatestr + "&to=" + enddatestr
    response = requests.get(request_url,  headers={"accept" : "application/json", "apikey" : api_key })
    
    success = False
    conefilepath = ''
    model_time = ''
    if response.status_code == 200:
    
        #Convert to json
        js = response.json()
        nfiles=len(js['data'])
        #print('Found: ' + str(nfiles))
                
        #get the latest file
        i = nfiles - 1
        found_cone = False
    
        
        #start with the most recent file and work back in time
        while i > 0:
            model_time = js['data'][i]['model_run_time']
            cone_file_name = js['data'][i]['cone_file']
            
            cone_file_url = url_base + "/" + version + "/" + cone_file_name
            
            if not found_cone: 
                response_cone = requests.get(cone_file_url,  headers={ "apikey" : api_key })
                if response_cone.status_code == 200:
                    conefilepath = os.path.join(datadir, cone_file_name)
                    open(conefilepath,"wb").write(response_cone.content)
                    found_cone = True
            i = i - 1
            if found_cone:
                success = True
                break
    #else: 
        #print('Found: 0')
        
    return success, conefilepath, model_time


def getMetOfficePFSS(startdate, enddate, datadir = ''):
    #downloads the most recent PFSS files for a given time 
    #window from the Met Office system. Requires an API key to be set as
    #a system environment variable
    #saves PFSS files to datadir, which defaults to the current directory
    #outputs the filepaths to the PFSS file

    pfssfilepath = ''
    modeltime = ''
    
    api_key = os.getenv("API_KEY")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    version = "v1"
    request_url = url_base+'/'+version+'/output'
    
    #get the list of files in the pfss_latest directory
    response = requests.get(request_url+'/list?directory=pfss_results', \
                     headers={"accept" : "*/*", \
                              "apikey" : api_key}
                              )
    success = False
    #extract the filenames
    if response.ok:
        response_dict = json.loads(response.content)
        file_list = response_dict["objects"]
 
        #extract the date info for each file
        date_list = []
        date_time_str = []
        for count, filename in enumerate(file_list):
            date_time_str.append(filename.split("pfss")[2][0:11])
            date_list.append(datetime.datetime.strptime(date_time_str[count], '%Y%m%d.%H'))
            
        
        #find the most recent date within the required date range
        filtered_dates = [date for date in date_list if startdate <= date <= enddate]
        
        if filtered_dates:
            most_recent_date = max(filtered_dates)
            #print("Most recent date:", most_recent_date)
        else:
            print("No PFSS data found within date range.")
            success = False
            return success, pfssfilepath, modeltime
        #get the list index
        index = date_list.index(most_recent_date)
        
        #get the associated file from the API
        response_pfss = requests.get(request_url + "?object_name=" + file_list[index]
                                     ,  headers={ "apikey" : api_key })
        pfss_filename = file_list[index].split("/")[1]
        if response_pfss.status_code == 200:
                         pfssfilepath = os.path.join(datadir, pfss_filename)
                         url = response_pfss.content.strip(b'"').decode('utf-8')
                         response = requests.get(url)

                         # Save the file
                         with open(pfssfilepath, 'wb') as f:
                             f.write(response.content)
                             
                         success = True
                         modeltime = date_time_str[index]

    return success, pfssfilepath, modeltime



def getMetOfficeCorTom(startdate, enddate, datadir = ''):
    #downloads the most recent CorTom files for a given time 
    #window from the Met Office system. Requires an API key to be set as
    #a system environment variable
    #saves PFSS files to datadir, which defaults to the current directory
    #outputs the filepaths to the CorTom file

    cortomfilepath = ''
    modeltime = ''
    
    api_key = os.getenv("API_KEY")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    version = "v1"
    request_url = url_base+'/'+version+'/output'
    
    #get the list of files in the pfss_latest directory
    response = requests.get(request_url+'/list?directory=cortom_results', \
                     headers={"accept" : "*/*", \
                              "apikey" : api_key}  )
    success = False
    #extract the filenames
    if response.ok:
        response_dict = json.loads(response.content)
        file_list = response_dict["objects"]
 
        #extract the date info for each file
        date_list = []
        date_time_str = []
        for count, filename in enumerate(file_list):
            date_time_str.append(filename.split("_")[3][0:8])
            date_list.append(datetime.datetime.strptime(date_time_str[count], '%Y%m%d'))
            
        
        #find the most recent date within the required date range
        filtered_dates = [date for date in date_list if startdate <= date <= enddate]
        
        if filtered_dates:
            most_recent_date = max(filtered_dates)
            #print("Most recent date:", most_recent_date)
        else:
            print("No CorTom data found within date range.")
            success = False
            return success, cortomfilepath, modeltime
        #get the list index
        index = date_list.index(most_recent_date)
        
        #get the associated file from the API
        response_cortom = requests.get(request_url + "?object_name=" + file_list[index]
                                     ,  headers={ "apikey" : api_key })
        cortom_filename = file_list[index].split("/")[1]
        if response_cortom.status_code == 200:
                         cortomfilepath = os.path.join(datadir, cortom_filename)
                         url = response_cortom.content.strip(b'"').decode('utf-8')
                         response = requests.get(url)

                         # Save the file
                         with open(cortomfilepath, 'wb') as f:
                             f.write(response.content)
                             
                         success = True
                         modeltime = date_time_str[index]

    return success, cortomfilepath, modeltime




def getMetOfficeDumfric(startdate, enddate, datadir = ''):
    #downloads the most recent DUMFRIC files for a given time 
    #window from the Met Office system. Requires an API key to be set as
    #a system environment variable
    #saves PFSS files to datadir, which defaults to the current directory
    #outputs the filepaths to the DUMFRIC file

    pfssfilepath = ''
    modeltime = ''
    
    api_key = os.getenv("API_KEY")
    url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
    version = "v1"
    request_url = url_base+'/'+version+'/output'
    
    #get the list of files in the pfss_latest directory
    response = requests.get(request_url+'/list?directory=dumfric_results', \
                     headers={"accept" : "*/*", \
                              "apikey" : api_key}
                              )
    success = False
    #extract the filenames
    if response.ok:
        response_dict = json.loads(response.content)
        file_list = response_dict["objects"]
 
        #extract the date info for each file
        date_list = []
        date_time_str = []
        for count, filename in enumerate(file_list):
            date_time_str.append(filename.split("windbound_b")[1][0:11])
            date_list.append(datetime.datetime.strptime(date_time_str[count], '%Y%m%d.%H'))
            
        
        #find the most recent date within the required date range
        filtered_dates = [date for date in date_list if startdate <= date <= enddate]
        
        if filtered_dates:
            most_recent_date = max(filtered_dates)
            #print("Most recent date:", most_recent_date)
        else:
            print("No DUMFRIC data found within date range.")
            success = False
            return success, pfssfilepath, modeltime
        #get the list index
        index = date_list.index(most_recent_date)
        
        #get the associated file from the API
        response_pfss = requests.get(request_url + "?object_name=" + file_list[index]
                                     ,  headers={ "apikey" : api_key })
        pfss_filename = file_list[index].split("/")[1]
        if response_pfss.status_code == 200:
                         pfssfilepath = os.path.join(datadir, pfss_filename)
                         url = response_pfss.content.strip(b'"').decode('utf-8')
                         response = requests.get(url)

                         # Save the file
                         with open(pfssfilepath, 'wb') as f:
                             f.write(response.content)
                             
                         success = True
                         modeltime = date_time_str[index]

    return success, pfssfilepath, modeltime

def get_CorTomPKL_vr_map(filepath):
    """
    A function to load, read and process CorTom density output in PKL format to 
    provide HUXt V boundary conditions as lat-long maps, 
    Maps returned in native resolution, not HUXt resolution.
    Maps are not transformed - make sure the CorTom maps are Carrington maps

    Args:
        filepath: String, The filepath for the CorTom.txt file

    Returns:
        vr_map: np.array, Solar wind speed as a Carrington longitude-latitude map. In km/s
        vr_lats: np.array, The latitudes for the Vr map, in radians from trhe equator
        vr_longs: np.array, The Carrington longitudes for the Vr map, in radians
        phi: meshgrid og longitudes
        theta: mesh grid of latitudes

    """
    
    with open(filepath, "rb") as file:
        data = pickle.load(file)
        
    lat = (np.pi/2 - data['colat'])*u.rad
    lon = (np.pi + data['lon'])*u.rad
    v = data['velocity']* u.km/u.s


    return v.T, lon, lat



def get_CorTom_long_profile(filepath, lat=0.0 * u.deg):
    """
    Function to read and process CorTom output to provide a longitude profile at a specified latitude
    of the solar wind speed for use as boundary conditions in HUXt.

    Args:
        filepath: A complete path to the CorTom data file
        lat: Latitude at which to extract the longitudinal profile, measure up from equator. Float with units of deg

    Returns:
        vr_in: Solar wind speed as a function of Carrington longitude at solar equator.
               Interpolated to the default HUXt longitudinal grid. np.array (NDIM = 1) in units of km/s
    """
    assert (lat >= -90.0 * u.deg)
    assert (lat <= 90.0 * u.deg)
    assert(os.path.isfile(filepath))

    vr_map, lon_map, lat_map = get_CorTomPKL_vr_map(filepath)

    # Extract the value at the given latitude
    vr = np.zeros(lon_map.shape)
    for i in range(lon_map.size):
        vr[i] = np.interp(lat.to(u.rad).value, lat_map.to(u.rad).value, vr_map[:, i].value)

    # Now interpolate on to the HUXt longitudinal grid
    #lon, dlon, nlon = H.longitude_grid(lon_start=0.0 * u.rad, lon_stop=2 * np.pi * u.rad)
    #vr_in = np.interp(lon.value, lon_map.value, vr) * u.km / u.s

    return vr * u.km / u.s


# #find the HSEs in  a given ensemble member

# n = 0
# dv = 100
# dt = 2
# plt.figure()
# plt.plot(time.to_datetime(), huxtoutput_cme[n,:])

# clear times
# ntimes=1;
# for i=1:L-1
#     event=0;
#     for j=1:nahead
#         if (i+j)<=L
#             if (data(i,5)-data(i+j,5))>=deltav
#                 times(ntimes,1)=data(i,1);
#                 times(ntimes,2)=data(i+j,1);
#                 event=1;
#             end
#         end
#     end
#     if event==1; ntimes=ntimes+1; end
# end


# %There will be mutliple references to single events in "times". merge the
# %multiple events together.
# clear final
# final(1,:)=times(1,:); k=1;
# for i=2:length(times(:,1))
#     if (times(i,1)-final(k,2))<=gapt
#         final(k,2)=times(i,2);
#     else 
#         k=k+1;
#         final(k,:)=times(i,:);
#     end
# end

# %find the properties of each event

# %first calculate the gradients of the speed
# clear temp vgrads
# n=round(gradavg/res);
# temp=data(1+n:L,:);
# vgrads(1:L-n)=(data(1:L-n,5)-temp(:,5))/(n*res);
# vgrads(L-n+1:L)=NaN;

# for i=1:length(final(:,1))
#     %find the start of the maximum gradient by searching +/- deltat either 
#     %side of the boundaries
#     if (final(i,1)-deltat) >= data(1,1)
#         startno=stdmjd2datano(final(i,1)-deltat,data);
#     else; startno=1; end;
#     if final(i,2)+deltat <= data(L,1)
#         stopno=stdmjd2datano(final(i,2)+deltat,data);
#     else; stopno=L; end;
#     [maxg maxgn]=max(vgrads(startno:stopno));
#     final(i,5)=data(maxgn+startno-1,1);
#     final(i,6)=maxg;
    
#     %find the maximum speed upto deltat after the end of the max gradient
#     maxvsno=startno+maxgn-1+n; if maxvsno < 1; maxvsno=1; end;
#     maxvfno=maxvsno+ceil(deltat/res); if maxvfno >L; maxvfno=L; end;
#     [maxv maxvn]=min(data(maxvsno:maxvfno,5));
#     final(i,3)=data(maxvn+maxvsno-1,1);
#     final(i,4)=maxv;
    
#     %find the minimum speed upto deltat before the start of the max gradient
#     minvfno=startno+maxgn-1-n; if minvfno>L; minvfno=L; end;
#     minvsno=maxvsno-ceil(deltat/res); if minvsno<1; minvsno=1; end;
#     [minv minvn]=max(data(minvsno:minvfno,5));
#     final(i,7)=data(minvn+minvsno-1,1);
#     final(i,8)=minv;
    
# end

# y=final;

        
def move_data_to_API(filelist):
      print("Moving data")
      api_key = os.getenv("API_KEY")
    
      url_base = "https://gateway.api-management.metoffice.cloud/swx_swimmr_s4/1.0"
      version = "v1"
    
      request_url = url_base+'/'+version+'/output'
    
      print("Files:")
      print(filelist)
    
      for f in filelist:
    
        filename = f 
    
        print("Transferring file: " + filename)
    
        data_file = open(filename, 'rb')
      
        response = requests.put(request_url, \
                   params = {"object_name": os.path.basename(filename)},\
                   headers={"apikey" : api_key},\
                            data=data_file\
                            )
    
        print(response)
    
      print("File transfer done.")
