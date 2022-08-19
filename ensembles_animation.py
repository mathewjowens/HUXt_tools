# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 13:01:07 2022

@author: mathewjowens
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import astropy.units as u
import os
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import huxt as H
import huxt_inputs as Hin
import huxt_ensembles as Hens

#demo of the input ensemble generation

cwd = os.path.abspath(os.path.dirname(__file__))
#the filenames should then be generated from the forecast date
wsafilepath =  os.path.join(cwd, 'data', '2022-02-24T22Z.wsa.gong.fits')
Nens = 100
lat_rot_sigma = 5*np.pi/180*u.rad
lat_dev_sigma = 2*np.pi/180*u.rad
long_dev_sigma = 2*np.pi/180*u.rad
Elat = -7.25*u.deg

wsa_vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
    = Hin.get_WSA_maps(wsafilepath)
    
    
#create the mesh grid
phi, theta = np.meshgrid(vr_longs, vr_lats, indexing = 'xy')


lats_E = np.ones(len(vr_longs))*Elat.to(u.rad)





vr_E = Hens.interp2d(vr_longs, lats_E, wsa_vr_map, phi, theta)
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
lat_ensemble = np.ones((Nens,len(vr_longs)))
long_ensemble = np.ones((Nens,len(vr_longs)))
#br_ensemble = np.ones((Nens,len(br_longs)))
#first ensemble member is the undeviated value
vr_ensemble[0,:] = vr_E
lat_ensemble[0,:] = lats_E
long_ensemble[0,:] = vr_longs
#br_ensemble[0,:] = br_E

#for each set of random params, generate a V long series
for i in range(1, Nens):
    this_lat = lats_E.value + lat_rots[i] * np.sin(vr_longs.value + long_rots[i]) +lat_devs[i]
    this_long = H._zerototwopi_(vr_longs.value + long_devs[i])
    
    #sort longitude into ascending order
    order = np.argsort(this_long)
    
    v = Hens.interp2d(this_long[order], this_lat[order], wsa_vr_map, phi, theta)
    vr_ensemble[i,:] = v
    lat_ensemble[i,:] = this_lat[order]
    long_ensemble[i,:] = this_long[order]
    
    
 
#run each V(21.5) through HUXt
#==============================
#get the huxt params for the start time

#Use the HUXt ephemeris data to get Earth lat over the CR
#========================================================
dummymodel = H.HUXt(v_boundary=np.ones((128))*400* (u.km/u.s), simtime=27.27*u.day, 
                   dt_scale=4, 
                   lon_out=0.0*u.deg,
                   r_min=21.5*u.solRad)

#retrieve a bodies position at each model timestep:
earth = dummymodel.get_observer('earth')
#get Earth r
E_r = np.mean(earth.r)

#find the gridcell which corresponds to this.
nE_r = np.argmin(abs(dummymodel.r - E_r))

#resample the ensemble to 128 longitude bins
vr128_ensemble = np.ones((Nens,128))  
dphi = 2*np.pi/128
phi128 = np.linspace(dphi/2, 2*np.pi - dphi/2, 128)
for i in range(0, Nens):
    vr128_ensemble[i,:] = np.interp(phi128,
                  vr_longs.value,vr_ensemble[i,:])
    
#==============================================================================
#run huxt with the ambient ensemble
#==============================================================================
nsteps = len(dummymodel.time_out)
huxtoutput_ambient = np.ones((Nens,nsteps))
#huxtoutput = []
for i in range(0,Nens):
    model = H.HUXt(v_boundary=vr128_ensemble[i]* (u.km/u.s), simtime=27.27*u.day, 
                   dt_scale=4, 
                   lon_out=0.0*u.deg, 
                   r_min=21.5*u.solRad)
    model.solve([]) 
    
    #find Earth location and extract the time series
    #huxtoutput.append(HA.get_earth_timeseries(model))
    
    #get value at Earth
    huxtoutput_ambient[i,:] = model.v_grid[:,nE_r,0]


    
# <codecell>
def drawframe(t):
    nframe = int(np.floor( Nens* t/duration)) + 1

    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    
    
    im = axs[0,0].pcolor(vr_longs.to(u.deg).value, vr_lats.to(u.deg).value, wsa_vr_map.value,
                         norm=plt.Normalize(200,750)) 
    axs[0,0].set_ylim(-60, 60) 
    axs[0,0].set_xlim(0, 360) 
    axs[0,0].set_ylabel('Lat [deg]')
    axs[0,0].set_yticks([-60, 0, 60])
    axs[0,0].set_xticks([ 0, 90, 180, 270, 360])
    
    #plot additional ensemble members
    for n in range(0,nframe):
        #plot Earth location
        if n == 0:
            axs[0,0].plot(long_ensemble[n,:]*180/np.pi,
                          lat_ensemble[n,:]*180/np.pi, 'r')
        elif n == nframe-1:
            axs[0,0].plot(long_ensemble[n,:]*180/np.pi,
                          lat_ensemble[n,:]*180/np.pi, 'k')
        else:
            axs[0,0].plot(long_ensemble[n,:]*180/np.pi,
                          lat_ensemble[n,:]*180/np.pi, 'grey')
                    
    
    axins = inset_axes(axs[0,0],
                        width="100%",  # width = 50% of parent_bbox width
                        height="15%",  # height : 5%
                        loc='upper right',
                        bbox_to_anchor=(0.45, 0.69, 0.5, 0.5),
                        bbox_transform=axs[0,0].transAxes,
                        borderpad=0,)
    
    #ax.plot([0, 360],[7.5, 7.5],'w--'); ax.plot([0, 360],[-7.5, -7.5],'w--');
    
    cb = fig.colorbar(im, cax = axins, orientation = 'horizontal',  pad = -0.1)
    cb.ax.tick_params(labelsize=12)
    axs[0,0].text(0.02,1.11,r'$V_{SW} (21.5 r_S)$ [km/s]                                                           ',
        fontsize = 14, transform=axs[0,0].transAxes, backgroundcolor = 'w')
    
    #plot V(long) at Sun for Earth
    axs[1,0].set_ylim(200, 700) 
    axs[1,0].set_xlim(0, 360) 
    axs[1,0].set_xticks([ 0, 90, 180, 270, 360])
    
    axs[1,0].set_ylabel(r'$V_{SW}$ [km/s]')
    axs[1,0].set_xlabel('Longitude [deg]')
    
    #plot additional ensemble members
    for n in range(0,nframe):
        #plot Earth location
        if n == 0:
            axs[1,0].plot(long_ensemble[n,:]*180/np.pi,
                          vr_ensemble[n,:], 'r')
        elif n == nframe-1:
            axs[1,0].plot(long_ensemble[n,:]*180/np.pi,
                          vr_ensemble[n,:], 'k')
        else:
            axs[1,0].plot(long_ensemble[n,:]*180/np.pi,
                          vr_ensemble[n,:], 'grey')
            
    axs[1,0].text(0.05, 0.05, r'$21.5 r_S$', fontsize = 14,  transform=axs[1,0].transAxes)
    
    
    #plot V(time) at Sun for Earth
    
    
    dt = 27.27/len(vr_longs)
    t = np.arange(0,27.27,dt)
    
    axs[0,1].set_ylim(200, 700) 
    axs[0,1].set_xlim(0, t[-1]) 
    axs[0,1].set_xticks([ 0, 10, 20])
    axs[0,1].yaxis.tick_right()
    axs[0,1].yaxis.set_label_position('right')
    axs[0,1].set_ylabel(r'$V_{SW}$ [km/s]')
    
    #plot additional ensemble members
    for n in range(0,nframe):
        #plot Earth location
        if n == 0:
            axs[0,1].plot(t,
                          np.flipud(vr_ensemble[n,:]), 'r')
        elif n == nframe-1:
            axs[0,1].plot(t,
                          np.flipud(vr_ensemble[n,:]), 'k')
        else:
            axs[0,1].plot(t,
                          np.flipud(vr_ensemble[n,:]), 'grey')
    axs[0,1].text(0.05, 0.05, r'$21.5 r_S$', fontsize = 14,  transform=axs[0,1].transAxes)        
    
    
    #plot V(time) at Earth
    
    axs[1,1].set_ylim(300, 800) 
    axs[1,1].set_xlim(0, t[-1]) 
    axs[1,1].set_xticks([ 0, 10, 20])
    axs[1,1].yaxis.tick_right()
    axs[1,1].yaxis.set_label_position('right')
    axs[1,1].set_ylabel(r'$V_{SW}$ [km/s]')
    axs[1,1].set_xlabel('Time [days]')
    
    
    #plot additional ensemble members
    for n in range(0,nframe):
        #plot Earth location
        if n == 0:
            axs[1,1].plot(model.time_out.to(u.day).value,
                          huxtoutput_ambient[n,:], 'r')
        elif n == nframe-1:
            axs[1,1].plot(model.time_out.to(u.day).value,
                          huxtoutput_ambient[n,:], 'k')
        else:
            axs[1,1].plot(model.time_out.to(u.day).value,
                          huxtoutput_ambient[n,:], 'grey')
                            
    axs[1,1].text(0.05, 0.05, r'1 AU', fontsize = 14,  transform=axs[1,1].transAxes)
    
    frame = mplfig_to_npimage(fig)
    plt.close('all')
    return frame

duration = 10
filename = 'ensemble_generation.mp4'
filepath = os.path.join(cwd, 'output', filename)
animation = mpy.VideoClip(drawframe, duration=duration)
animation.write_videofile(filepath, fps=24, codec='libx264')