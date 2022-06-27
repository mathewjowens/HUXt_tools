# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:11:46 2021

@author: mathewjowens
"""

# <codecell> Demo script - load data, generate ensemble, run HUXt

import os
import datetime
import huxt_ensembles as Hens



#==============================================================================
forecasttime = datetime.datetime(2022,2,24,22,0,0)

# set the directory of this file as the working directory
cwd = os.path.abspath(os.path.dirname(__file__))
#the filenames should then be generated from the forecast date
wsafilepath = os.path.join(cwd, 'data', '2022-02-24T22Z.wsa.gong.fits')
pfssfilepath = os.path.join(cwd, 'data','windbound_b_pfss20220224.22.nc') 
cortomfilepath = os.path.join(cwd, 'data','tomo_8.00_20220224_tomo.txt')
dumfricfile =  ""
#the cone file
conefilepath = os.path.join(cwd, 'data', 'cone2bc_modified.in')

#where to save the output images
savedir = os.path.join(cwd, 'output')

#==============================================================================


#run the ensemble master script
Hens.sweep_ensemble_run(forecasttime, savedir =savedir, 
                       wsafilepath = wsafilepath, pfssfilepath = pfssfilepath, 
                       cortomfilepath = cortomfilepath, dumfricfilepath = "",
                       conefilepath = conefilepath)



