import numpy as np
import astropy.units as u

import os
import h5py

import huxt as H
import huxt_analysis as HA


#a script to generate the streakline test case

v_boundary = np.ones(128) * 400 * (u.km/u.s)
v_boundary[30:50] = 600 * (u.km/u.s)
v_boundary[95:125] = 700 * (u.km/u.s)

#  Add a CME
cme = H.ConeCME(t_launch=0.5*u.day, longitude=0.0*u.deg, width=30*u.deg, v=1000*(u.km/u.s), thickness=5*u.solRad)
cme_list = [cme]

#  Setup HUXt to do a 5-day simulation, with model output every 4 timesteps (roughly half and hour time step)
model_test = H.HUXt(v_boundary=v_boundary, cr_num=2080, cr_lon_init=180*u.deg, simtime=5*u.day, dt_scale=4)


#trace a bunch of field lines from a range of evenly spaced Carrington longitudes
dlon = (20*u.deg).to(u.rad).value
lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad


#give the streakline footpoints (in Carr long) to the solve method
model_test.solve(cme_list, streak_carr = lon_grid)

#plot these streaklines
time = 4.5*u.day
fig, ax = HA.plot(model_test,time, save = 'True')

# Save the reference model output
dirs = H._setup_dirs_()
test_case_path = os.path.join(dirs['test_data'], 'HUXt_CR2080_streaklines_case.hdf5')


h5f = h5py.File(test_case_path, 'w')
h5f.create_dataset('vgrid', data = model_test.v_grid)
h5f.create_dataset('streak_particles_r', data = model_test.streak_particles_r)
h5f.close()   



#load in the test data
dirs = H._setup_dirs_()
test_case_path = os.path.join(dirs['test_data'], 'HUXt_CR2080_streaklines_case.hdf5')
h5f = h5py.File(test_case_path,'r')
vgrid = np.array(h5f['vgrid'])
streakline_particles_r = np.array(h5f['streak_particles_r'])
h5f.close()


#check the data agree - first check the vgrid si the same
assert np.allclose(model_test.v_grid.value, vgrid, atol=1e-3)

#only compare non-nan values of streakline positions
mask = np.isfinite(streakline_particles_r)
assert np.allclose(streakline_particles_r[mask], 
                   model_test.streak_particles_r[mask].value, atol=1e-3)
