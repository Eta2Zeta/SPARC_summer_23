"""
This script generates a large ensemble of markers uniformly
distributed in 3D space and randomly distributed in velocity
direction.  It runs the simulation them for only 1 microsecond.  
The marker generation is "ab initio", i.e.no lossmap is applied.

parent is template_step_1d:  large number of markers, short sim time
parent is 1261.  
"""
import os
import sys
import time

import numpy                       as np
import scipy.constants             as constants
import matplotlib.pyplot           as plt

import a5py.ascot5io.boozer        as boozer
import a5py.ascot5io.mhd           as mhd
import a5py.ascot5io.ascot5        as ascot5
import a5py.ascot5io.orbits        as orbits
import a5py.ascot5io.options       as options
import a5py.ascot5io.B_GS          as B_GS
import a5py.ascot5io.E_TC          as E_TC
import a5py.ascot5io.plasma_1D     as P_1D
import a5py.ascot5io.wall_2D       as W_2D
import a5py.ascot5io.wall_3D       as W_3D
import a5py.ascot5io.N0_3D         as N0_3D
import a5py.ascot5io.mrk_gc        as mrk
import mhd_wrapper                 as MW

import sds_helpers         as helpers
import sparc_processing    as sparc_proc
import marker_sets         as marker_sets
import check_results       as check_results
import options_sets        as options_sets

from a5py.ascotpy import Ascotpy

e       = constants.elementary_charge
m_e_AMU = constants.physical_constants["electron mass in u"][0]
m_e     = constants.physical_constants["electron mass"][0]
c       = constants.physical_constants["speed of light in vacuum"][0]

def init(fn_hdf5, fn_profiles, fn_bfield, fn_geqdsk, eq_index):
    """
    first attempt at collisional GCF simlulations

    fn_hdf5        filename for input file
    fn_profiles    filename for temperature and density profiles
    fn_bfield      filename for 3D bfield.
    fn_geqdsk      filename for equilibrium
    eq_index       time index for equilibrium
    """

    my_description = "V0_GO"
    settings={}
    settings["bfield_make_3D"] = False


    
    
    #  -------------------------------
    #  -  magnetic field             -
    #  -------------------------------

    #my_bboptions                              = {}
    #my_bboptions["z_offset_equilibrium"]      = 0.04
    #my_bboptions["rmajor_offset_equilibrium"] = 0.000
    
    if settings["bfield_make_3D"] == True:
        aa_bfield = sparc_proc.sparc_write_bfield_3d_hdf5_reverse(fn_bfield, fn_geqdsk, eq_index, fn_hdf5, desc=my_description, bmult=1.)
    else:
        BT     = 12.2
        Rmajor = 1.85

        psi0_mult = 1.05
        psi1_mult = 0.95
        
        aa_bfield = sparc_proc.sparc_write_bfield_2d_hdf5_reverse(BT, Rmajor, fn_geqdsk, eq_index, fn_hdf5, desc=my_description, psi0_mult=psi0_mult, psi1_mult=psi1_mult)

    #  -----------------------------------------------------------------
    #  -    Generate options for short collisionless  G0 simulation   -
    #  -----------------------------------------------------------------
    
    set = 17
    
    options_settings = {}
    
    options_settings["my_max_simtime"]           = 0.002     # 0.002
    options_settings["my_max_cputime"]           = 9000.
    
    options_settings["my_fixedstep_gyrodefined"] = 10
    
    options_settings["my_orbitwrite_npoint"]     = 10
    options_settings["my_orbitwrite_interval"]   = 2.e-9
    options_settings["my_no_orbitwrite"]         = 1
    options_settings["my_go_record_mode"]        = 1
    
    options_settings["my_no_rholim"]             = 1
    options_settings["my_wallhit"]               = 1

    options_settings["my_dist_min_ppa"]          = -1.e-19
    options_settings["my_dist_max_ppa"]          =  1.e-19
    options_settings["my_dist_nbin_ppa"]         =  100

    options_settings["my_dist_min_ppe"]          = 0.
    options_settings["my_dist_max_ppe"]          = 1.e-19
    options_settings["my_dist_nbin_ppe"]         = 50

    options_settings["my_dist_nbin_r"]           = 100
    options_settings["my_dist_nbin_z"]           = 100
    options_settings["my_dist_nbin_phi"]         =  1


    
    aa_options = options_sets.options_sets(fn_hdf5, set, options_settings, desc=my_description)


    #  
    
    marker_settings={}   #  from group_go_643
    
    marker_settings["birth_rhomax"]  = 1.00
    marker_settings["birth_rhomin"]  = 0.0     
    marker_settings["index"]         = 0        # index number of equilibrium in geqdsk file
    marker_settings["fn_profiles"]   = fn_profiles
    marker_settings["my_min_pitch"]  = -0.99
    marker_settings["my_max_pitch"]  =  0.99
    marker_settings["nplot_max"]     = 30000    # so plot files do not get too big
    marker_set  = 7
    
    # ++++++++++++++++++++++++++++++++++
    #  
    # Markers: 
    #
    Nmrk        =  10720            #    
    
    nrho_profiles = 101
    aa_markers = marker_sets.define_prt_markers_uniform(fn_hdf5, fn_geqdsk, marker_set, Nmrk, marker_settings, nrho_profiles) #desc=my_description)

    #  ----------------------------------------------------------
    #  -  plasma species, and temperature and density profiles  -
    #  ----------------------------------------------------------

    nrho_profiles   = 101
    rhomax_profiles = 1.2

    aa_profiles = sparc_proc.write_sparc_profiles(fn_profiles, nrho_profiles, rhomax_profiles, fn_hdf5, desc=my_description)

    #  ----------------------------
    #  -  3D wall shape           -
    #  ----------------------------

    wall_settings={}
    wall_settings["edge_thickness"] = 0.
    wall_settings["eq_index"] = 0
    
    aa_wall = sparc_proc.write_sparc_conformal_wall(fn_hdf5,fn_geqdsk, wall_settings, desc=my_description)
    
    #  ------------------------
    #  -  neutral density     -
    #  ------------------------

    N0_3D.write_hdf5_dummy(fn_hdf5, desc=my_description)   # neutral density specification

    #  --------------------------------------
    #  -  set electric field to zero        -
    #  --------------------------------------
    
    Exyz   = np.array([0, 0, 0])
    print("   ... I am about to write the electric field")
    E_TC.write_hdf5(fn_hdf5, Exyz, desc=my_description)
    print("   ... I have written the electric field")

    # ----------------------------------
    # -  dummy boozer and mhd data     -
    # ----------------------------------

    boozer.write_hdf5_dummy(fn_hdf5)

    mhd_settings = {}
    mhd_settings["mode_amplitude_multiplier"] = 1.e-3
    
    MW.mhd_wrapper(fn_mhd,fn_hdf5, mhd_settings)

    #mhd.write_hdf5_dummy(fn_hdf5)
#  ------------------------------------------------------------------------------------

if __name__ == '__main__':

    this_script =  __file__
    fn_hdf5     =  this_script.replace('.py','.h5') 
    if os.path.exists(fn_hdf5): raise ValueError(fn_hdf5+' already exists!')
    
    fn_profiles = 'v1e_profiles_3.txt'           #   possible_change 
    fn_bfield   = 'v1e_fixed_inout_case_a.txt'   #   IRRELEVANT:  bfield is 2D
    fn_wall_3d  = 'shape851_triangles.txt'       #   IRRELEVANT:  conformal wall shape used
    fn_geqdsk   = 'geqdsk_freegsu_run0_mod_00.geq'      #   possible_change
    fn_mhd      = 'mhd_data_for_ascot_n10_f482kHz.mat'   #   possible change
    
    #fn_geqdsk   = 'v1e_mod1.geq'      #   possible_change
    #fn_geqdsk    = 'v1e.geq'
    eq_index    = 0
    
    #fn_lossmap  = 'ascot_40560057.h5'  # not used
    #fn_weights  = 'ascot_41588768.h5'  # not used


    if( sys.argv[1] == "init" ):
        print("Initializing tests.")
        init(fn_hdf5, fn_profiles, fn_bfield, fn_geqdsk, eq_index)
        print("Initialization complete.")
        sys.exit()



