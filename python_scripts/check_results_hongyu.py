"""
plot results
"""
import get_rho_interpolator as get_rho
from readGeqdsk import geqdsk
import sys
import pdb
import numpy                   as np
import scipy.constants         as constants
import matplotlib.pyplot       as plt
import matplotlib              as mpl     # to allow large data sets

import a5py.ascot5io.ascot5    as ascot5
import a5py.ascot5io.orbits    as orbits
import a5py.ascot5io.options   as options
import a5py.ascot5io.B_GS      as B_GS
import a5py.ascot5io.E_TC      as E_TC
import a5py.ascot5io.plasma_1D as P_1D
import a5py.ascot5io.wall_2D   as W_2D
import a5py.ascot5io.N0_3D     as N0_3D
import a5py.ascot5io.mrk_gc    as mrk

import sds_helpers             as helpers
import sparc_processing        as sparc_proc
import marker_sets_hongyu             as marker_sets
import process_ascot           as proc_ascot
import h5py

from a5py.ascotpy import Ascotpy

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

from a5py.preprocessing.analyticequilibrium import psi0 as psifun
e       = constants.elementary_charge
m_e_AMU = constants.physical_constants["electron mass in u"][0]
m_e     = constants.physical_constants["electron mass"][0]
c       = constants.physical_constants["speed of light in vacuum"][0]

R0 = 1.65


def extract_runids(gg):
    
    return [key for key in gg.keys()]

def check_gca(fn_hdf5, simulation_name="ORBFOL_GCA"):
    """
    plot just the gca results
    """
    a5 = ascot5.Ascot(fn_hdf5)
    
    apy = Ascotpy(fn_hdf5)
    apy.init(bfield=True)

    raxis = R0            #a5["ORBFOL_GO"].bfield.read()["raxis"]

    ORBFOL = {}

    ORBFOL["GCA"] = {}
    orb = a5[simulation_name]["orbit"].read()
    print("check_gca:  have created the orb object")
    
    plot_checks_orbits(orb, apy, ORBFOL, fn_hdf5, dt_bin, fn_geqdsk, 'GCA')

def check_gcf(fn_hdf5, dt_bin, fn_geqdsk, fn_output="none", simulation_name="V0_GCF"):
    """
    plot just the gcf results.  simulation_name = e.g. ORBFOL_GCF
    """
    #import pdb; pdb.set_trace()
    print("")
    print("   ... here we are in check_results.check_gcf")
    a5 = ascot5.Ascot(fn_hdf5)
    apy = Ascotpy(fn_hdf5)
    #pdb.set_trace()

    apy.init(bfield=True)

    raxis = R0            #a5["ORBFOL_GO"].bfield.read()["raxis"]

    ORBFOL = {}

    ORBFOL["GCF"] = {}
           
    if (simulation_name == 'run_ID'):   # get from results file not h5 file

        aa               = proc_ascot.myread_hdf5(fn_output)
        
        ff_1             = h5py.File(fn_output)
        ff_2             = ff_1['results']
        runIDs           = extract_runids(ff_2)

        ff_2m            = ff_1['marker']
        runIDSm          = extract_runids(ff_2m)
        simname          = runIDSm[0]
        h5py_markers     = ff_2m[simname]
        
        simulation_name  = runIDs[0]
        ff_3             = ff_2[simulation_name]

        h5py_endstate    = ff_3['endstate']
        h5py_inistate    = ff_3['inistate']
        
        endcond          = np.array(h5py_endstate['endcond'])
    
        marker_r         = np.array(h5py_markers["r"])
        marker_z         = np.array(h5py_markers["z"])
        # marker_vpar      = np.array(h5py_markers["vpar"])  # vpar does not exist at marker
        marker_vphi      = np.array(h5py_markers["vphi"])
        marker_vr        = np.array(h5py_markers["vr"])
        marker_vz        = np.array(h5py_markers["vz"])
        
        marker_vtot      = np.sqrt( marker_vphi**2 + marker_vz**2 + marker_vr**2)
        marker_pitch     = marker_vphi / (marker_vtot + 1.e-11)   # approximate only
    
        h5py_orb         = ff_3['orbit']
        orb = {}

        orb["ini_r"]      = np.array(h5py_inistate["r"])
        orb["ini_z"]      = np.array(h5py_inistate["z"])
        orb["ini_vphi"]   = np.array(h5py_inistate["vphi"])
        orb["ini_vpar"]   = np.array(h5py_inistate["vpar"])
        orb["ini_vr"]     = np.array(h5py_inistate["vr"])
        orb["ini_vz"]     = np.array(h5py_inistate["vz"])

        orb["ini_pitch"]  = orb["ini_vpar"] / np.sqrt(orb["ini_vphi"]**2 + orb["ini_vr"]**2 + orb["ini_vz"]**2)
        
        orb["time"]   = np.array(h5py_orb["time"])
        orb["id"]     = np.array(h5py_orb["id"])
        orb["r"]      = np.array(h5py_orb["r"])
        orb["z"]      = np.array(h5py_orb["z"])
        orb["phi"]    = np.array(h5py_orb["phi"])
        orb["mu"]     = np.array(h5py_orb["mu"])
        orb["vpar"]   = np.array(h5py_orb["vpar"])
        orb["charge"] = np.array(h5py_orb["charge"])
        orb["br"]     = np.array(h5py_orb["br"])
        orb["bz"]     = np.array(h5py_orb["bz"])
        orb["bphi"]   = np.array(h5py_orb["bphi"])

        
        orb["endcond"]      = endcond
        orb["marker_r"]     = marker_r
        orb["marker_z"]     = marker_z
        orb["marker_pitch"] = marker_pitch

        print("check_gcf:  have created the orb object from results file")
        
    else:
        
        orb = a5[simulation_name]["orbit"].read()
        print("check_gcf:  have created the orb object from parent h5 file")

    # pdb.set_trace()

    print("   ... check_gcf:  about to call plot_checks_orbits")
    plot_checks_orbits(orb, apy, endcond, ORBFOL,fn_hdf5, 'GCF',aa, dt_bin, fn_geqdsk,orbthin=1)

def check_go(fn_hdf5):
    """
    plot just the go results
    """
    a5 = ascot5.Ascot(fn_hdf5)
    apy = Ascotpy(fn_hdf5)
    apy.init(bfield=True)

    raxis = R0            #a5["ORBFOL_GO"].bfield.read()["raxis"]

    ORBFOL = {}

    ORBFOL["GO"] = {}
    orb = a5["ORBFOL_GO"]["orbit"].read()
    print("check_go:  have created the orb object")
    my_orbthin = 1
    pdb.set_trace()
    plot_checks_orbits(orb,apy, ORBFOL, fn_hdf5, 'GO',orbthin=my_orbthin)

# =============================================================

def compute_r_z_wandering(time, rarray, zarray, dt_bin):

    tout = []
    rmax = []
    zmax = []
    rmin = []
    zmin = []

    if (time.size <= 10):

        out={}
            
        out["time"] = 0.
        out["zmin"] = 0.
        out["zmax"] = 0.
        out["rmin"] = 0.
        out["rmax"] = 0.

        return out
    
    for mm in range(100000000):   # infinite
        #pdb.set_trace()
        ii = (time <= (time[0]+dt_bin))   # time points within binning interval
        
        tout.append(np.mean(time[ii]))
              
        rmax.append(np.max(rarray[ii]))
        zmax.append(np.max(zarray[ii]))

        rmin.append(np.min(rarray[ii]))
        zmin.append(np.min(zarray[ii]))
        
        jj = (time>(time[0]+dt_bin))
   
        tt = time[jj]                  # shorten the time array
        rr = rarray[jj]
        zz = zarray[jj]

        #pdb.set_trace()
        if len(tt) <= 0:              # nothing left to do

            r_min = np.asarray(rmin)
            r_max = np.asarray(rmax)

            z_min = np.asarray(zmin)
            z_max = np.asarray(zmax)

            tmean = np.asarray(tout)

            out={}
            
            out["time"] = tmean
            out["zmin"] = z_min
            out["zmax"] = z_max
            out["rmin"] = r_min
            out["rmax"] = r_max
            
            return out
        else:
            time=tt
            rarray = rr
            zarray=zz
            
def drift_from_ends(time, rr, zz, rho, frac):

    drift_rmin = 0.
    drift_rmax = 0.
    drift_zmin = 0.
    drift_zmax = 0.
    drift_rhomax  = 0.
    drift_rhomin  = 0.
    delta_rhomin  = 0.
    delta_rhomax  = 0.

    npts = time.size
   
    if(time.size > 500):

        tstart    = time[0]
        tend      = time[-1]
       
        ii_offset = int(0.1*npts)

        rmin_start   =  np.min(rr[0:ii_offset])
        rmax_start   =  np.max(rr[0:ii_offset])
        zmin_start   =  np.min(zz[0:ii_offset])
        zmax_start   =  np.max(zz[0:ii_offset])
        rhomin_start = np.min(rho[0:ii_offset])
        rhomax_start = np.max(rho[0:ii_offset])       

        rmin_end   =  np.min(rr[-ii_offset:])
        rmax_end   =  np.max(rr[-ii_offset:])
        zmin_end   =  np.min(zz[-ii_offset:])
        zmax_end   =  np.max(zz[-ii_offset:])
        rhomin_end = np.min(rho[-ii_offset:])
        rhomax_end = np.max(rho[-ii_offset:]) 

        drift_rmin = 100.*(rmin_end - rmin_start) / (tend - tstart)
        drift_rmax = 100.*(rmax_end - rmax_start) / (tend - tstart)
        drift_zmin = 100.*(zmin_end - zmin_start) / (tend - tstart)
        drift_zmax = 100.*(zmax_end - zmax_start) / (tend - tstart)

        drift_rhomin = (rhomin_end - rhomin_start) / (tend - tstart)
        drift_rhomax = (rhomax_end - rhomax_start) / (tend - tstart)

        delta_rhomin = rhomin_end - rhomin_start
        delta_rhomax = rhomax_end - rhomax_start
        #print("   ... drift3:  rmin rmax zmin zmax: %8.2f %8.2f %8.2f %8.2f" % \
        #      (drift_rmin, drift_rmax, drift_zmin, drift_zmax))
    
    return drift_rmin, drift_rmax, drift_zmin, drift_zmax, drift_rhomin, drift_rhomax, delta_rhomin, delta_rhomax
 
def plot_checks_orbits(orb, apy, endcond, ORBFOL, fn_hdf5, SIM_NAME,bb, dt_bin,fn_geqdsk, orbthin=1):  

    #  get the equilibrium
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()

    eq_index = 0
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs
    
    psi_rmin       =  gg.equilibria[eq_index].rmin
    psi_rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    psi_nr         =  gg.equilibria[eq_index].nw
    psi_zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    psi_zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    psi_nz         =  gg.equilibria[eq_index].nh
        
    psiPolSqrt     =  gg.equilibria[eq_index].psiPolSqrt

    geq_rarray = np.linspace(psi_rmin, psi_rmax, psi_nr)
    geq_zarray = np.linspace(psi_zmin, psi_zmax, psi_nz)

    # transpose so that we are on a grid = [R,z]. define a function
        
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)   # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))

    eq_index = 0
    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    #
    #  end of equilibrium stuff
    # --------------------------------------------------------------------------
    
    string_list     = fn_hdf5.split('.')
    stub            = string_list[0]
    filename_orbits = stub + '_orbits.pdf'
    filename_summary= stub + '_summary.pdf'
    filename_drifts = stub + '_drifts.txt'

    ff = open(filename_drifts,"w")
    ff.write("\n")
    ff.write("    marker    drift   drift_2   drift_3  ini_r   ini_z   pitch    emin    emax    mumin    mumax    ctormin   ctormax  fate   dt  \n")
                       
    
    #  the following borrowed from process_ascot 6/1/2020
    do_rasterized   = False
    do_rasterized_2 = False
    
    mpl.rcParams['image.composite_image']=False
    mpl.rcParams['pdf.fonttype']=42
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['xtick.direction']='in'
    mpl.rcParams['ytick.direction']='in'
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['ytick.minor.size'] = 3.5
    mpl.rcParams.update({'font.size':12})
    padsize=1.5
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True

    size_factor = 4.5    # for nifty loss plot
    fontsize_2 = 12

    
    r_wall = bb['r_wall']
    z_wall = bb['z_wall']
    
    mpl.rcParams['agg.path.chunksize'] = 10000   # to allow big data sets
    
    B = np.sqrt(np.power(orb["br"],2) + np.power(orb["bphi"],2) +
                np.power(orb["bz"],2))

    # Note that mu is in eV / T

    psi_GCA_along_orbit = apy.evaluate(orb["r"], orb["phi"], orb["z"], orb["time"], "psi")

    gamma = np.sqrt( ( 1 + 2 * orb["mu"] * e * B / ( m_e * c * c ) ) /
                     ( 1 - orb["vpar"] * orb["vpar"] / ( c * c ) ) )

    ORBFOL[SIM_NAME]["time"] = orb["time"]

    print("   ... have populated ORBFOL with time")
    
    ORBFOL[SIM_NAME]["id"]   = orb["id"]
    ORBFOL[SIM_NAME]["r"]    = orb["r"]
    ORBFOL[SIM_NAME]["z"]    = orb["z"]
    ORBFOL[SIM_NAME]["ekin"] = (gamma - 1) * m_e * c * c
    ORBFOL[SIM_NAME]["mu"]   = orb["mu"] * e
    ORBFOL[SIM_NAME]["ctor"] = gamma * m_e * orb["r"] * orb["vpar"] + orb["charge"] * e * psi_GCA_along_orbit

    print("   ... plot_checks_orbits:  have populated ORBFOL")
    
    #***************************************#
    #*     make the plots                   #
    #*                                      #
    #***************************************#

    max_marker = np.max(orb["id"])

    print("   ... plot_checks_orbits:  max_marker = ", max_marker)
    
    id_array   = np.linspace(0, max_marker-1,max_marker).astype(int)

    # print("   ... plot_checks_orbits:  id_array = ", id_array)
    
    stub_name  = fn_hdf5.split('.')[0]   # for generating unique plot filenames

    nn_wall = 0   # count number of markers that hit wall
    nn_max  = 300

    # pdb.set_trace()

    threshold_drift = 0.0    # generate plots only for "bad" markers  typ. = 4.0

    bad_orbits_35 = [367,372,379,412,416,442,450,493,523,524,556,577,578,592,606,612,613,614,650,679,680,683,684,685,686,697,698, \
                  706,707,709,710,711,712,713,714,715,716,717,718,719,720]
    
    my_array = id_array
    nn_pts_total   = my_array.size
    #my_array = my_array[0:50]
    nn_pts = my_array.size
    print("  number of markers: ", nn_pts_total, "  number that I will process: ", nn_pts)

    marker_fate = np.zeros(nn_pts).astype('int')

    mean_delta_time = np.zeros(nn_pts)
    
    drift_emin = np.zeros(nn_pts)
    drift_emax = np.zeros(nn_pts)
    drift_mumin   = np.zeros(nn_pts)
    drift_mumax   = np.zeros(nn_pts)
    drift_ctormin = np.zeros(nn_pts)
    drift_ctormax = np.zeros(nn_pts)

    
    drift_rmax = np.zeros(nn_pts)
    drift_rmin = np.zeros(nn_pts)
    drift_zmax = np.zeros(nn_pts)
    drift_zmin = np.zeros(nn_pts)
    drift_max  = np.zeros(nn_pts)

    rmax_drift_2 =  np.zeros(nn_pts)
    rmin_drift_2 =  np.zeros(nn_pts)
    rmax_drift_2 =  np.zeros(nn_pts)
    rmax_drift_2 =  np.zeros(nn_pts)

    drift_rmin2 = np.zeros(nn_pts)
    drift_rmax2 = np.zeros(nn_pts)
    drift_zmin2 = np.zeros(nn_pts)
    drift_zmax2 = np.zeros(nn_pts)
    drift_max2  =  np.zeros(nn_pts)

    rmin_drift3   = np.zeros(nn_pts)
    rmax_drift3   = np.zeros(nn_pts)
    zmin_drift3   = np.zeros(nn_pts)
    zmax_drift3   = np.zeros(nn_pts)
    drift3_array  = np.zeros(nn_pts)

    drift_rho    = np.zeros(nn_pts)
    rhomin_drift = np.zeros(nn_pts)
    rhomax_drift = np.zeros(nn_pts)
    delta_rho    = np.zeros(nn_pts)
    
    endcond = orb["endcond"]

    ii_loss = (endcond == 8) ^ (endcond == 32) ^ (endcond==520) ^ (endcond==544)
    ii_ok   = (endcond != 8) & (endcond != 32) & (endcond!=520) & (endcond!=544)
    
    id_lost_markers = id_array[ii_loss]
    id_ok_markers   = id_array[ii_ok]

    for marker in my_array:

        marker_lost = 0
        if marker in id_lost_markers:
                marker_lost = 1
        marker_fate[marker] = marker_lost
        #print("  %4d   %4d "%(marker, marker_lost))
    
    
    with PdfPages(filename_orbits) as pdf:

        for marker in my_array:    # id_array  bad_orbits my_array
            #print("  starting marker ", marker)
            # pdb.set_trace()
            id_gca = (ORBFOL[SIM_NAME]["id"] == marker + 1)

            my_dummy = 10
            if (my_dummy == 10):
            #if ((endcond[marker] == 8) or (endcond[marker] == 32)) :   # plot only markers that hit the wall  == 8  rhomax=32

                # print("   ... marker number ", marker+1, " escaped the plasma so I will plot its orbit")

                # -------------------------------------------------------------------
                # compute binned time, r, and z so we can easily look for orbit drift

                #print("   ... about to get my_time, my_r, my_z for marker ", marker)
                my_time = ORBFOL[SIM_NAME]["time"][id_gca]

                my_r    =    ORBFOL[SIM_NAME]["r"][id_gca]
                my_z    =    ORBFOL[SIM_NAME]["z"][id_gca]
                my_vpar =    orb["vpar"][id_gca]

                my_time = my_time[0:-1]    # last point confuses yrange of the plots
                my_r    =    my_r[0:-1]
                my_z    =    my_z[0:-1]
                my_vpar = my_vpar[0:-1]

                ii_sort = np.argsort(my_time)

                my_time = my_time[ii_sort]
                my_r    =    my_r[ii_sort]
                my_z    =    my_z[ii_sort]
                my_vpar = my_vpar[ii_sort]

                my_ntimes = my_time.size
                my_rho    = np.zeros(my_ntimes)
                for ijk in range(my_ntimes):
                    my_rho[ijk] = rho_interpolator(my_r[ijk], my_z[ijk])

                drift3_rmin, drift3_rmax, drift3_zmin, drift3_zmax, drift_rhomin, drift_rhomax, delta_rhomin, delta_rhomax \
                    = drift_from_ends(my_time, my_r, my_z, my_rho, 0.1)

                delta_rho[marker] =  np.max(np.array([np.abs(delta_rhomin), delta_rhomax]))
                drift_rho[marker]    = np.max(np.array([np.abs(drift_rhomin), drift_rhomax]))
                rhomin_drift[marker] = drift_rhomin
                rhomax_drift[marker] = drift_rhomax
                
                drift3_list = [np.abs(drift3_rmin), np.abs(drift3_rmax), np.abs(drift3_zmin), np.abs(drift3_zmax)]
                local_array = np.array(drift3_list)
                drift3 = np.max(local_array)

                rmin_drift3[marker]  = drift3_rmin
                rmax_drift3[marker]  = drift3_rmax
                zmin_drift3[marker]  = drift3_zmin
                zmax_drift3[marker]  = drift3_zmax
                #pdb.set_trace()
                drift3_array[marker] = drift3

                if (my_time.size > 0):
                    mean_delta_time[marker] = (my_time[-1] - my_time[0]) / my_time.size
                    aa                      = compute_r_z_wandering(my_time, my_r, my_z, dt_bin)
                    time_check              = aa["time"]     # in case we have just one point
                else:
                    time_check = 0.
                # print("   ... marker, type(time_check):  ", marker, type(time_check))
                
                if (type(time_check) != float):
                    
                    #print("   ... we are at position m for marker ", marker)
                    
                    time_binned  = aa["time"][0:-1]    # last point confuses yrange
                    zmax_binned  = aa["zmax"][0:-1]
                    zmin_binned  = aa["zmin"][0:-1]
                    rmax_binned  = aa["rmax"][0:-1]
                    rmin_binned  = aa["rmin"][0:-1]
                    
                    #print("   ... we are at position n for markerk time_binned.size ", marker, time_binned.size)
                    
                    rmax_drift  = 0.
                    rmin_drift  = 0.
                    zmax_drift  = 0.
                    zmin_drift  = 0.
                    rmax_drift2 = 0.
                    rmin_drift2 = 0.
                    zmax_drift2 = 0.
                    zmin_drift2 = 0.
                    print("   ... marker, time_binned.size:  ", marker, time_binned.size)
                    if(time_binned.size > 20):
                        
                        #print("   ... we are at position q for marker ", marker)
                        delta_t  = time_binned[-1] - time_binned[0]

                        #print("   ... marker, delta_t: ", marker, delta_t)
                        
                        if (delta_t > 1.e-5):
                            
                            #print("   ... we are at position q2 for marker ", marker)
                            
                            rmax_drift = 100.*(rmax_binned[-1] - rmax_binned[0] ) / delta_t
                            rmin_drift = 100.*(rmin_binned[-1] - rmin_binned[0] ) / delta_t
                            zmax_drift = 100.*(zmax_binned[-1] - zmax_binned[0] ) / delta_t
                            zmin_drift = 100.*(zmin_binned[-1] - zmin_binned[0] ) / delta_t

                            ntimes   = time_binned.size
                            ii_start = int(ntimes/10)
                            ii_end   = - ii_start
                            #pdb.set_trace()
                            #if(marker == 31):
                            #   pdb.set_trace()
                            rmax_binned_start = np.max(rmax_binned[0:ii_start])
                            rmax_binned_end   = np.max(rmax_binned[ii_end:])
                            rmin_binned_start = np.min(rmin_binned[0:ii_start])
                            rmin_binned_end   = np.min(rmin_binned[ii_end:])

                            zmax_binned_start = np.max(zmax_binned[0:ii_start])
                            zmax_binned_end   = np.max(zmax_binned[ii_end:])
                            zmin_binned_start = np.min(zmin_binned[0:ii_start])
                            zmin_binned_end   = np.min(zmin_binned[ii_end:])
                            
                            rmax_drift2 = 100.*(rmax_binned_end - rmax_binned_start ) / delta_t
                            rmin_drift2 = 100.*(rmin_binned_end - rmin_binned_start ) / delta_t
                            zmax_drift2 = 100.*(zmax_binned_end - zmax_binned_start ) / delta_t
                            zmin_drift2 = 100.*(zmin_binned_end - zmin_binned_start ) / delta_t
                            
                            drift_rmin[marker] = rmin_drift
                            drift_rmax[marker] = rmax_drift
                            drift_zmin[marker] = zmin_drift
                            drift_zmax[marker] = zmax_drift

                            drift_rmin2[marker] = rmin_drift2
                            drift_rmax2[marker] = rmax_drift2
                            drift_zmin2[marker] = zmin_drift2
                            drift_zmax2[marker] = zmax_drift2
                            
                            #print("   ... we are at position r for marker ", marker)
                    #print("   ... we are at position r2 for marker ", marker)     

                           
                    drift_list        = [np.abs(rmin_drift),  np.abs(rmax_drift),  np.abs(zmin_drift),  np.abs(zmax_drift)]
                    drift_list2       = [np.abs(rmin_drift2), np.abs(rmax_drift2), np.abs(zmin_drift2), np.abs(zmax_drift2)]  
                    drift_array       = np.array(drift_list)
                    drift_array2      = np.array(drift_list2)
                    max_drift         = np.max(drift_array)
                    max_drift2        = np.max(drift_array2)
                
                    drift_max[marker] = max_drift
                    drift_max2[marker] = max_drift2

                    #print("  position-1")
                    #print("  position-2")
                    if(max_drift < threshold_drift):

                        print("   ... max_drift = %7.2f so I will skip plots" % (max_drift))

                    else:

                        #print("   ... about to start the plots for marker ", marker)

                        plt.figure()

                        vertical_padsize = 0.
                        graph_name = 'Rmax  marker ' + str(marker) + '   drifts: '+ str(" %9.3f %9.3f " %(rmax_drift, rmax_drift2)) + " cm/s"
                        ax=plt.subplot(511)
                        plt.plot(time_binned, rmax_binned, rasterized=do_rasterized)
                        plt.title(graph_name, fontsize=9, pad=2)
                        plt.tight_layout(h_pad=vertical_padsize)
                        ax.set_xticklabels([])

                        graph_name = 'Rmin  marker ' + str(marker) + '   drifts: ' + str(" %9.3f %9.3f" %(rmin_drift,rmin_drift2)) + " cm/s"
                        ax=plt.subplot(512)
                        plt.plot(time_binned, rmin_binned, rasterized=do_rasterized)
                        plt.title(graph_name, fontsize=9, pad=2)
                        plt.tight_layout(h_pad=vertical_padsize)
                        ax.set_xticklabels([])

                        graph_name = 'Zmax  marker ' + str(marker) + '   drifts: ' + str(" %9.2f %9.2f" %(zmax_drift,zmax_drift2)) + " cm/s"
                        ax=plt.subplot(513)
                        plt.plot(time_binned, zmax_binned, rasterized=do_rasterized)
                        plt.title(graph_name, fontsize=9, pad=2)
                        plt.tight_layout(h_pad=vertical_padsize)
                        ax.set_xticklabels([])

                        graph_name = 'Zmin  marker' + str(marker) + '   drifts: ' + str(" %9.2f %9.2f" %(zmin_drift, zmin_drift2)) + " cm/s"
                        ax=plt.subplot(514)
                        plt.plot(time_binned, zmin_binned, rasterized=do_rasterized)
                        plt.title(graph_name, fontsize=9,pad=2)
                        
                        
                        plt.tight_layout(h_pad=vertical_padsize)
                        ax.set_xticklabels([])

                        graph_name = str('rho  marker%4d  drift: %9.4f %9.4f' % (marker, rhomin_drift[marker], rhomax_drift[marker]))
                        ax=plt.subplot(515)
                        plt.plot(my_time,my_rho, rasterized=do_rasterized)
                        plt.title(graph_name, fontsize=9,pad=2)
                        plt.xlabel('time [sec]', fontsize=10)
                        plt.tight_layout(h_pad=vertical_padsize)
                        #plt.show()
                        pdf.savefig()

                        #if (max_drift >= 5.):
                        #    plotfile_name = stub_name + '_' + SIM_NAME + '_marker_' + str(marker) + '_drift_bad.pdf'
                        #    plt.savefig(plotfile_name, dpi=300)

                        plt.close()

                        nn_wall = nn_wall + 1

                        #if(nn_wall > nn_max):
                        #    return

                        f = plt.figure(figsize=(11.9/2.54, 8/2.54))

                        plt.rc('xtick', labelsize=10)
                        plt.rc('ytick', labelsize=10)
                        plt.rc('axes',  labelsize=10)

                        plt.rcParams['mathtext.fontset'] = 'stix'
                        plt.rcParams['font.family'] = 'STIXGeneral'

                        h1 = f.add_subplot(1,4,1)
                        h1.set_position([0.16, 0.72, 0.44, 0.25], which='both')

                        h2 = f.add_subplot(1,4,2)
                        h2.set_position([0.16, 0.44, 0.44, 0.25], which='both')

                        h3 = f.add_subplot(1,4,3)
                        h3.set_position([0.16, 0.155, 0.44, 0.25], which='both')

                        h4 = f.add_subplot(1,4,4)
                        h4.set_position([0.52, 0.15, 0.62, 0.78], which='both')

                        colors = ["b", "r", "forestgreen", "cyan", "magenta", "dodgerblue", "r", "tomato"]

                        #plt.show()

                        #***********************************************************#
                        #*   Finalize and print and show the figure                 #
                        #***********************************************************#

                        time_array = ORBFOL[SIM_NAME]["time"][id_gca]
                        ekin_array = ORBFOL[SIM_NAME]["ekin"][id_gca]
                        ctor_array = ORBFOL[SIM_NAME]["ctor"][id_gca]
                        mu_array   = ORBFOL[SIM_NAME]["mu"][id_gca]
                        z_array    = ORBFOL[SIM_NAME]["z"][id_gca]
                        r_array    = ORBFOL[SIM_NAME]["r"][id_gca]

                        jj_sort = np.argsort(time_array)

                        time_array = time_array[jj_sort]    # per Libby 11/19/19
                        ekin_array = ekin_array[jj_sort]
                        ctor_array = ctor_array[jj_sort]
                        mu_array   =   mu_array[jj_sort]
                        r_array    =    r_array[jj_sort]
                        z_array    =    z_array[jj_sort]

                        ntimes = time_array.size
                        nbins  = 10
                        indices_start = int((ntimes/nbins))
                        indices_end   = -1 * indices_start
                        
                        emin_start = np.min(ekin_array[0:indices_start])
                        emax_start = np.max(ekin_array[0:indices_start])
                        emin_end   = np.min(ekin_array[indices_end:])
                        emax_end   = np.max(ekin_array[indices_end:])

                        mu_start_min = np.min(mu_array[0:indices_start])
                        mu_start_max = np.max(mu_array[0:indices_start])
                        mu_end_min   = np.min(mu_array[indices_end:])
                        mu_end_max   = np.max(mu_array[indices_end:])
                        
                        ctor_start_min = np.min(ctor_array[0:indices_start])
                        ctor_start_max = np.max(ctor_array[0:indices_start])
                        ctor_end_min   = np.min(ctor_array[indices_end:])
                        ctor_end_max   = np.max(ctor_array[indices_end:])

                        drift_emin[marker]    = np.abs( (emin_start - emin_end)/emin_start)
                        drift_emax[marker]    = np.abs( (emax_start - emax_end)/emax_start)
                        drift_mumin[marker]   = np.abs( (  mu_start_min -   mu_end_min)/  mu_start_min)
                        drift_mumax[marker]   = np.abs( (  mu_start_max -   mu_end_max)/  mu_start_max)
                        drift_ctormin[marker] = np.abs( (ctor_start_min - ctor_end_min)/ctor_start_min)
                        drift_ctormax[marker] = np.abs( (ctor_start_max - ctor_end_max)/ctor_start_max)
                        
                        print( " %d  lost: %2d  drifts: %8.3f %11.4f %8.3f %9.5f %8.5f %8.3f %5.3f %6.3f %6.3f  %8.3f %8.3f %8.3f   %8.2e"  %     \
                               (marker, marker_fate[marker], max_drift, max_drift2, drift3_array[marker],  drift_rho[marker], delta_rho[marker], \
                               orb["ini_r"][marker], orb["ini_z"][marker], orb["ini_pitch"][marker], rmax_drift, rmin_drift,  \
                                zmax_drift, zmin_drift, mean_delta_time[marker]))
                        #if(marker >= 355):
                        #    pdb.set_trace()
                        
                        ff.write("  %5d   %9.3f  %9.4f  %9.4f  %9.5f %8.5f %5.3f %6.3f %6.3f %10.2e %10.2e  %10.2e  %10.2e  %10.2e  %10.2e  %4d %8.2e  \n" %             \
                                 (marker,                   \
                                  drift_max[marker],        \
                                  drift_max2[marker],       \
                                  drift3_array[marker],     \
                                  drift_rho[marker],        \
                                  delta_rho[marker],        \
                                  orb["ini_r"][marker],     \
                                  orb["ini_z"][marker],     \
                                  orb["ini_pitch"][marker], \
                                  drift_emin[marker],       \
                                  drift_emax[marker],       \
                                  drift_mumin[marker],      \
                                  drift_mumax[marker],      \
                                  drift_ctormin[marker],    \
                                  drift_ctormax[marker],    \
                                  marker_fate[marker],      \
                                  mean_delta_time[marker]))
                        #if(marker >=9):
                        #    pdb.set_trace()
                        if (orbthin > 1):
                            time_array = time_array[::orbthin]
                            ekin_array = ekin_array[::orbthin]
                            ctor_array = ctor_array[::orbthin]
                            mu_array   =   mu_array[::orbthin]
                            z_array    =    z_array[::orbthin]
                            r_array    =    r_array[::orbthin]

                        # pdb.set_trace()
                        #print("   ... position z")
                        #pdb.set_trace()
                        plot_relerr(h1, time_array, ekin_array,  colors[2])
                        
                        #graph_name = str('rho  marker%4d  drift: %9.4f %9.4f' % (marker, rhomin_drift[marker], rhomax_drift[marker]))
                        
                        graph_name = str(' %9.4f %9.4f %9.5f' % (rhomin_drift[marker], rhomax_drift[marker], \
                                                                     delta_rho[marker]))
                        h2.plot(my_time,my_rho, colors[2], rasterized=do_rasterized)
    
                        plot_relerr(h3, time_array, ctor_array,  colors[2])
                        #h4.plot(        r_array,   z_array, colors[2])

                        # the following is OK if there are more than 100 points recorded.  but if only 59 points, then we have a problem
                        #h4.plot(ORBFOL[SIM_NAME]["r"][id_gca][0:100], ORBFOL[SIM_NAME]["z"][id_gca][0:100], 'bo', ms=0.05,rasterized=True,fillstyle='full')
                        #h4.plot(r_array,     z_array, 'r-', ms=0.015, rasterized=True)   # 'ro'

                        h4.plot(r_wall, z_wall, 'b', linewidth=1)
                        h4.plot(r_array,     z_array, 'ro', ms=0.1, fillstyle='full', rasterized=True)   # was 'ro'
                        h4.plot(r_array[0], z_array[0], 'go', ms=4, fillstyle='none')
                        h4.plot(r_array[-1], z_array[-1], 'ko', ms=4, fillstyle='full')       

                        # pdb.set_trace()
                        tgca_min = np.min(ORBFOL[SIM_NAME]["time"][id_gca])
                        tgca_max = np.max(ORBFOL[SIM_NAME]["time"][id_gca])

                        # print(SIM_NAME, ":  time min, max: ", tgca_min, tgca_max)

                        time_max   = tgca_max

                        microsec_ticks = int(1.e6*time_max)+1

                        my_ticks = np.linspace(0,microsec_ticks, microsec_ticks+1)*1.e-6
                        float_ticklabels = np.linspace(0,microsec_ticks, microsec_ticks+1)
                        my_ticklabels = [ int(i) for i in float_ticklabels]

                        h1.set_xlim(0, time_max)
                        h1.xaxis.set(ticklabels=[])
                        #h1.xaxis.set()
                        # h1.yaxis.set(ticks=np.array([-20, -10, 0, 10, 20])*1e-10, ticklabels=[-20, '', 0, '', 20])
                        h1.tick_params(axis='y', direction='in')
                        h1.tick_params(axis='x', direction='in')
                        h1.spines['right'].set_visible(False)
                        h1.spines['top'].set_visible(False)
                        h1.yaxis.set_ticks_position('left')
                        h1.xaxis.set_ticks_position('bottom')
                        h1.set(ylabel=r"$\Delta E/E_0\;[10^{-11}]$")

                        h2.set_xlim(0, time_max)
                        h2.xaxis.set(ticklabels=[])
                        #h2.yaxis.set(ticks=np.array([-10, 0, 10])*1e-3, ticklabels=[-10, 0, 10])
                        h2.tick_params(axis='y', direction='in')
                        h2.tick_params(axis='x', direction='in')
                        h2.spines['right'].set_visible(False)
                        h2.spines['top'].set_visible(False)
                        h2.yaxis.set_ticks_position('left')
                        h2.xaxis.set_ticks_position('bottom')
                        xpos = time_max / 2.
                        ypos = np.min(my_rho) + 0.85 * (np.max(my_rho)-np.min(my_rho))
                        h2.text(xpos, ypos, graph_name, horizontalalignment='center', verticalalignment='center', fontsize=10)
                        h2.set(ylabel="rho")

                        h3.set_xlim(0, time_max)
                        #h3.xaxis.set(ticklabels=my_ticklabels)
                        #h3.yaxis.set(ticks=np.array([-6, 0, 6])*1e-4, ticklabels=[-6, 0, 6])
                        h3.tick_params(axis='y', direction='in')
                        h3.tick_params(axis='x', direction='in')
                        h3.spines['right'].set_visible(False)
                        h3.spines['top'].set_visible(False)
                        h3.yaxis.set_ticks_position('left')
                        h3.xaxis.set_ticks_position('bottom')
                        h3.set(ylabel=r"$\Delta P/P_0\;[10^{-4}]$", xlabel=r"Time [s]")

                        h4.axis('scaled')
                        h4.set_xlim(1.,2.4)
                        h4.set_ylim(-1.2,1.2)
                        h4.yaxis.set(ticks=[-0.8,  -0.4,  0., 0.4,  0.8 ])
                        h4.xaxis.set(ticks=[1.2,  1.6,  2.0])
                        h4.tick_params(axis='y', direction='in')
                        h4.tick_params(axis='x', direction='in')
                        h4.set(xlabel="$R$ [m]", ylabel="")
                        h4.grid(True)
                        #legend = [r"GCA"]

                        #h4.text(1.2, 0.95, legend[0], fontsize=9, color=colors[0])
                        #h4.text(1.5, 0.95, legend[1], fontsize=9, color=colors[1])
                        #h4.text(1.8, 0.95, legend[2], fontsize=9, color=colors[2])

                        plot_title = 'marker: ' + str(marker)
                        plt.title(" %d  RZp %6.3f %7.3f %7.3f" % (marker, orb["ini_r"][marker], \
                                                                   orb["ini_z"][marker], orb["ini_pitch"][marker]), fontsize=9)

                        plotfile_name = stub_name + '_' + SIM_NAME + '_marker_' + str(marker) + '_orbit.pdf'
                        #plt.show()
                        pdf.savefig()
                        #print("check_gca:  about to call plt.show()")
                        #plt.show()
                        plt.close()

                        #  ----------------------------------------------------
                        #
                        #    plots of r and z versus time

                        plt.rc('xtick', labelsize=14)
                        plt.rc('ytick', labelsize=14)
                        plt.rc('axes',  labelsize=14)

                        f = plt.figure(figsize=(11.9/2.54, 8/2.54))

                        plt.rcParams['mathtext.fontset'] = 'stix'
                        plt.rcParams['font.family'] = 'STIXGeneral'

                        graph_title = "R max/min,  Z max/min " + stub_name  +  '_marker_' + str(marker)

                        h1 = f.add_subplot(4,1,1,title=graph_title)
                        h2 = f.add_subplot(4,1,2)
                        h3 = f.add_subplot(4,1,3)
                        h4 = f.add_subplot(4,1,4)

                        h1.set_title(graph_title, fontsize=10 )

                        time_min = time_array[0]
                        time_max = time_array[-1]

                        delta_show = 0.06    # meters

                        rmax_hi  = np.max(r_array)+ 0.001
                        rmax_lo  = rmax_hi        - delta_show

                        rmin_lo  = np.min(r_array) - 0.001
                        rmin_hi  = rmin_lo        + delta_show

                        zmax_hi  = np.max(z_array)+ 0.001
                        zmax_lo  = zmax_hi - delta_show

                        zmin_lo  = np.min(z_array) - 0.001
                        zmin_hi  = zmin_lo + delta_show

                        rmax_hi = 100. * rmax_hi
                        rmax_lo = 100. * rmax_lo
                        rmin_hi = 100. * rmin_hi
                        rmin_lo = 100. * rmin_lo

                        zmax_hi = 100. * zmax_hi
                        zmax_lo = 100. * zmax_lo
                        zmin_hi = 100. * zmin_hi
                        zmin_lo = 100. * zmin_lo

                        # pdb.set_trace()

                        h1.set_xlim(time_min, time_max)
                        h2.set_xlim(time_min, time_max)
                        h3.set_xlim(time_min, time_max)
                        h4.set_xlim(time_min, time_max)

                        h1.set_ylim(rmax_lo, rmax_hi)
                        h2.set_ylim(rmin_lo, rmin_hi)
                        h3.set_ylim(zmax_lo, zmax_hi)
                        h4.set_ylim(zmin_lo, zmin_hi)

                        h4.set(ylabel="", xlabel="")

                        h1.tick_params(axis='y', direction='in', width=0.5, labelsize=5)
                        h2.tick_params(axis='y', direction='in', width=0.5, labelsize=5)
                        h3.tick_params(axis='y', direction='in', width=0.5, labelsize=5)
                        h4.tick_params(axis='y', direction='in', width=0.5, labelsize=5)

                        h1.tick_params(axis='x', direction='in', width=0.5, labelsize=5)
                        h2.tick_params(axis='x', direction='in', width=0.5, labelsize=5)
                        h3.tick_params(axis='x', direction='in', width=0.5, labelsize=5)
                        h4.tick_params(axis='x', direction='in', width=0.5, labelsize=5)

                        #h1.spines['right'].set_visible(False)
                        #h2.spines['right'].set_visible(False)
                        #h3.spines['right'].set_visible(False)
                        #h4.spines['right'].set_visible(False)

                        #h1.spines['top'].set_visible(False)
                        #h2.spines['top'].set_visible(False)
                        #3.spines['top'].set_visible(False)
                        #4.spines['top'].set_visible(False)

                        h1.yaxis.set_ticks_position('left')
                        h2.yaxis.set_ticks_position('left')
                        h3.yaxis.set_ticks_position('left')
                        h4.yaxis.set_ticks_position('left')

                        h1.xaxis.set_ticks_position('bottom')
                        h2.xaxis.set_ticks_position('bottom')
                        h3.xaxis.set_ticks_position('bottom')
                        h4.xaxis.set_ticks_position('bottom')

                        h1.set_xticklabels([])
                        h2.set_xticklabels([])
                        h3.set_xticklabels([])

                        t_ref = np.array([time_array[0], time_array[-1]])

                        y1_ref = np.array([100.*np.max(r_array), 100.*np.max(r_array)])
                        y2_ref = np.array([100.*np.min(r_array), 100.*np.min(r_array)])
                        y3_ref = np.array([100.*np.max(z_array), 100.*np.max(z_array)])
                        y4_ref = np.array([100.*np.min(z_array), 100.*np.min(z_array)])

                        h1.plot(t_ref, y1_ref, 'r-', linewidth=1.5)
                        h2.plot(t_ref, y2_ref, 'r-', linewidth=1.5)
                        h3.plot(t_ref, y3_ref, 'r-', linewidth=1.5)
                        h4.plot(t_ref, y4_ref, 'r-', linewidth=1.5)

                        #h1.plot(time_array, 100. * r_array, 'b-', linewidth=0.25, rasterized=True)
                        #h2.plot(time_array, 100. * r_array, 'b-', linewidth=0.25, rasterized=True)
                        #h3.plot(time_array, 100. * z_array, 'b-', linewidth=0.25, rasterized=True)
                        #h4.plot(time_array, 100. * z_array, 'b-', linewidth=0.25, rasterized=True)

                        # print(marker, rmax_hi, rmax_lo, zmax_hi, zmax_lo)
                        plotfile_name = stub_name + '_' + SIM_NAME + '_marker_' + str(marker) + '_RZ_time.pdf'
                        #plt.show()
                        #pdb.set_trace()
                        pdf.savefig()
                        plt.close()
                        
                        #print("   ... I have completed the orbit plots for marker", marker)
    print("   ... about to compute statistics")
    #pdb.set_trace()
    mm = drift_max.size
    jj_ok = ii_ok[0:mm]
    yy = drift_max[jj_ok]
    nn_1 =  yy[ (yy <= 0.5)].size
    nn_2 =  yy[ (yy >=0.5) & (yy <=1.5)].size
    nn_3 =  yy[ (yy >=1.5) & (yy <=2.5)].size
    nn_4 =  yy[ (yy >=2.5) & (yy <=4.0)].size
    nn_5 =  yy[ (yy >=4.0) & (yy <=10.)].size
    nn_6 =  yy[ (yy >=10.) & (yy <=20.)].size
    nn_7 =  yy[ (yy >=20.) & (yy <=40.)].size
    nn_8 =  yy[ (yy >=40.) & (yy <=100)].size
    nn_9 =  yy[ (yy >=100.)& (yy <=200.)].size
    nn_10 = yy[ (yy >=200.)& (yy <=300.)].size
    nn_11 = yy[ (yy >=300.)& (yy <=400.)].size
    nn_12 = yy[ (yy >=400) ].size
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("  0    < drift <   0.5  %4d \n" % (nn_1))
    ff.write(" 0.5   < drift <   1.5  %4d \n" % (nn_2))
    ff.write(" 1.5   < drift <   2.5  %4d \n" % (nn_3))
    ff.write(" 2.5   < drift <     4  %4d \n" % (nn_4))
    ff.write("   4   < drift <    10  %4d  \n" % (nn_5))
    ff.write("  10   < drift <    20  %4d  \n" % (nn_6))
    ff.write("  20   < drift <    40  %4d  \n" % (nn_7))
    ff.write("  40   < drift <   100  %4d  \n" % (nn_8))
    ff.write(" 100   < drift <   200  %4d  \n" % (nn_9))
    ff.write(" 200   < drift <   300  %4d  \n" % (nn_10))
    ff.write(" 300   < drift <   400  %4d  \n" % (nn_11))
    ff.write(" 400   < drift          %4d  \n" % (nn_12))
    ff.write("\n")
    ff.write(" maximum drift = %8.2f \n "%(nn_max))
    ff.write("\n")
    
    yy = drift_max2[jj_ok]
    nn_1 =  yy[ (yy <= 0.01)].size
    nn_2 =  yy[ (yy >=0.01) & (yy <=0.05)].size
    nn_3 =  yy[ (yy >=0.05) & (yy <=0.10)].size
    nn_4 =  yy[ (yy >=0.1) & (yy <=0.5)].size
    nn_5 =  yy[ (yy >=0.5) & (yy <=2.)].size
    nn_6 =  yy[ (yy >=2.) & (yy <=3.)].size
    nn_7 =  yy[ (yy >=3.) & (yy <=5.)].size
    nn_8 =  yy[ (yy >=5.) & (yy <=10)].size
    nn_9 =  yy[ (yy >=10.)& (yy <=30.)].size
    nn_10 = yy[ (yy >=30.)& (yy <=100.)].size
    nn_11 = yy[ (yy >=100.)& (yy <=200.)].size
    nn_12 = yy[ (yy >=200) ].size
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write(" 0.00  < drift2 <   0.01  %4d  \n" % (nn_1))
    ff.write(" 0.01  < drift2 <   0.05  %4d  \n" % (nn_2))
    ff.write(" 0.05  < drift2 <   0.1   %4d  \n" % (nn_3))
    ff.write(" 0.1   < drift2 <   0.5   %4d  \n" % (nn_4))
    ff.write(" 0.5   < drift2 <     2   %4d   \n" % (nn_5))
    ff.write("   2   < drift2 <     3   %4d   \n" % (nn_6))
    ff.write("   3   < drift2 <     5   %4d   \n" % (nn_7))
    ff.write("   5   < drift2 <    10   %4d   \n" % (nn_8))
    ff.write("  10   < drift2 <    30   %4d   \n" % (nn_9))
    ff.write("  30   < drift2 <   100   %4d   \n" % (nn_10))
    ff.write(" 100   < drift2 <   200   %4d   \n" % (nn_11))
    ff.write(" 200   < drift2           %4d   \n" % (nn_12))
    ff.write("\n")
    ff.write(" maximum drift2 = %8.2f \n"%(nn_max))
    ff.write("\n")

    yy = drift_max2[jj_ok]

    hh_total = yy.size
    nn_1 =  100.*(yy[ (yy >=200.)].size)/hh_total
    nn_2 =  100.*(yy[ (yy >=100.)].size)/hh_total
    nn_3 =  100.*(yy[ (yy >=50.) ].size)/hh_total
    nn_4 =  100.*(yy[ (yy >=25.) ].size)/hh_total
    nn_5 =  100.*(yy[ (yy >=15.) ].size)/hh_total
    nn_6 =  100.*(yy[ (yy >=10.) ].size)/hh_total
    nn_7 =  100.*(yy[ (yy >=5.)  ].size)/hh_total
    nn_8 =  100.*(yy[ (yy >=2.5) ].size)/hh_total
    nn_9 =  100.*(yy[ (yy >=1.0) ].size)/hh_total
    nn_10 = 100.*(yy[ (yy >=0.5) ].size)/hh_total
    nn_11 = 100.*(yy[ (yy >=0.25)].size)/hh_total
    nn_12 = 100.*(yy[ (yy >=0.10)].size)/hh_total
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("percent with drifts (cm/sec) greater than: \n\n")
    ff.write(" 200.   %7.3f  \n" % (nn_1)  )
    ff.write(" 100.   %7.3f  \n" % (nn_2))
    ff.write("  50.   %7.3f  \n" % (nn_3))
    ff.write("  25.   %7.3f  \n" % (nn_4))
    ff.write("  15.   %7.3f  \n" % (nn_5))
    ff.write("  10.   %7.3f  \n" % (nn_6))
    ff.write("   5.   %7.3f  \n" % (nn_7))
    ff.write("   2.5  %7.3f  \n" % (nn_8))
    ff.write("   1.0  %7.3f  \n" % (nn_9))
    ff.write("   0.5  %7.3f  \n" % (nn_10))
    ff.write("  0.25  %7.3f  \n" % (nn_11))
    ff.write("  0.10  %7.3f  \n" % (nn_12))
    ff.write("\n \n")

    # -------------------------------------------------------------
    
    yy = drift3_array[jj_ok]

    hh_total = yy.size
    nn_1 =  100.*(yy[ (yy >=200.)].size)/hh_total
    nn_2 =  100.*(yy[ (yy >=100.)].size)/hh_total
    nn_3 =  100.*(yy[ (yy >=50.) ].size)/hh_total
    nn_4 =  100.*(yy[ (yy >=25.) ].size)/hh_total
    nn_5 =  100.*(yy[ (yy >=15.) ].size)/hh_total
    nn_6 =  100.*(yy[ (yy >=10.) ].size)/hh_total
    nn_7 =  100.*(yy[ (yy >=5.)  ].size)/hh_total
    nn_8 =  100.*(yy[ (yy >=2.5) ].size)/hh_total
    nn_9 =  100.*(yy[ (yy >=1.0) ].size)/hh_total
    nn_10 = 100.*(yy[ (yy >=0.5) ].size)/hh_total
    nn_11 = 100.*(yy[ (yy >=0.25)].size)/hh_total
    nn_12 = 100.*(yy[ (yy >=0.10)].size)/hh_total
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("percent with drift-3 (cm/sec) greater than: \n\n")
    ff.write(" 200.   %7.3f  \n" % (nn_1)  )
    ff.write(" 100.   %7.3f  \n" % (nn_2))
    ff.write("  50.   %7.3f  \n" % (nn_3))
    ff.write("  25.   %7.3f  \n" % (nn_4))
    ff.write("  15.   %7.3f  \n" % (nn_5))
    ff.write("  10.   %7.3f  \n" % (nn_6))
    ff.write("   5.   %7.3f  \n" % (nn_7))
    ff.write("   2.5  %7.3f  \n" % (nn_8))
    ff.write("   1.0  %7.3f  \n" % (nn_9))
    ff.write("   0.5  %7.3f  \n" % (nn_10))
    ff.write("  0.25  %7.3f  \n" % (nn_11))
    ff.write("  0.10  %7.3f  \n" % (nn_12))
    ff.write("\n \n")

    # -------------------------------------------------------------
    
    yy = drift_rho[jj_ok]

    hh_total = yy.size
    
    nn_1 =  100.*(yy[ (yy >=5.)].size)/hh_total
    nn_2 =  100.*(yy[ (yy >=2.)].size)/hh_total
    nn_3 =  100.*(yy[ (yy >=1.) ].size)/hh_total
    nn_4 =  100.*(yy[ (yy >=0.5) ].size)/hh_total
    nn_5 =  100.*(yy[ (yy >=0.2) ].size)/hh_total
    nn_6 =  100.*(yy[ (yy >=0.1) ].size)/hh_total
    nn_7 =  100.*(yy[ (yy >=0.05)  ].size)/hh_total
    nn_8 =  100.*(yy[ (yy >=0.02) ].size)/hh_total
    nn_9 =  100.*(yy[ (yy >=0.01) ].size)/hh_total
    nn_10 = 100.*(yy[ (yy >=0.005) ].size)/hh_total
    nn_11 = 100.*(yy[ (yy >=0.002)].size)/hh_total
    nn_12 = 100.*(yy[ (yy >=0.001)].size)/hh_total
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("percent with drift-rho (sec^-1) greater than: \n\n")
    ff.write(" 5.    %7.3f   \n" % (nn_1)  )
    ff.write(" 2.    %7.3f   \n" % (nn_2))
    ff.write(" 1.    %7.3f   \n" % (nn_3))
    ff.write(" 0.5    %7.3f  \n" % (nn_4))
    ff.write(" 0.2    %7.3f  \n" % (nn_5))
    ff.write(" 0.1    %7.3f  \n" % (nn_6))
    ff.write(" 0.05   %7.3f  \n" % (nn_7))
    ff.write(" 0.02   %7.3f  \n" % (nn_8))
    ff.write(" 0.01   %7.3f  \n" % (nn_9))
    ff.write(" 0.005  %7.3f  \n" % (nn_10))
    ff.write(" 0.002  %7.3f  \n" % (nn_11))
    ff.write(" 0.001  %7.3f  \n" % (nn_12))
    ff.write("\n")
    ff.write(" maximum %7.3f \n" % (nn_max))
    ff.write("\n \n")
    
    
    # --------------------------------------------------------------

   # -------------------------------------------------------------
    
    yy = delta_rho[jj_ok]

    hh_total = yy.size
    
    nn_1 =  100.*(yy[ (yy >=0.5)].size)/hh_total
    nn_2 =  100.*(yy[ (yy >=0.2)].size)/hh_total
    nn_3 =  100.*(yy[ (yy >=0.1) ].size)/hh_total
    nn_4 =  100.*(yy[ (yy >=0.05) ].size)/hh_total
    nn_5 =  100.*(yy[ (yy >=0.02) ].size)/hh_total
    nn_6 =  100.*(yy[ (yy >=0.01) ].size)/hh_total
    nn_7 =  100.*(yy[ (yy >=0.005)  ].size)/hh_total
    nn_8 =  100.*(yy[ (yy >=0.002) ].size)/hh_total
    nn_9 =  100.*(yy[ (yy >=0.001) ].size)/hh_total
    nn_10 = 100.*(yy[ (yy >=5.e-4) ].size)/hh_total
    nn_11 = 100.*(yy[ (yy >=2.e-4)].size)/hh_total
    nn_12 = 100.*(yy[ (yy >=1.e-4)].size)/hh_total
    nn_13 = 100.*(yy[ (yy >=5.e-5)].size)/hh_total
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("percent with delta-rhogreater than: \n\n")
    ff.write(" 0.5    %7.3f   \n" % (nn_1)  )
    ff.write(" 0.2    %7.3f   \n" % (nn_2))
    ff.write(" 0.1    %7.3f   \n" % (nn_3))
    ff.write(" 0.05    %7.3f  \n" % (nn_4))
    ff.write(" 0.02    %7.3f  \n" % (nn_5))
    ff.write(" 0.01   %7.3f  \n" % (nn_6))
    ff.write(" 0.005   %7.3f  \n" % (nn_7))
    ff.write(" 0.002   %7.3f  \n" % (nn_8))
    ff.write(" 0.001   %7.3f  \n" % (nn_9))
    ff.write(" 5.e-4  %7.3f  \n" % (nn_10))
    ff.write(" 2.e-4  %7.3f  \n" % (nn_11))
    ff.write(" 1.e-4  %7.3f  \n" % (nn_12))
    ff.write(" 5.e-5  %7.3f  \n" % (nn_13))
    ff.write("\n \n")
 
    
    yy = drift_emax[jj_ok]
    nn_1 =  100.*(yy[ (yy >= 0.03) ].size)/hh_total
    nn_2 =  100.*(yy[ (yy >= 0.01) ].size)/hh_total
    nn_3 =  100.*(yy[ (yy >=0.003) ].size)/hh_total
    nn_4 =  100.*(yy[ (yy >=0.001) ].size)/hh_total
    nn_5 =  100.*(yy[ (yy >=3.e-4) ].size)/hh_total
    nn_6 =  100.*(yy[ (yy >=1.e04) ].size)/hh_total
    nn_7 =  100.*(yy[ (yy >=3.e-5) ].size)/hh_total
    nn_8 =  100.*(yy[ (yy >=1.e-5) ].size)/hh_total
    nn_9 =  100.*(yy[ (yy >=3.e-6) ].size)/hh_total
    nn_10 = 100.*(yy[ (yy >=1.e-6) ].size)/hh_total
    nn_11 = 100.*(yy[ (yy >=3.e-7) ].size)/hh_total
    nn_12 = 100.*(yy[ (yy >=1.e-7) ].size)/hh_total
    nn_13 = 100.*(yy[ (yy >=3.e-8) ].size)/hh_total
    nn_14 = 100.*(yy[ (yy >=1.e-8) ].size)/hh_total
    nn_max = np.max(yy)

    ff.write(" \n")
    ff.write("percent with drift_emax (rel) greater than: \n\n")
    ff.write(" 0.03    %7.3f  \n" % (nn_1)  )
    ff.write(" 0.01    %7.3f  \n" % (nn_2))
    ff.write(" 0.003   %7.3f  \n" % (nn_3))
    ff.write(" 0.001   %7.3f  \n" % (nn_4))
    ff.write(" 3.e-4   %7.3f  \n" % (nn_5))
    ff.write(" 1.e-4   %7.3f  \n" % (nn_6))
    ff.write(" 3.e-5   %7.3f  \n" % (nn_7))
    ff.write(" 1.e-5   %7.3f  \n" % (nn_8))
    ff.write(" 3.e-6   %7.3f  \n" % (nn_9))
    ff.write(" 1.e-6   %7.3f  \n" % (nn_10))
    ff.write(" 3.e-7   %7.3f  \n" % (nn_11))
    ff.write(" 1.e-7   %7.3f  \n" % (nn_12))
    ff.write(" 3.e-8   %7.3f  \n" % (nn_13))
    ff.write(" 1.e-8   %7.3f  \n" % (nn_14))
    ff.write("\n \n")
    ff.write(" average delta_time: %8.2e \n\n" % np.mean(mean_delta_time))
    ff.write(" minimum delta_time: %8.2e \n"   %  np.min(mean_delta_time))
    ff.write(" maximum delta_time: %8.2e \n"   %  np.max(mean_delta_time))
    ff.write(" stddev  delta_time: %8.2e \n"   %  np.std(mean_delta_time))
    
    # -------------------------------------
    #  print out markers with big drifts
    
    nbad = -30
    nbad_p = -nbad
    
    qq = np.argsort(drift_max)
    #pdb.set_trace()
    print("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = drift_max[qq[nbad:]]
    ff.write("  marker  drift \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e\n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(drift_max2)
    ff.write("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = drift_max2[qq[nbad:]]
    ff.write("  marker  drift2 \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(drift3_array)
    ff.write("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = drift_max2[qq[nbad:]]
    ff.write("  marker  drift3 \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(drift_rho)
    print("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = drift_rho[qq[nbad:]]
    ff.write("  marker  drift-rho \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.5f  %10.3e\n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(delta_rho)
    print("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = delta_rho[qq[nbad:]]
    ff.write("  marker  delta-rho \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.5f  %10.3e\n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    
    qq = np.argsort(drift_emin)
    ff.write("\n")
    bad_markers =  my_array[qq[nbad:]]
    bad_drifts  = drift_emin[qq[nbad:]]
    ff.write("  marker  drift_emin \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    print("\n")

    qq = np.argsort(drift_emax)
    ff.write("\n")
    bad_markers =   my_array[qq[nbad:]]
    bad_drifts  = drift_emax[qq[nbad:]]
    ff.write("  marker  drift_emax \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n " % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(drift_ctormin)
    ff.write("\n")
    bad_markers = my_array[qq[nbad:]]
    bad_drifts  = drift_ctormin[qq[nbad:]]
    ff.write("  marker  drift_ctormin \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")

    qq = np.argsort(drift_ctormax)
    ff.write("\n")
    bad_markers = my_array[qq[nbad:]]
    bad_drifts  = drift_ctormax[qq[nbad:]]
    ff.write("  marker  drift_ctormax \n \n")
    for ii in range(nbad_p):
        ff.write("  %4d  %9.3f  %10.3e \n" % (bad_markers[ii], bad_drifts[ii], bad_drifts[ii]))
    ff.write("\n")
    
    ff.close()  # close the text file
    
    with PdfPages(filename_summary) as pdf:

        my_symbol = 'ro'
        
        plt.close()
        plt.figure(figsize=(7.,5.))

        nn = (drift_max > 0.)
        xx_local = np.log10(drift_max[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift   max = %10.3e"%(np.max(drift_max))
        plt.title(my_title)
        plt.xlabel(' log_10 (drift rate) [cm/sec]')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_drift_rate.pdf'
        pdf.savefig()
        plt.close()

    
        plt.figure(figsize=(7.,5.))
        nn = (drift_max2 > 0.)
        xx_local = np.log10(drift_max2[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift2  max = %10.3e"%(np.max(drift_max2))
        plt.title(my_title)
        plt.xlabel('log_10 (drift rate-2) [cm/sec]')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_drift_rate_2.pdf'
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        nn = (drift3_array > 0.)
        xx_local = np.log10(drift3_array[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift3  max = %10.3e"%(np.max(drift3_array))
        plt.title(my_title)
        plt.xlabel('log_10 (drift rate-3) [cm/sec]')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_drift_rate_3.pdf'
        pdf.savefig()
        plt.close()



        plt.figure(figsize=(7.,5.))
        nn = (drift3_array > 0.)
        xx_local = np.log10(drift3_array[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift3  max = %10.3e"%(np.max(drift3_array))
        plt.title(my_title)
        plt.xlabel('log_10 (drift rate-3) [cm/sec]')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_drift_rate_3.pdf'
        pdf.savefig()
        plt.close()

        
    
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = ( drift_rho > 0.)
        xx_local = np.log10(drift_rho[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift in rho  max = %10.3e"%(np.max(drift_rho))
        plt.title(my_title)
        plt.xlabel('log_10(drift rho)')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_rho_drift.pdf'
        pdf.savefig()
        plt.close()



        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = ( drift_rho > 0.)
        xx_local = np.log10(drift_rho[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift in rho  max = %10.3e"%(np.max(drift_rho))
        plt.title(my_title)
        plt.xlabel('log_10(drift rho)')
        plt.xlim((-4,1))
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_rho_drift.pdf'
        pdf.savefig()
        plt.close()


        

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = ( delta_rho > 0.)
        xx_local = np.log10(delta_rho[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Delta rho  max = %10.3e"%(np.max(delta_rho))
        plt.title(my_title)
        plt.xlabel('log_10(delta rho)')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_rho_delta.pdf'
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = ( delta_rho > 0.)
        xx_local = np.log10(delta_rho[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Delta rho  max = %10.3e"%(np.max(delta_rho))
        plt.title(my_title)
        plt.xlabel('log_10(delta rho)')
        plt.ylabel('')
        plt.xlim((-5.,0.))
        plt.tight_layout(pad=1)
        plotfile = stub + '_rho_delta.pdf'
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_emax > 0.)
        xx_local = np.log10(drift_emax[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift in Emax  max = %10.3e"%(np.max(drift_emax))
        plt.title(my_title)
        plt.xlabel('log_10(delta Emax / Emax)')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        plotfile = stub + '_emax_drift.pdf'
        pdf.savefig()
        plt.close()
        
        #plt.figure(figsize=(7.,5.))
        #plt.hist(drift_mumin, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        #plt.title('Drift in mu_min')
        #plt.xlabel('delta mu_min / mu_min')
        #plt.ylabel('')   
        #plt.tight_layout(pad=1)
        #plotfile = stub + '_mumin_drift.pdf'
        #pdf.savefig()
        #plt.close()

        #plt.figure(figsize=(7.,5.))
        #plt.hist(drift_mumax, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        #plt.title('Drift in mu_max')
        #plt.xlabel('delta mu_max / mu_max')
        #plt.ylabel('')   
        #plt.tight_layout(pad=1)
        #plotfile = stub + '_mumax_drift.pdf'
        #pdf.savefig()
        #plt.close()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_ctormin > 0.)
        xx_local = np.log10(drift_ctormin[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift in ctor_min  max = %10.3e"%(np.max(drift_ctormin))
        plt.title(my_title)
        plt.xlabel('log_10(delta ctor_min / ctor_min)')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        #plt.xlim((1.e-2, 1000.))
        #plt.ylim((1.e-2, 1000.))
        plotfile = stub + '_ctormin_drift.pdf'
        pdf.savefig()
        plt.close()
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_ctormax > 0.)
        xx_local = np.log10(drift_ctormax[nn])
        plt.hist(xx_local, bins=30, histtype='step', rwidth=1, color='b', log=True)
        #plt.grid(axis='both', alpha=0.75)
        my_title = " Drift in ctormax  max = %10.3e"%(np.max(drift_ctormax))
        plt.title(my_title)
        plt.xlabel('log_10(delta ctor_max / ctor_max)')
        plt.ylabel('')   
        plt.tight_layout(pad=1)
        #plt.xlim((1.e-2, 1000.))
        #plt.ylim((1.e-2, 1000.))
        plotfile = stub + '_ctormax_drift.pdf'
        pdf.savefig()
        plt.close()

        xx_1 = np.linspace(1.e-5, 1000.)
        yy_1 = xx_1
        plt.close()
        nn = (drift_max > 0.) & (drift_max2 > 0)
        plt.figure(figsize=(7.,5.))
        plt.plot(drift_max[nn], drift_max2[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.plot(xx_1, yy_1, 'k-', linewidth=1)
        plt.xlabel('drift_max [cm/s]')
        plt.ylabel('drift_max2')
        plt.xlim((1.e-2, 1000.))
        plt.ylim((1.e-4, 10.))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_max2 vs drift_max')
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_max2 > 0) & (drift_emax > 0)
        plt.plot(drift_max2[nn], drift_emax[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.xlabel('drift_max2 [cm/s]')
        plt.ylabel('drift_emax [rel]')
        #plt.xlim((1.e-3, 100.))
        #plt.ylim((1.e-3, 100.))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_emax vs drift_max2')
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_max2 > 0) & (drift_emax > 0)
        plt.plot(drift_max2[nn], drift_emax[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.xlabel('drift_max2 [cm/s]')
        plt.ylabel('drift_emax [rel]')
        plt.xlim((1.e-4, 10.))
        plt.ylim((1.e-9, 1.e-3))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_emax vs drift_max2')
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_emax > 0.) & (drift_ctormax > 0.)
        plt.plot(drift_emax[nn],drift_ctormax[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.xlabel('drift_emax')
        plt.ylabel('ctormax')
        #plt.xlim((1.e-3, 100.))
        #plt.ylim((1.e-3, 100.))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_ctormax vs drift_emax')
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift_emax > 0.) & (drift_ctormax > 0.)
        plt.plot(drift_emax[nn],drift_ctormax[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.xlabel('drift_emax')
        plt.ylabel('ctormax')
        plt.xlim((1.e-9, 1.e-3))
        plt.ylim((1.e-8, 3.e-3))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_ctormax vs drift_emax')
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        nn = (drift3_array > 0.) & (drift_rho > 0.)
        plt.plot(drift3_array[nn],drift_rho[nn], my_symbol, fillstyle='none', ms=4, rasterized=do_rasterized)
        plt.xlabel('drift_emax')
        plt.ylabel('ctormax')
        #plt.xlim((1.e-9, 1.e-3))
        #plt.ylim((1.e-8, 3.e-3))
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('drift_rho vs drift3_array')
        pdf.savefig()
    
def check_01(fn_hdf5):
    """
    Plot the results of these tests.

    This function makes four plots.
    - One that shows conservation of energy for all cases
    - One that shows conservation of magnetic moment for all cases
    - One that shows conservation of toroidal canoncical momentum for all cases
    - And one that shows trajectories on a Rz plane for all cases
    """
    a5 = ascot5.Ascot(fn_hdf5)
    apy = Ascotpy(fn_hdf5)
    apy.init(bfield=True)

    raxis = R0            #a5["ORBFOL_GO"].bfield.read()["raxis"]

    #****************************************************#
    #*     Evaluate quantities for ORBFOL_GO             #
    #*                                                   #
    #****************************************************#

    ORBFOL = {}
    ORBFOL["GO"] = {}
    print('')
    print('contents of a5')
    print(a5)
    print('------------------------------')
    orb = a5["ORBFOL_GO"]["orbit"].read()
    
    B = np.sqrt( orb["br"] * orb["br"] + orb["bphi"] * orb["bphi"] +
                 orb["bz"] * orb["bz"] )

    # originally got psi from the ITER equilibrium
    #psi = psifun(orb["r"]/raxis, orb["z"]/raxis, psi_coeff[0], psi_coeff[1],
    #             psi_coeff[2], psi_coeff[3], psi_coeff[4], psi_coeff[5],
    #             psi_coeff[6], psi_coeff[7], psi_coeff[8], psi_coeff[9],
    #             psi_coeff[10], psi_coeff[11], psi_coeff[12]) * psi_mult

    #  we will now read psi directly from the hdf5 file
    
    psi_GO_along_orbit = apy.evaluate(orb["r"], orb["phi"], orb["z"], orb["time"], "psi")
    
    vnorm = np.sqrt( orb["vr"]   * orb["vr"] +
                     orb["vphi"] * orb["vphi"] +
                     orb["vz"]   * orb["vz"] )

    vpar  = ( orb["vr"] * orb["br"] + orb["vphi"] * orb["bphi"] +
              orb["vz"] * orb["bz"] ) / B

    gamma = np.sqrt(1 / ( 1 - vnorm * vnorm / (c * c) ) )

    ORBFOL["GO"]["time"] = orb["time"]
    ORBFOL["GO"]["id"]   = orb["id"]
    ORBFOL["GO"]["r"]    = orb["r"]
    ORBFOL["GO"]["z"]    = orb["z"]
    ORBFOL["GO"]["ekin"] = (gamma - 1) * m_e * c * c
    ORBFOL["GO"]["mu"]   = ( ( m_e * gamma * gamma ) / ( 2 * B ) ) * \
                           ( vnorm * vnorm - vpar * vpar )

    #print(" shape of psi:        ", psi.shape)
    #print(" shape of orb-r:      ", orb["r"].shape)
    #print(" shape of orb-vphi:   ", orb["vphi"].shape)
    #print(" shape of orb-charge: ", orb["charge"].shape)
    
    #ORBFOL["GO"]["ctor"] = gamma * m_e * orb["r"] * orb["vphi"] + orb["charge"] * e * psi_GO_along_orbit

    #********************************************************#
    #*     Evaluate quantities for ORBFOL_GCF                #
    #*                                                       #
    #********************************************************#
    
    ORBFOL["GCF"] = {}
    orb = a5["ORBFOL_GCF"]["orbit"].read()

    B = np.sqrt(np.power(orb["br"],2) + np.power(orb["bphi"],2) +
                np.power(orb["bz"],2))

    #psi = psifun(orb["r"]/raxis, orb["z"]/raxis, psi_coeff[0], psi_coeff[1],
    #             psi_coeff[2], psi_coeff[3], psi_coeff[4], psi_coeff[5],
    #             psi_coeff[6], psi_coeff[7], psi_coeff[8], psi_coeff[9],
    #             psi_coeff[10], psi_coeff[11], psi_coeff[12]) * psi_mult


    
    # Note that mu is in eV / T

    #psi_GCF_along_orbit = apy.evaluate(orb["r"], orb["phi"], orb["z"], orb["time"], "psi")
    
   
    gamma = np.sqrt( ( 1 + 2 * orb["mu"] * e * B / ( m_e * c * c ) ) /
                     ( 1 - orb["vpar"] * orb["vpar"] / ( c * c ) ) )

    ORBFOL["GCF"]["time"] = orb["time"]
    ORBFOL["GCF"]["id"]   = orb["id"]
    ORBFOL["GCF"]["r"]    = orb["r"]
    ORBFOL["GCF"]["z"]    = orb["z"]
    ORBFOL["GCF"]["ekin"] = (gamma - 1) * m_e * c * c
    ORBFOL["GCF"]["mu"]   = orb["mu"] * e
    #ORBFOL["GCF"]["ctor"] = gamma * m_e * orb["r"] * orb["vpar"] + orb["charge"] * e * psi_GCF_along_orbit

    print("")
    print("GCF")
    print(" shape of time: ", ORBFOL["GCF"]["time"].shape)
    print(" shape of ekin: ", ORBFOL["GCF"]["ekin"].shape)
    print(" shape of mu:   ", ORBFOL["GCF"]["mu"].shape)
    
    

    #****************************************************#
    #*     Evaluate quantities for ORBFOL_GCA            #
    #*                                                   #
    #****************************************************#
    ORBFOL["GCA"] = {}
    orb = a5["ORBFOL_GCA"]["orbit"].read()

    B = np.sqrt(np.power(orb["br"],2) + np.power(orb["bphi"],2) +
                np.power(orb["bz"],2))

    #psi = psifun(orb["r"]/raxis, orb["z"]/raxis, psi_coeff[0], psi_coeff[1],
    #             psi_coeff[2], psi_coeff[3], psi_coeff[4], psi_coeff[5],
    #             psi_coeff[6], psi_coeff[7], psi_coeff[8], psi_coeff[9],
    #             psi_coeff[10], psi_coeff[11], psi_coeff[12]) * psi_mult

    
    # Note that mu is in eV / T

    #psi_GCA_along_orbit = apy.evaluate(orb["r"], orb["phi"], orb["z"], orb["time"], "psi")
    
    gamma = np.sqrt( ( 1 + 2 * orb["mu"] * e * B / ( m_e * c * c ) ) /
                     ( 1 - orb["vpar"] * orb["vpar"] / ( c * c ) ) )

    ORBFOL["GCA"]["time"] = orb["time"]
    ORBFOL["GCA"]["id"]   = orb["id"]
    ORBFOL["GCA"]["r"]    = orb["r"]
    ORBFOL["GCA"]["z"]    = orb["z"]
    ORBFOL["GCA"]["ekin"] = (gamma - 1) * m_e * c * c
    ORBFOL["GCA"]["mu"]   = orb["mu"] * e
    #ORBFOL["GCA"]["ctor"] = gamma * m_e * orb["r"] * orb["vpar"] + orb["charge"] * e * psi_GCA_along_orbit

    
    #***************************************#
    #*     make the plots                   #
    #*                                      #
    #***************************************#

    max_marker = np.min([np.max(orb["id"]),1000])   # added np.min 7/15/2020
    id_array   = np.linspace(1, max_marker,max_marker).astype(int)
    stub_name  = fn_hdf5.split('.')[0]   # for generating unique plot filenames
    pdb.set_trace()
    
    for marker in id_array:   # id_array
    
        f = plt.figure(figsize=(11.9/2.54, 8/2.54))
    
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes',  labelsize=10)
    
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        h1 = f.add_subplot(1,4,1)
        h1.set_position([0.12, 0.72, 0.4, 0.25], which='both')

        h2 = f.add_subplot(1,4,2)
        h2.set_position([0.12, 0.44, 0.4, 0.25], which='both')

        h3 = f.add_subplot(1,4,3)
        h3.set_position([0.12, 0.155, 0.4, 0.25], which='both')

        h4 = f.add_subplot(1,4,4)
        h4.set_position([0.52, 0.15, 0.60, 0.75], which='both')

        #colors = ["#373e02", "#0a481e", "#03719c", "#0165fc", "#7e1e9c", "#cea2fd"]
        colors = ["b", "r", "forestgreen", "cyan", "magenta", "dodgerblue", "r", "tomato"]
        
        id_go  = ORBFOL["GO"]["id"]  == marker
        id_gcf = ORBFOL["GCF"]["id"] == marker
        id_gca = ORBFOL["GCA"]["id"] == marker

        #import pdb
        #pdb.set_trace()
        
        plot_relerr(h1, ORBFOL["GO"]["time"][id_go], ORBFOL["GO"]["ekin"][id_go], colors[0])
        plot_relerr(h2, ORBFOL["GO"]["time"][id_go], ORBFOL["GO"]["mu"][id_go],   colors[0])
        #plot_relerr(h3, ORBFOL["GO"]["time"][id_go], ORBFOL["GO"]["ctor"][id_go], colors[0])
        h4.plot(        ORBFOL["GO"]["r"][id_go],    ORBFOL["GO"]["z"][id_go],    colors[0])

        plot_relerr(h1, ORBFOL["GCF"]["time"][id_gcf], ORBFOL["GCF"]["ekin"][id_gcf], colors[1])
        plot_relerr(h2, ORBFOL["GCF"]["time"][id_gcf], ORBFOL["GCF"]["mu"][id_gcf],   colors[1])
        #plot_relerr(h3, ORBFOL["GCF"]["time"][id_gcf], ORBFOL["GCF"]["ctor"][id_gcf], colors[1])
        h4.plot(        ORBFOL["GCF"]["r"][id_gcf],    ORBFOL["GCF"]["z"][id_gcf],    colors[1])
    
        plot_relerr(h1, ORBFOL["GCA"]["time"][id_gca], ORBFOL["GCA"]["ekin"][id_gca],  colors[2])
        plot_relerr(h2, ORBFOL["GCA"]["time"][id_gca], ORBFOL["GCA"]["mu"][id_gca],    colors[2])
        #plot_relerr(h3, ORBFOL["GCA"]["time"][id_gca], ORBFOL["GCA"]["ctor"][id_gca],  colors[2])
        h4.plot(        ORBFOL["GCA"]["r"][id_gca],    ORBFOL["GCA"]["z"][id_gca], colors[2])

        #***********************************************************#
        #*   Finalize and print and show the figure                 #
        #*                                                          #
        #***********************************************************#


        tgo_min  = np.min(ORBFOL["GO"]["time"][id_go])
        tgcf_min = np.min(ORBFOL["GCF"]["time"][id_gcf])
        tgca_min = np.min(ORBFOL["GCA"]["time"][id_gca])
        
        tgo_max  = np.max(ORBFOL["GO"]["time"][id_go])
        tgcf_max = np.max(ORBFOL["GCF"]["time"][id_gcf])
        tgca_max = np.max(ORBFOL["GCA"]["time"][id_gca])

        print(" go:  time min, max: ", tgo_min, tgo_max)
        print(" gcf:  time min, max: ", tgcf_min, tgcf_max)
        print(" gca:  time min, max: ", tgca_min, tgca_max)
        

        time_array = np.array((tgo_max, tgcf_max, tgca_max))
        time_max   = np.max(time_array)                         # use same max time for all graphs

        microsec_ticks = int(1.e6*time_max)+1
        
        my_ticks = np.linspace(0,microsec_ticks, microsec_ticks+1)*1.e-6
        float_ticklabels = np.linspace(0,microsec_ticks, microsec_ticks+1)
        my_ticklabels = [ int(i) for i in float_ticklabels]
        
        h1.set_xlim(0, time_max)
        h1.xaxis.set(ticks=my_ticks, ticklabels=[])
        # h1.yaxis.set(ticks=np.array([-20, -10, 0, 10, 20])*1e-10, ticklabels=[-20, '', 0, '', 20])
        h1.tick_params(axis='y', direction='in')
        h1.tick_params(axis='x', direction='in')
        h1.spines['right'].set_visible(False)
        h1.spines['top'].set_visible(False)
        h1.yaxis.set_ticks_position('left')
        h1.xaxis.set_ticks_position('bottom')
        h1.set(ylabel=r"$\Delta E/E_0\;[10^{-11}]$")

        h2.set_xlim(0, time_max)
        h2.xaxis.set(ticks=my_ticks, ticklabels=[])
        h2.yaxis.set(ticks=np.array([-10, 0, 10])*1e-3, ticklabels=[-10, 0, 10])
        h2.tick_params(axis='y', direction='in')
        h2.tick_params(axis='x', direction='in')
        h2.spines['right'].set_visible(False)
        h2.spines['top'].set_visible(False)
        h2.yaxis.set_ticks_position('left')
        h2.xaxis.set_ticks_position('bottom')
        h2.set(ylabel=r"$\Delta \mu/\mu_0\;[10^{-3}]$")

        h3.set_xlim(0, time_max)
        h3.xaxis.set(ticks=my_ticks,ticklabels=my_ticklabels)
        h3.yaxis.set(ticks=np.array([-6, 0, 6])*1e-4, ticklabels=[-6, 0, 6])
        h3.tick_params(axis='y', direction='in')
        h3.tick_params(axis='x', direction='in')
        h3.spines['right'].set_visible(False)
        h3.spines['top'].set_visible(False)
        h3.yaxis.set_ticks_position('left')
        h3.xaxis.set_ticks_position('bottom')
        h3.set(ylabel=r"$\Delta P/P_0\;[10^{-4}]$", xlabel=r"Time [$10^{-6}$ s]")

        h4.axis('scaled')
        h4.set_xlim(1.,2.4)
        h4.set_ylim(-1.2,1.2)
        h4.yaxis.set(ticks=[-0.8,  -0.4,  0., 0.4,  0.8 ])
        h4.xaxis.set(ticks=[1.2,  1.6,  2.0])
        h4.tick_params(axis='y', direction='in')
        h4.tick_params(axis='x', direction='in')
        h4.set(xlabel="$R$ [m]", ylabel="$z$ [m]")

        legend = [r"GO", r"GCF", r"GCA"]
        
        h4.text(1.2, 0.95, legend[0], fontsize=9, color=colors[0])
        h4.text(1.5, 0.95, legend[1], fontsize=9, color=colors[1])
        h4.text(1.8, 0.95, legend[2], fontsize=9, color=colors[2])

        plot_title = 'marker: ' + str(marker)
        plt.title(plot_title)

        plotfile_name = stub_name + '_marker_' + str(marker) + '.pdf'
        plt.savefig(plotfile_name, dpi=300)
        plt.show()

def plot_relerr(axis, x, y, color):    
    axis.plot(x, y/y[0] - 1, color, rasterized=True) # rasterization added 11/2/19 per krs, then removed


