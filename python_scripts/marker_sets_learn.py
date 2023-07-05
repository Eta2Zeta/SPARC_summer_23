from scipy.interpolate import RectBivariateSpline
import wrapper_geqdskfile as WG
import compute_vperp_hat as CV
import compute_bfield_arrays_02 as CB
import sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy                   as np
import get_rho_interpolator as get_rho
import a5py.ascot5io.mrk_gc    as mrk
from scipy.constants import physical_constants as const
from a5py.ascot5io import mrk_prt
import pdb
import matplotlib.pyplot as plt
import numpy as np
from readGeqdsk import geqdsk
from scipy import interpolate
import matplotlib as mpl
import sparc_processing as proc
import process_ascot as process_ascot
import read_any_file as read_any
import read_any_file as RR
#import scipy.constants as const
import h5py
import time as clock
import plotlossmap_AT as loss
from a5py.marker.pploss_AT import applylossmap
import get_ascot_AT as   get
import compute_rho as rho
import validate_bfield as VB
from a5py.ascot5io.ascot5 import Ascot
from a5py.ascotpy.ascotpy import Ascotpy
from shapely.geometry import Point, Polygon   # 7/14/22
import time as time
import sds_utilities   as sds
import scipy.constants             as constants
import construct_velocity_vector as CVV
import vector_arithmetic         as VA

#  the following added 3/27/23.  I will adopt convention that
#  global constants are in uppercase.
    
ELECTRON_CHARGE   = constants.elementary_charge
ELECTRON_MASS     = constants.physical_constants["electron mass"][0]
LIGHT_SPEED       = constants.physical_constants["speed of light in vacuum"][0]
ALPHA_MASS        = constants.physical_constants["alpha particle mass"][0]
PROTON_MASS       = constants.physical_constants["proton mass"][0]
AMU_MASS          = constants.physical_constants["unified atomic mass unit"][0]
ELECTRON_MASS_AMU = ELECTRON_MASS / AMU_MASS

# Used in markers_set_1,5,8 and 10
def construct_rho_interpolator(fn_geqdsk, eq_index):

    # --------------------------------------
    #  get the equliibrium and [R,Z] shape of LCFS
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs
    #print("   ... construct_rho_interpolator:  I have read the equilibrium")

    # ---------------------------------------------

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

    # ---------------------------------------------
    #

    #  get a function that we can use to map (R,z) --> rho

    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)

    return rho_interpolator, geq_rarray, geq_zarray, rhogeq_transpose_2d

# Useed in the markers_set_1
def compute_rho(mm, tolerance, rmajor, aminor, kappa, delta, rr, zz):



    # we do not care if this is inaccurate for rho > 1.0 since
    # we will exclude those markers anyway
    
    # mm = dimension of arrays

    if ( abs(rmajor - rr) < 0.015):   #  special case
        
        if (zz >= 0):
            theta = np.pi/2.
            for kk in range(10):
                theta = np.pi/2. - delta * np.sin(theta)
        else:
            theta = -1. * np.pi/2.
            for kk in range(10):
                theta = -1.*np.pi/2. - delta * np.sin(theta)
        #print("position-1")
        #pdb.set_trace()
        rho = zz / (aminor * kappa * np.sin(theta))
                    
    else:
                    
        if(zz >= 0.):
           thetas = np.linspace(0., np.pi, mm)
        else:
            thetas = np.linspace(np.pi, 2.*np.pi, mm)

        my_fun = np.zeros(mm)

        for ii in range(mm):
            my_fun[ii] = np.sin(thetas[ii]) / np.cos( thetas[ii] + delta * np.sin(thetas[ii]) )

        rhs = zz / (kappa * (rr - rmajor))

        error_signed = my_fun - rhs
        error      = abs(my_fun - rhs)
        best       = np.unravel_index(error.argmin(), error.shape)

        #pdb.set_trace()
    
        theta   = thetas[best]
        # do a linear fit to get better estimate of theta
    
        gg = best[0]
    
        if (gg >= 1) & (gg <= thetas.size-2):
            ii = [gg-1, gg, gg+1]
        elif (gg == 0):
            ii = [0,1]
        else:
            ii = [gg-1,gg]

        xx = thetas[ii]
        yy = error_signed[ii]
        coeffs = np.polyfit(xx, yy, 1)
        theta = -1.*coeffs[1]/coeffs[0]
        
        #pdb.set_trace()
    
        if (theta == 0.):
            rho = abs(rr - rmajor) / aminor
        else:   
            rho     = zz / (aminor * kappa * np.sin(theta))

    rr_calc = rmajor + aminor * rho * np.cos( theta + delta * np.sin(theta))
    zz_calc = aminor * kappa * rho * np.sin(theta)
    
    best_error = np.sqrt( (rr-rr_calc)**2 + (zz-zz_calc)**2)

    if(best_error > tolerance) & (rho < 1.05):
        print("  compute_rho:  best_error, tolerance: ", best_error, tolerance, " so I must quit")
        pdb.set_trace()
        sys.exit()
    #print(' rho, theta, error: ', rho, theta, error[best])
    #print('delta R: ', rr - rr_calc[best])
    #print('delta z: ', zz - zz_calc[best])

    out={}

    out['rho']   = rho
    out['theta'] = theta
    out['error'] = best_error
    out['rr']    = rr_calc
    out['zz']    = zz_calc

    # pdb.set_trace()

    return out

# Used in the markers_set_uniform
def compute_bhats(R, phi, z, BB_3D):

        #  s scott 4/8/2023
        #
        #  compute the local direction of the magnetic field vector at the
        #  [R,phi,z] location of each marker
        #
        # contents of dictionary BB_3D
        #
        # b_nr,     b_nphi, b_nz  number of grid points
        # b_rmin,   b_rmax        minimum, maximum radius of rmajor grid [m]
        # b_phimin, b_phimax      minimum, maximum phi of toroidal-angle grid [radians]
        # b_zmin,   b_zmax        minimum, maximum elevation of z-grid [m]
        #
        # br(nr,nphi,nz)    radial magnetic field [Tesla]
        # bphi(nr,nphi,nz)  toroidal magnetic field [Tesla]
        # bz(nr,nphi,nz)    vertical magnetic field [Tesla]

        # compute local unit vector of B-field direction at R, phi, z location
        # of each birth marker

        br          = BB_3D["br"]       # components of B-field 
        bphi        = BB_3D["bphi"]
        bz          = BB_3D["bz"]
        
        rrs_B3D     = np.linspace(BB_3D["b_rmin"],   BB_3D["b_rmax"],   BB_3D["b_nr"])
        phis_B3D    = np.linspace(BB_3D["b_phimin"], BB_3D["b_phimax"], BB_3D["b_nphi"])
        zzs_B3D     = np.linspace(BB_3D["b_zmin"],   BB_3D["b_zmax"],   BB_3D["b_nz"])

        
        nn  = R.size
        ii_rr  = np.zeros(nn)               # was (incorrect) BB_3D["b_nr"]
        ii_phi = np.zeros(nn)
        ii_zz  = np.zeros(nn)

        for jj in range(nn):                                 # indices in B-field rmajor array closest to marker R
             ii_rr[jj] = np.absolute(R[jj]    - rrs_B3D).argmin()
             
        for jj in range(nn):                                # indices in B-field phi array closest to marker phi
             ii_phi[jj] = np.absolute(phi[jj] - phis_B3D).argmin()

        for jj in range(nn):                                  # indices in B-field rmajor array closest to marker z
             ii_zz[jj] = np.absolute(z[jj]    - zzs_B3D).argmin()

        ii_rr  = ii_rr.astype(int)
        ii_phi = ii_phi.astype(int)
        ii_zz  = ii_zz.astype(int)
        
 
        bhats = np.zeros((nn,3))

        btots = np.sqrt( br**2 + bphi**2 + bz**2)
        #pdb.set_trace()
        #  note that we do not currently interpolate into the 3D Bfield array to get the
        #  local direction of B, rather we just take the closest grid point to the
        #  [R, phi, z] location of the marker
    
        for jj in range(nn):
            #print("  jj = ", jj)
            bfield_vector_3d = np.array((   br[  ii_rr[jj],  ii_phi[jj], ii_zz[jj]  ],  \
                                          bphi[  ii_rr[jj],  ii_phi[jj], ii_zz[jj]  ],  \
                                            bz[  ii_rr[jj],  ii_phi[jj], ii_zz[jj]  ]     ))
            
            bhat         = VA.vector_hat(bfield_vector_3d)
            bhats[jj,:]  = bhat                              #  replaces bfield_vector_3d/btots[ii_rr[jj],  ii_phi[jj], ii_zz[jj]]
        
        return bhats

# Used in the markers_set_3
def compute_random_pitch(Nmrk,pitch_min, pitch_max):

    #   Only used in markers set 3
    
    nn_pitch        = np.max((3* Nmrk,50))
    pitch_candidate = -1 + 2 * np.random.rand(nn_pitch)
    pitch_weights   = np.sqrt(1-pitch_candidate**2)
    rr_pitch        = np.random.rand(nn_pitch)
    pitch_out       = np.zeros(nn_pitch)
    mm_pitch        = 0
         
    for kk in range(nn_pitch):
        if(rr_pitch[kk] <= pitch_weights[kk]):
            pitch_out[mm_pitch] = pitch_candidate[kk]
            mm_pitch += 1
    pitch_out = pitch_out[0:mm_pitch-1]
    kk_good = (pitch_out >= pitch_min) & (pitch_out <= pitch_max)
    pitch_out = pitch_out[kk_good]
    mm_good   = pitch_out.size
    if(mm_good < Nmrk):
        print("  ... compute_random_pitch:I could not find enough good pitch angles")
        exit()
    else:
        pitch = pitch_out[0:Nmrk]
    return pitch

         # ---------------------------------------------------------


def define_prt_markers_uniform(settings):
    #  "settings" is an input dictionary that contains user-supplied
    #  values that control the construction of the ensemble of markers
    #
    #  To the best of my knowledge, the ONLY information that is supplied to this module comes through
    #  the "settings" dictionary.  There are no global variables or other complications.
    
    #  Purpose of this module:  generate an ensemble of markers that is distributed
    #  uniformily (rather than randomly) in phase space. But this is not quite true ...
    #  to be precise, this module generates an ensemble of markers that is distributed
    #  uniformily in [R,Z] space and uniformily in velocity-direction.
    #
    #  Alas, being uniformily in [R,Z] space is not quite the same as being
    #  distributed uniformily in 3D space.  The reason for this is that the
    #  volume element in toroidal coordinates is dV = (2 pi R dphi) * dR * dZ.
    #
    #  So if we really wanted to construct a marker ensemble that is distributed
    #  uniformily in 3D space, we would have to figure out how to properly deal with
    #  that.  Presumably, there is a simple way to do that (maybe a homework assignment
    #  for an energetic student?) but instead, this module simply constructs
    #  the marker ensemble that is uniformily distributed in [R,Z] space, and
    #  then adds an additional weighting factor that is proportional to Rmajor.
    #  
    #  zzz:  To handle the fact that the volume increases with R,
    #  the weights will be proportional to Rmajor.  
    #
    #
    #    Nmrk    nominal total number of markers             (input)
    #    Nphi    number of grid points in the phi direction  (input)
    #    Npitch  number of grid points in pitch              (input)
    #    Ngyro   number of grid points in gyro-angle         (input)
    #    NR      computed to yield approx Nmrk markers and   (computed)
    #            so that the grid resolution is about the
    #            same in the radial and vertical directions
    #    NZ      computed to yield approx Nmrk markers and   (computed)
    #            so that the grid resolution is about the
    #            same in the radial and vertical directions
    
    mpl.rcParams['image.composite_image'] = True    # minimize size of PDF

    time_before = clock.time()
    
    electron_charge = 1.602e-19

    #  set=1   markers are the usual alpha particles or RF tail etc.
    #  set=2   markers are relativistic electrons
    #  TODO: set=3 make a center weighted distribution (maybe copy what set=1 do and modify it)
    set = settings["set"]

    #  Define the grid in Rmajor-space and Z-space over this the ensemble will
    #  be constructed.
    #
    #  We also allow the user to change the sign of the magnetic field.  This allows for
    #  the possibility that SPARC and SPIRAL may have different sign conventions for
    #  the magnetic field ... always a source of confusion.
    
    # allows us to change sign of B from what is in the fn_geqdsk
    # b is magnetic field, tor is toroidal field 
    try:
        btor_multiplier        = settings["btor_multiplier"]         
    except:
        btor_multiplier = 1.
        
    try:                                          
        rmajor_min = settings["rmajor_min"]
    except:
        rmajor_min = 0.
    try:                                          
        rmajor_max = settings["rmajor_max"]
    except:
        rmajor_max = 0.
    try:                                      
        z_min = settings["z_min"]
    except:
        z_min = 0.
    try:                                      
        z_max = settings["z_max"]
    except:
        z_max = 0.

    #  Generally, we give the magnetic field to ASCOT from two sources:
    #
    #   1.  The magnetic field generated by the discrete toroidal field (TF) coils.
    #       If we had an infinite number of TF coils, this magnetic field would be
    #       entirely in the toroidal direction, with zero values for the components
    #       in the major radius direction and vertical direction.
    #
    #       We do in fact have the capability to define the magnetic field
    #       generated by the TF coils by pretending that there are an infinite
    #       number of TF coils (confusingly, this is called a "2D" field),
    #       but more typically we supply a 3D field that has large components in
    #       the toroidal direction, and small -- but not zero -- components in the
    #       major-radius and vertical direction.
    #
    #       Then, separate from whatever magnetic field that is generated by the
    #       TF coils, we supply the poloidal magnetic field that is generated
    #       by the plasma itself -- this is called the 'equilibrium'.
    #
    #       This is the approach that we will take for alpha simulations in
    #       the presence of AE modes.
    #   2.  In some simulations, for example simulations of runaway electrons (REs)
    #       during disruptions, we get the entire magnetic field form a single source.
    #
    #  "rho" is a radial coordinate that varies from 0 at the magnetic axis to 1.000 at
    #  the plasma edge.

    #  extract user-supplied values from the "settings" dictionary  
    
    bfield_single_file     = settings["bfield_single_file"]     # If true, get entire B-field from bfield_single_filename
    bfield_single_filename = settings["bfield_single_filename"]
    birth_rhomin           = settings["birth_rhomin"]           # markers with rho_poloidal less than birth_rhomin will be excluded
    birth_rhomax           = settings["birth_rhomax"]           # markers with rho_poloidal greater than birth_rhomax will be excluded
    Nmrk                   = settings["Nmrk"]                   # the generated ensemble of markers will have this number of markers
    Pitch_min              = settings["Pitch_min"]              # pitch is defined as the component of alpha velocity along the magnetic field                                                 
    Pitch_max              = settings["Pitch_max"]              # to the total velocity (at birth).  so pitch=1.0 means that the alpha's velocity
                                                                # entirely parallel to B, pitch=-1 means that the alpha's velocity is
                                                                # entirely anti-parllel to B, and pitch=0 means that the alpha's velocity
                                                                # is perpendicular to B.
    Phi_min                = settings["Phi_min"]
    Phi_max                = settings["Phi_max"]
    Ekev                   = settings["Ekev"]                   # For DT-generated alphas, Ekev will be 3500. This is the from relativity 
    mass_amu               = settings["mass_amu"]               # Mass of markers in units of AMU.  I think this is ignored for set=2
    q                      = settings["q"]                      # charge, e.g. 1 or 2
    eq_index               = settings["index"]                  # index number in equilibrium.  this will generally be 0.  In principle,
                                                                # an "equilibrium file" (which generally has the extension .geq) can have
                                                                # data for multiple time slices, so we need the ability to be able to select
                                                                # the time slice of interest.
    gyro_angle_fixed       = settings["gyro_fixed"]             # use this gyro angle if randomize_gyro = F
    fn_geqdsk              = settings["fn_geqdsk"]              # name of equilibriumfile
    fn_hdf5                = settings["fn_hdf5"]                # name of ASCOT input file into which the marker info will be written
    fn_profiles            = settings["fn_profiles"]            # filename of file that contains profile information, e.g. temperature and density.

                                                                # There is more discussion of Npitch, Nphi, and Ngryo below
    Npitch                 = settings["Npitch"]                 # Number of different pitch angles at each rectangular [r,z] bin
    Nphi                   = settings["Nphi"]                   # Number of different toroidal angles
    Ngyro                  = settings["Ngyro"]                  # Number of gyro angles
    Nrho_profiles          = settings["Nrho_profiles"]          # number of points in kinetic radial profiles
    
    Nmrk_original  = Nmrk

    # +++++++++++++++++++++++++++++++++++++++++++
    #   optionally, get full 3D magnetic field
    #   from a single file


        
    # +++++++++++++++++++++++++++++++++++++++++++
    #  get equilibrium
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    geq_strings  = fn_geqdsk.split('.')
    stub         = geq_strings[0] + '_'
    geq_filename = stub + 'equilibrium.pdf'
    
    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    gg.plotEquilibrium(eq_index)
    plt.savefig(geq_filename)
 
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #    generate the spatial grids and pitch grid

    #  first, we read (from the equlibrium file) the grid on which
    #  the equilibrium itself is defined)
    
    Rmin       =  gg.equilibria[eq_index].rmin
    Rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    Zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    Zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    rlcfs      =  gg.equilibria[eq_index].rlcfs
    zlcfs      =  gg.equilibria[eq_index].zlcfs
    psi_nr     =  gg.equilibria[eq_index].nw
    psi_nz     =  gg.equilibria[eq_index].nh
    psiPolSqrt =  gg.equilibria[eq_index].psiPolSqrt

    #  rhogeq_transpose_2d is an array on [R,Z] on which the equilibrium is
    #  defined.  Using this array, we can compute BZ and BR (as generated by
    #  the plasma itself at any [R,Z] point in the plasma.
    
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)         # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))
    geq_rarray          = np.linspace(Rmin, Rmax, psi_nr)
    geq_zarray          = np.linspace(Zmin, Zmax, psi_nz)
    
    # f = (Zmax-Zmin)/(Rmax-Rmin) 

    #  
 
    #  The logic below that computes the number of grid points in the R and Z directions might
    #  seem (and might in fact be) a little confusing and could be sub-optimal.
    #
    #  One might think that a simple approach would be to ask the user to define (through the
    #  "settings" dictionary) the number of grid points in the Rmajor direction (NR), the
    #  number in the Z-direction (NZ), the number in pitch-angle space (Npitch), the number
    #  in the toroidal direction (Nphi), and the number in gyro-direction (Ngryo), and
    #  then let the code compute the total number of markers Nmrk = NR * NZ * Npitch * Nphi * Ngyro.
    #
    #  We do want to ensure that the chosen step sizes in the major-radius direction and the
    #  vertical direction are approximately the same.

    #  Because the number of markers is of considerable interest to an orbit simulation -- the
    #  CPU time is proportional to the number of markers - I chose instead to let the user
    #  define the total number of markers Nmrk, along with Npitch, Nphi, and Ngyro, and then
    #  the logic below then computes the NR and NZ to be consistent with those values.

    #  In hindsight, this really doesn't releave the user from having to carefully consider
    #  the chosen values of Nmrk, Npitch, Nphi, and Ngyro, because if you choose a value
    #  for Nmrk that isn't big enough, then the grid resolution in the R and Z directions
    #  will be too coarse.

    #  see test_rz.py which confirms that the arithmetic below yields equal grid
    #  step sizes in the Rmajor and Z directions, and approximately conserves
    #  number of markers

    
    NRZ        = int(Nmrk/(Npitch*Nphi*Ngyro))                # total number of R-z grid points
    NR         = int( np.sqrt(NRZ*(Rmax-Rmin)/(Zmax-Zmin)) )  # ensures approx. equal step sizes in R & Z dirn's
    NZ         = int( np.sqrt(NRZ*(Zmax-Zmin)/(Rmax-Rmin)) )  # ensures approx. equal step sizes in R 7 Z dirn's

    if(rmajor_min !=0.):
        Rmin_grid = rmajor_min
    else:
        Rmin_grid = Rmin
    if(rmajor_max !=0.):
        Rmax_grid = rmajor_max
    else:
        Rmax_grid = Rmax

    if(z_min !=0.):
        zmin_grid = z_min
    else:
        zmin_grid = Zmin
    if(rmajor_max !=0.):
        zmax_grid = z_max
    else:
        zmax_grid = Zmax

    print("   ... marker grid:  Rmin, Rmax, Zmin, Zmax =  %7.4f %7.4f %7.4f %7.4f"%(Rmin_grid, Rmax_grid,zmin_grid, zmax_grid))      
    R_array    = np.linspace(Rmin_grid, Rmax_grid, NR)
    Z_array    = np.linspace(zmin_grid, zmax_grid, NZ)
    
    deltaR_bin = R_array[1] - R_array[0]
    deltaZ_bin = Z_array[1] - Z_array[0]

    if(Nphi == 1):
       phi_array = Phi_min * np.ones(1)
    else:
       deltaPhi = (Phi_max-Phi_min)/(Nphi-1)
       phi_array = Phi_min + deltaPhi*np.linspace(0, Nphi, Nphi,endpoint=False)

    if(Ngyro==1):
        gyro_array = np.ones(1) * gyro_angle_fixed
    else:
        gyro_array = np.linspace(0., 2.*np.pi, Ngyro, endpoint=False)
        
    if(Npitch==1):
         pitch_array = Pitch_min * np.ones(Nmrk)
         delta_pitch = 0.
    else:
         delta_pitch = (Pitch_max-Pitch_min)/(Npitch-1)
         pitch_array = Pitch_min + delta_pitch * np.linspace(0, Npitch, Npitch, endpoint=False)
         delta_pitch = pitch_array[1] - pitch_array[0]

    
    Nmrk_big = NR * NZ * Npitch * Nphi * Ngyro
    
    print("   ... marker_sets.py/define_prt_markers_uniform:  Nmrk_big = ", Nmrk_big)
    print("   ... marker_sets.py/define_prt_markers_uniform:  deltaR, deltaZ = ", deltaR_bin, deltaZ_bin)
    print("       these should be nearly equal")

    marker_Rs      = np.zeros(Nmrk_big)
    marker_Zs      = np.zeros(Nmrk_big)
    marker_pitches = np.zeros(Nmrk_big)
    marker_phis    = np.zeros(Nmrk_big)
    marker_gyros   = np.zeros(Nmrk_big)

    # +++++++++++++++++++++++++++++++++++++++++++
    #  construct ensemble of uniformily-spaced
    #  markers, then decide which markers lie
    #  inside the plasma
             
    ictr = 0
    
    for ir in range(NR):
        for jz in range(NZ):
           for kphi in range(Nphi):
              for kpitch in range(Npitch):
                  for mgyro in range(Ngyro):

                      marker_Rs[ictr]      = R_array[ir]
                      marker_Zs[ictr]      = Z_array[jz]
                      marker_pitches[ictr] = pitch_array[kpitch]
                      marker_phis[ictr]    = phi_array[kphi]
                      marker_gyros[ictr]   = gyro_array[mgyro]
                      
                      ictr += 1

    # rho_interolator is a method that computes the "rho" coorindates (0 at
    # the magnetic axis, 1 at the plasma edge) for a given set of R- and
    # Z-locations of the birth-marker positions

    # note that rho_interpolator works for scalar values of R and Z and
    # also for arrays of R and Z
    
    marker_Rhos =  rho_interpolator(marker_Rs, marker_Zs, grid=False)
    
    #  ii_good[ii] is True if the ii_th marker lies inside the plasma
    ii_good  =   (marker_Rhos <=  birth_rhomax)    \
               & (marker_Rhos >=  birth_rhomin)    \
               & (marker_Zs   <=  np.max(zlcfs))   \
               & (marker_Zs   >=  np.min(zlcfs))
    #pdb.set_trace()
    xx = 2.
    Nmrk = marker_Rhos[ii_good].size

    print("   ... number of candidate markers, number of qualified markers: ", Nmrk_big, Nmrk)
    
    # ++++++++++++++++++++++++++++++++++++++++
    # get profile data

    aa_profiles   = proc.read_sparc_profiles_new(fn_profiles, Nrho_profiles)

    alpha_source  = aa_profiles["alpha_source"]/1.e18
    rho_array     = aa_profiles["rhosqrt"]
     
    # +++++++++++++++++++++++++++++++++++++++++++++++
    #    now generate the qualified markers

    # create arrays to write into hdf5 file
                  
    R           =      marker_Rs[ii_good]       
    z           =      marker_Zs[ii_good]
    rhos        =    marker_Rhos[ii_good]
    phi         =    marker_phis[ii_good]
    pitches     = marker_pitches[ii_good]
    gyro_angles =   marker_gyros[ii_good]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    #  arithmetic to get velocity for relativistic electrons  3/27/23
    #
    #   relativistic electron mass = energy-in-joules / c**2
    #   relativistic elecron mass = m_e / ((1-v^2/c^2)**0.5)    # m_e = electron mass as rest
    #
    #    so 1-v^2/c^2 = (m_e / relativistic elecron mass)**2
    #
    #  so v = c * (1 - (m_e/relativistic-electron-mass)**2 )**0.5
    
    if(set == 1):    #  the usual  
        
       vtot       = np.sqrt(2. * electron_charge * Ekev * 1000. / AMU_MASS *PROTON_MASS)    # m/s
       
    elif(set==2):    # relatavistic electrons
        
       E_joules                   = Ekev * 1000. * ELECTRON_CHARGE
       mass_electron_relativistic = E_joules / (LIGHT_SPEED**2)
       vtot                       = LIGHT_SPEED * np.sqrt( 1. - (ELECTRON_MASS/mass_electron_relativistic)**2)
       
       print("  define_prt_markers_uniform:  vtot, c, vtot/c: ", vtot/1.e8, LIGHT_SPEED/1.e8, vtot/LIGHT_SPEED)
       
        
    vtots      = vtot       * np.ones(Nmrk)
    
    #vphi       = vtots      * np.cos(pitches)
    #vpoloidals = vtots      * np.sqrt(1-pitches*pitches)
    #vR         = vpoloidals * np.cos(gyro_angles)
    #vz         = vpoloidals * np.sin(gyro_angles)

    vphi      = np.zeros(vtots.size)
    vR        = np.zeros(vtots.size)
    vz        = np.zeros(vtots.size)
    vparallel = np.zeros(vtots.size)   # not given to ASCOT, but overplotted for validation
    vperp     = np.zeros(vtots.size)   # not given to ASCOT, but overplotted for validation
    
    if(bfield_single_file):
        
        BB_3D = proc.read_sparc_bfield_3d(bfield_single_filename)

        bhats = compute_bhats(R, phi, z, BB_3D)

    else:

         #  compute the local magnetic field from the equilibrium
        
         RR_bgrid, ZZ_bgrid, BTor_grid, BR_grid, BZ_grid, BTot_grid, Psi_dummy = WG.wrapper_geqdskfile(fn_geqdsk)

         if( btor_multiplier !=1.0):
             print("   ... marker_sets/define_prt_markers_uniform:  I will multiply Btor by factor: ", btor_multiplier)
             BTor_grid = BTor_grid * btor_multiplier

         # 'bhat' is a *unit* vector that is pointed in the direction of the local B-field direction.
         #  bhats is an array of bhat-vectors, one for each marker.
         #
         #  what the following few lines of code do is:  compute the local B-field components in
         #  the phi, Rmajor, and vertical direction.  Then combine them so that we have a Nx3 vector
         #  rather than 3 individual N-length vectors.
         #
         #  The 'vector_hat' module then divides the local B-field vector by the size of the B-field
         #  vector, thereby yielding 'bhat'
         
         bhats             = np.zeros((R.size,3))

         this_btor = RectBivariateSpline(RR_bgrid, ZZ_bgrid, BTor_grid)(R, z, grid=False)
         this_bz   = RectBivariateSpline(RR_bgrid, ZZ_bgrid, BZ_grid)(R,   z, grid=False)
         this_br   = RectBivariateSpline(RR_bgrid, ZZ_bgrid, BR_grid)(R,   z, grid=False)
         this_btot = np.sqrt( this_btor**2 + this_bz**2 + this_br**2)
         
         Bfield_vectors_3D = np.vstack((this_br,this_btor, this_bz)).T
         
         for jgg in range(R.size):
             bhat             = VA.vector_hat(Bfield_vectors_3D[jgg,:])
             bhats[jgg,:]     = bhat

         field_line_vertical_angles   = (180./np.pi)*np.arctan(this_bz/this_btot)
         field_line_horizontal_angles = (180./np.pi)*np.arctan(this_br/this_btot)
         
    for jj in range(vtots.size):

        #  If we know the total particle velocity ('vtots'), its pitch angle respect to the magnetic field ('pitches') and
        #  the gyro-angle (gyro_angles[jj]), and the 3D direction of the magnetic field (bhats'), we can uniquely
        #  compute the direction of the particle in Rmajor, phi, Z coordinate directions
        
        velocity_vector_R_phi_Z = CVV.construct_velocity_vector(bhats[jj], vtots[jj], pitches[jj], gyro_angles[jj])

        vR[jj]        = velocity_vector_R_phi_Z[0]
        vphi[jj]      = velocity_vector_R_phi_Z[1]
        vz[jj]        = velocity_vector_R_phi_Z[2]

        vparallel[jj] = pitches[jj] * vtots[jj]
        vperp[jj]     = np.sqrt(vtots[jj]**2 - vparallel[jj]**2)
        
    # +++++++++++++++++++++++++++++++++++++++++
    #   marker weighting
    #
    #   proportional to alpha source
    #   proportional to Rmajor (because volume element is proportional to Rmajor
    #   proportional to sqrt(1-pitch**2) because velocity volume element propto sin(pitch)
    #   ... we need to verify the pitch dependence of weighting
    #
    #  For Hongyu's code, we would need to add a weighting factor due to "central", i.e.
    #  if we artifically construct an extra factor of five more markers for a given [R,Z] rectangle
    #  near the magnetic axis, then we need to given them only 20% of their otherwise-computed
    #  weight.

    if (set == 1):
        
        weights_alpha_source = np.interp(rhos, rho_array, alpha_source)
        weights_pitch        = np.sqrt(1. - pitches*pitches)
        weight               = R * weights_alpha_source * weights_pitch
        weight               = weight / np.total(weight)
        
    elif (set == 2):    # for runaway electrons:  weight only by Rmajor
        
        weight = R
        weight = weight/np.sum(weight)
        
    ids    = np.linspace(1,Nmrk,Nmrk).astype(int)    # astype(int) added 4/27/23 9:09 AM

    if(set == 1):
        
        mass   = mass_amu * np.ones(ids.shape)
        charge = charge   * np.ones(ids.shape)
        anum   = mass_amu * np.ones(ids.shape)
        znum   = charge   * np.ones(ids.shape)
        time   = 0.  * np.ones(ids.shape)

    elif (set == 2):

        anum   =  1.               * np.ones(ids.shape)
        znum   =  1.               * np.ones(ids.shape)
        charge = -1.               * np.ones(ids.shape)
        mass   = ELECTRON_MASS_AMU * np.ones(ids.shape)
        time   = 0.                * np.ones(ids.shape)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   to verify that the mapping from vtot, pitch, gyro to
    #   vr, vphi, vz is correct, reconstruct the vparallel and
    #   verp using the 3D velocity components and the 3D
    #   B-field geometry

    vtot_reconstructed      = np.zeros(Nmrk)
    vparallel_reconstructed = np.zeros(Nmrk)
    vperp_reconstructed     = np.zeros(Nmrk)
    pitch_reconstructed     = np.zeros(Nmrk)

    zhat = np.array((0,0.,1.))     # unit vector in z direction
    
    for jk in range(Nmrk):

        #  the coordinates [hhat, khat] form the space perpendicular to B
        
        bhat                   = bhats[jk]                       # unit vector in direction of B
        hhat                   = VA.cross_product(bhat, zhat)    # unit vector perp to B
        khat                   = VA.cross_product(hhat, bhat)    # unit vector perp to B and hhat

        hhat = VA.vector_hat(hhat)   # we really do want UNIT vectors
        khat = VA.vector_hat(khat)

        velocity_vector             = np.array((vR[jk], vphi[jk], vz[jk]))
        vtot_reconstructed[jk]      = np.sqrt(np.sum(velocity_vector**2))
        vparallel_reconstructed[jk] = VA.dot_product(bhat, velocity_vector)
        pitch_reconstructed[jk]     = vparallel_reconstructed[jk]/vtot_reconstructed[jk]
        vperp_reconstructed[jk]     = np.sqrt(   (VA.dot_product(velocity_vector, hhat)) **2 \
                                               + (VA.dot_product(velocity_vector, khat))**2    )

    print(" +++++++++++++++++++++++++++++++++++++++++++++++ ")
    print("\n  marker_sets.py/define_prt_markers_uniform \n")
    print("    Nmrk_original:   %d"%(Nmrk_original))
    print("    Nmrk (final)     %d"%(Nmrk))
    print("    Npitch:          %d"%(Npitch))
    print("    Ngyro            %d"%(Ngyro))
    print("    NR:              %d"%(NR))
    print("    NZ:              %d"%(NZ))
    print("    Nphi:            %d"%(Nphi))
    print("    delta_R_bin    %7.4f"%(deltaR_bin))
    print("    delta_Z_bin    %7.4f"%(deltaZ_bin))
    print("    delta_pitch    %7.4f"%(delta_pitch))
    print("    Rmin           %7.4f"%(np.min(R_array)))
    print("    Rmax           %7.4f"%(np.max(R_array)))
    print("    Zmin           %7.4f"%(np.min(Z_array)))
    print("    Zmax           %7.4f"%(np.max(Z_array)))
    print("    pitch_min      %7.4f"%(np.min(pitch_array)))
    print("    pitch_max      %7.4f"%(np.max(pitch_array)))
    print("    vtot           %11.3e"%(vtot))
    print("    Rlcfs min      %7.4f"%(np.min(rlcfs)))
    print("    Rlcfs max      %7.4f"%(np.max(rlcfs)))
    print("    Zlcfs min      %7.4f"%(np.min(zlcfs)))
    print("    Zlcfs min      %7.4f"%(np.max(zlcfs)))

    print("   time to create markers (sec)   %8.1f"%(clock.time()-time_before))
    #pdb.set_trace(header="inside uniform")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
    print("   ... now writing marker information to .h5 file. \n")
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc="None")

    print("   ... starting common plots")
    stub_hdf5       = fn_hdf5.split(".")[0]
    filename_inputs = stub_hdf5 + '_marker_inputs.pdf'
    print("   ... marker_sets.py/define_prt_markers_uniform: starting plots in file: ", filename_inputs,"\n")

    my_graph_label = "define_prt_markers_uniform/"+filename_inputs
    nn = Nmrk
    try:
        if(settings['nplot_max']):
            nn = np.min((settings["nplot_max"], Nmrk))
    except:
        xx = 1.  # dummy
    
    with PdfPages(filename_inputs) as pdf:

        jjj_inside  = (np.abs(R-1.3) <0.02)
        jjj_middle  = (np.abs(R-1.88)<0.02)
        jjj_outside = (np.abs(R-2.24)<0.02)
        
        my_labels=[]
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        this_label, = plt.plot(z[jjj_inside],  field_line_vertical_angles[jjj_inside], 'ro', ms=2, rasterized=True,   label="R near 1.30")
        my_labels.append(this_label)
        this_label, = plt.plot(z[jjj_middle],  field_line_vertical_angles[jjj_middle], 'go', ms=2, rasterized=True,   label="R near 1.88")
        my_labels.append(this_label)
        this_label, = plt.plot(z[jjj_outside],  field_line_vertical_angles[jjj_outside], 'bo', ms=2, rasterized=True, label="R near 2.24")
        my_labels.append(this_label)
        plt.title('Field-line vertical angles')
        plt.xlabel('Z [m]')
        plt.ylabel('[degrees]')
        plt.tight_layout(pad=2)
        plt.legend(handles=my_labels, loc='upper right', fontsize=10)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        jjj_low2    = (np.abs(z+0.80) <0.02)
        jjj_low     = (np.abs(z+0.40) <0.02)
        jjj_middle  = (np.abs(z)      <0.02)
        jjj_high    = (np.abs(z-0.40) <0.02)
        jjj_high2   = (np.abs(z-0.80) <0.02)
        
        my_labels=[]
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        this_label, = plt.plot(R[jjj_low2],  field_line_vertical_angles[jjj_low2], 'o', color='orange', ms=4, rasterized=True,   label="Z near -0.80")
        my_labels.append(this_label)                      
        this_label, = plt.plot(R[jjj_low],  field_line_vertical_angles[jjj_low], 'co', ms=4, rasterized=True,   label="Z near -0.40")
        my_labels.append(this_label)
        this_label, = plt.plot(R[jjj_middle],  field_line_vertical_angles[jjj_middle], 'ko', ms=2, rasterized=True,   label="Z near 0")
        my_labels.append(this_label)
        this_label, = plt.plot(R[jjj_high],  field_line_vertical_angles[jjj_high], 'bo', ms=2, rasterized=True, label="Z near +0.40")
        my_labels.append(this_label)
        this_label, = plt.plot(R[jjj_high2],  field_line_vertical_angles[jjj_high2], 'ro', ms=2, rasterized=True, label="Z near +0.80")
        my_labels.append(this_label)
                       
        plt.title('Field-line vertical angles')
        plt.xlabel('Rmajor [m]')
        plt.ylabel('[degrees]')
        plt.tight_layout(pad=2)
        plt.legend(handles=my_labels, loc='upper right', fontsize=10)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()
        
        
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(pitches[0:nn], gyro_angles[0:nn], 'bo', ms=3, rasterized=True)
        plt.title('Original pitch and gyro_angles')
        plt.xlabel('original pitch')
        plt.ylabel('original gyro_angle')
        plt.tight_layout(pad=2)
        plt.xlim((-1.1,1.1))
        plt.ylim((-0.1, 2.*np.pi))
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()
    
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(pitches[0:nn], pitch_reconstructed[0:nn], 'bo', ms=3, rasterized=True)
        plt.title('Original and reconstructed pitch')
        plt.xlabel('original')
        plt.ylabel('reconstructed')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(pitches[0:nn], pitch_reconstructed[0:nn], 'bo', ms=3, rasterized=True)
        plt.title('Original and reconstructed pitch')
        plt.xlabel('original')
        plt.ylabel('reconstructed')
        plt.xlim((-1.1,1.1))
        plt.ylim((-1.1,1.1))
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(pitches[0:nn], pitch_reconstructed[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed pitch (limited range)')
        plt.xlabel('original')
        plt.ylabel('reconstructed')
        plt.xlim((-0.0005,0.0005))
        plt.ylim((-0.0005,0.0005))
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

            
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vtots[0:nn], vtot_reconstructed[0:nn]/vtots[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed/original vtot')
        plt.xlabel('original')
        plt.ylabel('reconstructed')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vparallel[0:nn], vparallel_reconstructed[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed vparallel')
        plt.xlabel('original')
        plt.ylabel('ratio')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vparallel[0:nn], vparallel_reconstructed[0:nn]/vparallel[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed/original vparallel')
        plt.xlabel('original')
        plt.ylabel('ratio')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vparallel[0:nn], vparallel_reconstructed[0:nn]/vparallel[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed/original vparallel (limited range)')
        plt.xlabel('original')
        plt.ylabel('ratio')
        plt.tight_layout(pad=2)
        plt.ylim((0.995,1.005))
        sds.graph_label(my_graph_label)
        pdf.savefig()
  

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vperp[0:nn], vperp_reconstructed[0:nn]/vperp[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed/original vperp')
        plt.xlabel('original')
        plt.ylabel('ratio')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vperp[0:nn], vperp_reconstructed[0:nn]/vperp[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('Original and reconstructed/original vperp (limited range)')
        plt.xlabel('original')
        plt.ylabel('ratio')
        plt.tight_layout(pad=2)
        plt.ylim((0.995, 1.005))
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vphi[0:nn], vR[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('original velocity components')
        plt.xlabel('v_phi')
        plt.ylabel('v_R')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vphi[0:nn], vz[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('original velocity components')
        plt.xlabel('v_phi')
        plt.ylabel('v_z')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vR[0:nn], vz[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('original velocity components')
        plt.xlabel('v_R')
        plt.ylabel('v_z')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(vparallel_reconstructed[0:nn], vperp_reconstructed[0:nn], 'bo', ms=2, rasterized=True)
        plt.title('reconstructed velocity components')
        plt.xlabel('v_parallel')
        plt.ylabel('v_perp')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
   
        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(R[0:nn], weight[0:nn], 'ro', ms=1, rasterized=True)
        #plt.xlim([0.,1.])
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('Rmajor [m]')
        plt.ylim(bottom=0.)
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()
        plt.close(fig=None)  # first one is corrupted
        plt.clf()

        plt.close(fig=None)
        plt.clf()
        plt.figure(figsize=(8.,6.))
        plt.plot(z[0:nn], weight[0:nn], 'ro', ms=1, rasterized=True)
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('Z [m]')
        plt.ylim(bottom=0.)
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close(fig=None) 
        plt.clf()
    
        plt.plot(rhos[0:nn], weight[0:nn], 'ro', ms=1, rasterized=1)
        xx_fake = np.linspace(0.,1., 100)
        yy_fake = np.zeros(100)
        plt.plot(xx_fake, yy_fake, 'k-')
        plt.xlim([0.,1.])
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('rho [sqrt(norm poloidal flux)]')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()
        plt.close(fig=None)
        plt.clf()

        # ------------------------------------------------
        #  histogram of rmajor

        mmm = (np.abs(z[0:nn]) < 0.1)
        plt.figure(figsize=(7.,5.))
        plt.hist(R[0:nn][mmm], bins=50, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker Rmajor (abs(z)<0.1)')
        plt.xlabel('Rmajor')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()

        # ------------------------------------------------
        #  histogram of z

        plt.figure(figsize=(7.,5.))
        plt.hist(z[0:nn], bins=100, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker Z')
        plt.xlabel('Z')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()
    
        # ------------------------------------------------
        #  histogram of pitch angles

        plt.figure(figsize=(7.,5.))
        plt.hist(pitches[0:nn], bins=100, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker pitch angles')
        plt.xlabel('pitch angle')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf() 
                
        # -------------------------------------------------------------
        #  velocity vector directions

        #fig=plt.figure(figsize=(6., 6.))
        #vr_direction = vR[0:nn]/v_poloidal[0:nn]
        #vz_direction = vz[0:nn]/v_poloidal[0:nn] 
        #plt.plot(vr_direction, vz_direction, 'ro', ms=1,rasterized=1)
        #plt.xlim([-1.1,1.1])
        #plt.ylim([-1.1,1.1])
        #plt.xlabel('vR/v_poloidal')
        #plt.ylabel('vz/v_poloidal')
        #plt.title(' poloidal velocity direction')
        #plt.tight_layout()
        #plt.savefig(stub + '_marker_vpoloidal_dirn.pdf')
        #sds.graph_label(my_graph_label)
        #pdf.savefig()
        #plt.close()
        #plt.clf() 
    
        # -------------------------------------------------------------
        
        z_over_r = (Zmax - Zmin) / (Rmax - Rmin)
        xsize = 5.
        ysize = xsize * z_over_r
        plt.axis([Rmin, Rmax, Zmin, Zmax])
        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=True)
        plt.plot(R[0:nn],z[0:nn], 'ro', ms=0.4,zorder=2, rasterized=True)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        rho_contours_2 = [1.02, 1.04]

        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='g',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro', ms=3., fillstyle='none',zorder=2, rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
       #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((1.4,1.6))
        plt.ylim((0.90, 1.15))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro', zorder=2, ms=3, fillstyle='none', rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((2.4,2.6))
        plt.ylim((-0.1,0.1))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro',  ms=3, fillstyle='none',zorder=2,rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((2.28,2.40))
        plt.ylim((0.35, 0.45))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()
    
        z_over_r = (Zmax - Zmin) / (Rmax - Rmin)
        xsize = 5.
        ysize = xsize * z_over_r
        fig=plt.figure(figsize=(xsize, ysize))
        plt.axis([Rmin, Rmax, Zmin, Zmax])
        #plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=2, rasterized=1)
    
        rho_contours = np.linspace(0.1, 1.0, 10)
        #rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=1.0, colors='g',zorder=1)
        plt.clabel(cs)

        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(' contours of rho-pol')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.close(fig=None)
        plt.plot(R[0:nn], pitches[0:nn], 'ro', ms=2,rasterized=True)
        plt.xlabel('Rmajor [m]')
        plt.title('pitch of markers')
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()


        plt.plot(R[0:nn], pitches[0:nn], 'ro', ms=2,rasterized=True)
        plt.xlabel('Rmajor [m]')
        plt.title('pitch of markers')
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.close(fig=None)
        plt.plot(z[0:nn], pitches[0:nn], 'ro', ms=2, rasterized=True)
        plt.xlabel('elevation [m]')
        plt.title('pitch of markers')
        sds.graph_label(my_graph_label)
        plt.tight_layout(pad=2)
        pdf.savefig()
        plt.close()
        plt.clf()


def define_prt_markers(fn_hdf5, set, Nmrk, settings, desc=None, fn_geqdsk=None, eq_index=0):
    """
    define_prt_markers: 
       hdf5_fn     name of hdf5 file
       set         set number = 1, 2, 3, ...
       Nmrk        number of markers
       desc        description
    """


    #if (set < 1) or (set > 2):
    #   print("define_prt_markers:  set must be one or two")
    #    return None
      
    if set == 10:
        
        # space the birth positions out radially, vertically, and in pitch angle
        # 11/21/21

        NR          = settings["NR"]
        NZ          = settings["NZ"]
        NPITCH      = settings["NPITCH"]
        rstart      = settings["rstart"]
        zstart      = settings["zstart"]
        pitch_start = settings["pitch_start"]
        rend        = settings["rend"]
        zend        = settings["zend"]
        pitch_end   = settings["pitch_end"]
        
        birth_rhomax = 0.99
        
        Nmrk   = NR * NZ * NPITCH

        gyro_angles =  np.zeros(Nmrk)    # or (2. * np.pi) * np.random.rand(Nmrk)
         
        ids    =         np.linspace(1,Nmrk,Nmrk)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)

        vphi      = np.zeros(Nmrk)
        vperp     = np.zeros(Nmrk)
        vR        = np.zeros(Nmrk)
        vz        = np.zeros(Nmrk)
        rho_birth = np.zeros(Nmrk)
        R         = np.zeros(Nmrk)
        z         = np.zeros(Nmrk)
        pitch     = np.zeros(Nmrk)
        
        clight    = 2.998e8
        mproton   = 1.673e-27
        qelectron = 1.602e-19    # also Joules per eV

        v   = np.sqrt(2. * qelectron * energy / mass)  # array of vtots

        ii = 0   # a counter

        for jj in range(NR):
            for kk in range(NZ):
                for mm in range(NPITCH):

                    R[ii]     = rstart + (rend - rstart) * jj/float(NR-1)
                    z[ii]     = zstart + (zend - zstart) * kk/float(NZ-1)
                    
                    pitch[ii] = pitch_start + (pitch_end - pitch_start) * mm / float(NPITCH-1)
                    
                    vphi[ii]  =     v[ii] * pitch[ii]
                    vperp[ii] =     v[ii] * np.sqrt(1. - pitch[ii]**2)
                    vR[ii]    = vperp[ii] * np.cos(gyro_angles[ii])
                    vz[ii]    = vperp[ii] * np.sin(gyro_angles[ii])

                    ii += 1
                    
        print("   ... number of candidate markers: ", R.size)

        # ----------------------------------------------------------------------
        #  get equilibrium so we can compute rho_poloidal of candidate markers
       
        gg = geqdsk(fn_geqdsk)
        gg.readGfile()

        rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index) 
         
        psi_rmin       =  gg.equilibria[eq_index].rmin
        psi_rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
        psi_nr         =  gg.equilibria[eq_index].nw
        psi_zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
        psi_zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
        psi_nz         =  gg.equilibria[eq_index].nh
        
        psiPolSqrt     =  gg.equilibria[eq_index].psiPolSqrt

        geq_rarray = np.linspace(psi_rmin, psi_rmax, psi_nr)
        geq_zarray = np.linspace(psi_zmin, psi_zmax, psi_nz)
         
        rhogeq_transpose_2d = np.transpose(psiPolSqrt) 

        rhogeq_interp = interpolate.interp2d(geq_rarray, geq_zarray, psiPolSqrt, kind='cubic')
    
        # -------------------------------------------
        #  eliminate markers whose rho > birth_rhomax

        for qq in range(Nmrk):
            rho_birth[qq]  = rhogeq_interp(R[qq], z[qq])     # compute rho_poloidal
        
        ii_good = (rho_birth <= birth_rhomax)

        ids       =       ids[ii_good]     
        mass      =      mass[ii_good]    
        charge    =    charge[ii_good]    
        anum      =      anum[ii_good]    
        znum      =      znum[ii_good]    
        phi       =       phi[ii_good]    
        weight    =    weight[ii_good]    
        time      =      time[ii_good]    
        energy    =    energy[ii_good]
        v         =         v[ii_good]
        vphi      =      vphi[ii_good]    
        vperp     =     vperp[ii_good]    
        vR        =        vR[ii_good]    
        vz        =        vz[ii_good]    
        rho_birth = rho_birth[ii_good]
        R         =         R[ii_good]
        z         =         z[ii_good]

        Nmrk = ids.size
        
        print("   ... final number of markers (satisfies rhobirth): ", R.size)
        
    elif set==1:
        
        # space the birth positions out radially
        
        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + aa/(2.*(Nmrk+1))    # e.g. rho = 0.1
        Rend   = R0 + aa - aa/(2.*(Nmrk+1))    # e.g. rho = 0.9
        R      = np.linspace(Rstart, Rend, Nmrk)
        print("Rstart, Rend: ", Rstart, Rend)
        print("Rarray: ", R)
        z = np.zeros((Nmrk))

        ids    =         np.linspace(1,Nmrk,Nmrk)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        pitch  = 0.20  * np.ones(ids.shape)   # 0.35
        
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?

    elif set==3:
        
        # four particles, rho = 0.2 and rho = 0.8 and pitch = 0.2 and 0.8
        
        assert Nmrk == 4
        
        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + 0.2*aa   
        Rend   = R0 + 0.8*aa
        R      = np.array([Rstart, Rstart, Rend, Rend])
        pitch  = np.array([0.2, 0.8, 0.2, 0.8])
        
        z = np.zeros((Nmrk))

        ids    =         np.linspace(1,Nmrk,Nmrk)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        
        
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?

    elif set==4:
        
        # Nmrk particles, rho = 0.2 thru rho = 0.8 and pitch = 0.2 
       
        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + 0.2*aa   
        Rend   = R0 + 0.8*aa
        R      = np.array(np.linspace(Rstart, Rend, Nmrk))

        z = np.zeros((Nmrk))

        ids    =         np.linspace(1,Nmrk,Nmrk)
        pitch  = 0.2   * np.ones(ids.shape)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        
        
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?                 

    elif set==5:
        
        # Nmrk particles, rho = 0.3 thru rho = 0.9 and pitch = 0.2 
       
        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + 0.3*aa   
        Rend   = R0 + 0.9*aa
        R      = np.array(np.linspace(Rstart, Rend, Nmrk))

        z = np.zeros((Nmrk))

        ids    =         np.linspace(1,Nmrk,Nmrk)
        pitch  = 0.2   * np.ones(ids.shape)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        
        
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?
        
    elif set == 2:
        
        rmajor_birth = 1.65
        rminor_birth = 0.49
        kappa_birth  = 1.79
        nrho_birth   = 30     # number of radial grid points for alpha birth
        theta        = 2*np.pi*np.random.rand(Nmrk)

        rho_birth  = np.linspace(0.,1., nrho_birth, endpoint=False)
        prho       = rho_birth * ( 1.- rho_birth)**2
        prho_sum   = np.sum(prho)

        birth_rho_numbers = Nmrk * prho / prho_sum
        birth_rho_numbers = birth_rho_numbers.astype(int)

        print('Initial {0:6d} markers.  Corrected: {1:6d}'.format(Nmrk,np.sum(birth_rho_numbers)))

        birth_total = np.sum(birth_rho_numbers)

        R = np.zeros((Nmrk))
        z = np.zeros((Nmrk))

        ctr = 0
    
        for ir in range(nrho_birth):
            for iparticle in range(birth_rho_numbers[ir]):
                R[ctr] = rmajor_birth + rminor_birth  * rho_birth[ir] * np.cos(theta[ctr])
                z[ctr] = rminor_birth * kappa_birth   * rho_birth[ir] * np.sin(theta[ctr])
                ctr += 1
    
        # but we might have a few missing.  recompute number of markers
    
        Nmrk = np.sum(birth_rho_numbers)
        R = R[0:Nmrk]
        z = z[0:Nmrk]

        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 4*np.ones(ids.shape)
        charge = 2*np.ones(ids.shape)
        anum   = 4*np.ones(ids.shape)
        znum   = 2*np.ones(ids.shape)
        phi    = 360*rand(ids.shape)
        weight = 1*np.ones(ids.shape)
        time   = 0*np.ones(ids.shape)
        energy = 3.5e6*np.ones(ids.shape)
        pitch  = 0.999-1.999*np.random.rand(Nmrk)
        theta  = 2*np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?

    elif set == 6:

        # sds 9/21/2019
        
        #  position markers randomly in space and pitch angle.
        #  weight the marker proportional to local volumetric alpha
        #  birth rate from V0 multiplied by rho (to compensate
        #  for volume).  see alpha_birth_approx.pdf

        #  the following optional parameters can be passed in
        #  via the "settings" dictionary:
        #
        #     sparc_version
        #     total_weight
        #     birth_rhomax

        mm = 100                # grid for rr, zz
        tolerance = 0.02
        
        if settings["sparc_version"] == 0:
            
            rmajor_birth = 1.65
            rminor_birth = 0.49
            kappa_birth  = 1.79
            delta_birth  = 0.45
            
        elif settings["sparc_version"] == 1:
            
            rmajor_birth = 1.78
            rminor_birth = 0.55
            kappa_birth  = 1.75
            delta_birth  = 0.45

        # our numbers for rmajor, rminor, kappa, and delta may not match
        # the actual equilibrium exactly.  To be safe, I don't want to launch
        # markers outside the plasma ... what might happen then?  So
        # for the time being, limit the range of rho over which markers
        # will be launched
        
        birth_rhomax = 0.98
        total_weight = 100.

        if settings["total_weight"]:
            total_weight = settings["total_weight"]
            
        if settings["birth_rhomax"]:
            birth_rhomax = settings["birth_rhomax"]

        weight = np.zeros((Nmrk))

   
        Nmrk_local = 3 * Nmrk

        random_r  = np.random.rand(Nmrk_local)
        random_z  = np.random.rand(Nmrk_local)

        rr_candidate = rmajor_birth - rminor_birth +  2. * rminor_birth * random_r
        zz_candidate = rminor_birth * kappa_birth * (2. * random_z - 1.)
        
        rho_birth = np.zeros(Nmrk_local)

        print("")
        print("  marker sets:  computing rho of markers (1)")
        print("")
              
        for imark in range(Nmrk_local):
            aa = compute_rho(mm, tolerance, rmajor_birth, rminor_birth, kappa_birth, \
                             delta_birth, rr_candidate[imark], zz_candidate[imark])

            rho_birth[imark] = aa["rho"]

            #print("%5d  %7.3f  %7.3f  %8.4f" % (imark, rr_candidate[imark], zz_candidate[imark],  rho_birth[imark]))

        ii_good = (rho_birth < birth_rhomax)

        R_temp    =  rr_candidate[ii_good]
        z_temp    =  zz_candidate[ii_good]
        rhos_temp =     rho_birth[ii_good]

        good_size = R_temp.size
 
        if(good_size < Nmrk):
            print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
            sys.exit()

        R     =     R_temp[0:Nmrk]
        z     =     z_temp[0:Nmrk]
        rhos  =  rhos_temp[0:Nmrk]

        for imark in range(Nmrk):

            if rhos[imark] <= 0.5:
                weight[imark] = (total_weight/Nmrk) * rhos[imark] * (1-rhos[imark]**1.1)**3.5
            else:
                weight[imark] = (total_weight/Nmrk) * rhos[imark] * (0.003 + (1-rhos[imark])**3.2)

        # pdb.set_trace()
        
        ids    = np.linspace(1,Nmrk,Nmrk)
        
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 360   * np.random.rand(Nmrk)
        time   = 0.    * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        pitch  = 0.999-1.998*np.random.rand(Nmrk)

        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
        vR   = np.sqrt(1-pitch*pitch) * v
        vphi = pitch * v
        vz   = 0.* v           # why vz=0 identically?

    elif set == 7:

        # sds 10/8/2019

        #  parent is 7.  difference:  birth weights are
        #  consistent with v1c tf design
     
        #  position markers randomly in space and pitch angle.
        #  weight the marker proportional to local volumetric alpha
        #  birth rate from V0 multiplied by rho (to compensate
        #  for volume).  see alpha_birth_approx.pdf

        #  the following optional parameters can be passed in
        #  via the "settings" dictionary:
        #
        #     sparc_version
        #     total_weight    <--- not any more
        #     birth_rhomax
        
        mm = 100                # grid for rr, zz
        tolerance = 0.02
        
        if settings["sparc_version"] == 0:
            
            rmajor_birth = 1.65
            rminor_birth = 0.49
            kappa_birth  = 1.79
            delta_birth  = 0.45
            
        elif settings["sparc_version"] == 1:
            
            rmajor_birth = 1.78
            rminor_birth = 0.55
            kappa_birth  = 1.75
            delta_birth  = 0.45

        # our numbers for rmajor, rminor, kappa, and delta may not match
        # the actual equilibrium exactly.  To be safe, I don't want to launch
        # markers outside the plasma ... what might happen then?  So
        # for the time being, limit the range of rho over which markers
        # will be launched
        
        birth_rhomax = 0.98
        total_weight = 100.

        #if settings["total_weight"]:
        #    total_weight = settings["total_weight"]
            
        if settings["birth_rhomax"]:
            birth_rhomax = settings["birth_rhomax"]

        weight = np.zeros((Nmrk))

   
        Nmrk_local = 3 * Nmrk

        random_r  = np.random.rand(Nmrk_local)
        random_z  = np.random.rand(Nmrk_local)

        rr_candidate = rmajor_birth - rminor_birth +  2. * rminor_birth * random_r
        zz_candidate = rminor_birth * kappa_birth * (2. * random_z - 1.)
        
        rho_birth = np.zeros(Nmrk_local)

        print("")
        print("  marker sets:  computing rho of markers (2)")
        print("")
              
        for imark in range(Nmrk_local):
            aa = compute_rho(mm, tolerance, rmajor_birth, rminor_birth, kappa_birth, \
                             delta_birth, rr_candidate[imark], zz_candidate[imark])

            rho_birth[imark] = aa["rho"]

            #print("%5d  %7.3f  %7.3f  %8.4f" % (imark, rr_candidate[imark], zz_candidate[imark],  rho_birth[imark]))

        ii_good = (rho_birth < birth_rhomax)

        R_temp    =  rr_candidate[ii_good]
        z_temp    =  zz_candidate[ii_good]
        rhos_temp =     rho_birth[ii_good]

        good_size = R_temp.size
 
        if(good_size < Nmrk):
            print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
            sys.exit()

        R     =     R_temp[0:Nmrk]
        z     =     z_temp[0:Nmrk]
        rhos  =  rhos_temp[0:Nmrk]

        for imark in range(Nmrk):

            if rhos[imark] <= 0.6:
                weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
            else:
                weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)

            if (weight[imark] <= 0.):
                weight[imark] = 1.e-6

        total_weight = np.sum(weight)
        weight = weight / total_weight
        
        # pdb.set_trace()
        
        ids    = np.linspace(1,Nmrk,Nmrk)
        
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 360   * np.random.rand(Nmrk)
        time   = 0.    * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
        pitch  = 0.999-1.998*np.random.rand(Nmrk)

        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
        vR   = np.sqrt(1-pitch*pitch) * v
        vphi = pitch * v
        vz   = 0.* v           # why vz=0 identically?

        plt.close()
        plt.plot(rhos, weight, 'ro', ms=1,rasterized=1)
        plt.xlim([0.,1.])
        plt.title('Marker weights (marker set 7)')
        plt.xlabel('rho [sqrt(poloidal flux)]')
        plt.savefig('marker_weights_set_7.pdf')
        
    #print("")
    #print("=======================================================")
    #print("")
    #print("    values for marker-particle set ", set)
    #print("")
    #print("   Nmrk: {0:6d}".format(Nmrk))
    #print("   mass:   ", mass)
    #print("   charge: ", charge)
    #print("   anum:   ", anum)
    #print("   znum:   ", znum)
    #print("   phi:    ", phi)
    #print("   weight: ", weight)
    #print("   time:   ", time)
    #print("   energy: ", energy)
    #print("   pitch:  ", pitch)
    #print("   gamma:  ", gamma)
    #print("   v:      ", v)
    #print("   vR:     ", vR)
    #print("   vphi:   ", vphi)
    #print("   vz:     ", vz)
    #print("   R:      ", R)
   # print("   z:      ", z)
    #print("")

    # pdb.set_trace()
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    out={}
    out['z'] = z
    out['R'] = R 
    out['markers'] = Nmrk
    
    return out


def define_prt_markers_02(fn_hdf5, set, Nmrk, rhomin, rhomax, pitch_angle, desc=None):
    """
    define_prt_markers: 
       hdf5_fn     name of hdf5 file
       set         set number = 1, 2, 3, ...
       Nmrk        number of markers
       desc        description
    """

    if set==1:
       
        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + rhomin * aa   
        Rend   = R0 + rhomax * aa
        R      = np.array(np.linspace(Rstart, Rend, Nmrk))

        z = np.zeros((Nmrk))

        ids    =         np.linspace(1,Nmrk,Nmrk)
        pitch  = pitch_angle   * np.ones(ids.shape)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
                
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?                 

    print("")
    print("=======================================================")
    print("")
    print("    values for marker-particle set ", set)
    print("")
    print("   Nmrk: {0:6d}".format(Nmrk))
    print("   mass:   ", mass)
    print("   charge: ", charge)
    print("   anum:   ", anum)
    print("   znum:   ", znum)
    print("   phi:    ", phi)
    print("   weight: ", weight)
    print("   time:   ", time)
    print("   energy: ", energy)
    print("   pitch:  ", pitch)
    print("   theta:  ", theta)
    print("   gamma:  ", gamma)
    print("   v:      ", v)
    print("   vR:     ", vR)
    print("   vphi:   ", vphi)
    print("   vz:     ", vz)
    print("   R:      ", R)
    print("   z:      ", z)
    print("")

    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    out={}
    out['z'] = z
    out['R'] = R 
    out['markers'] = Nmrk
    
    return out


def define_prt_markers_03(fn_hdf5, fn_geqdsk, set, Nmrk, settings, nrho_in=100, desc=None):

    print(" ... just started define_prt_markers_03 \n")
    t_very_start = clock.time()
    do_rasterized=1

    # sds 10/14/2019

#  parent is marker_2, set7.  difference:  get rho
#  from actual equilibrium
     
#  position markers randomly in space and pitch angle.
#  weight the marker proportional to local volumetric alpha
#  birth rate from V0 multiplied by rho (to compensate
#  for volume).  see alpha_birth_approx.pdf

#  the following optional parameters can be passed in
#  via the "settings" dictionary:
#
#     sparc_version
#     birth_rhomax

    proton_mass     = 1.67e-27
    amu_mass        = 1.66053904e-27
    electron_charge = 1.602e-19
    
    mpl.rcParams['image.composite_image'] = False     # so we can edit the pdf files

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    geq_strings  = fn_geqdsk.split('.')
    stub         = geq_strings[0] + '_'
    geq_filename = stub + 'equilibrium.pdf'

    eq_index = settings["index"]
    #print("   ... at top of define_prt_markers_03:  I am about to call construct_rho_interpolator")
    #rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index) # 5/24/2020
    #print("   ... define_prt_markers_03 position B:  type of rho_interpolator = ", type(rho_interpolator))

    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    gg.plotEquilibrium(eq_index)
    plt.savefig(geq_filename)
    
    psi_rmin       =  gg.equilibria[eq_index].rmin
    psi_rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    psi_nr         =  gg.equilibria[eq_index].nw
    psi_zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    psi_zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    psi_nz         =  gg.equilibria[eq_index].nh
        
    psiPolSqrt     =  gg.equilibria[eq_index].psiPolSqrt
    rlcfs          =  gg.equilibria[eq_index].rlcfs
    zlcfs          =  gg.equilibria[eq_index].zlcfs

    #lcfs_coordinates_2d = np.zeros((zlcfs.size,2))
    #lcfs_polygon        = Polygon(lcfs_coordinates_2d)    # used to eliminate markers outside lcfs  7/14/22


    geq_rarray = np.linspace(psi_rmin, psi_rmax, psi_nr)
    geq_zarray = np.linspace(psi_zmin, psi_zmax, psi_nz)

    # transpose so that we are on a grid = [R,z]. define a function
    # psiPolSqrt_interp so that we can determine the local
    # value
         
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)   # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))
    #print("   ... define_prt_markers_03 position C:  type of rho_interpolator = ", type(rho_interpolator))
    #rhogeq_interp = interpolate.interp2d(geq_rarray, geq_zarray, psiPolSqrt, kind='cubic')
    #print("   ... position A in  define_prt_markers_03:  type of rhogeq_interp = ", type(rhogeq_interp))
    #print("   ... define_prt_markers_03 position D:  type of rho_interpolator = ", type(rho_interpolator))
    
    my_strings = fn_hdf5.split('.')
    stub = my_strings[0]
    
    #  previously we set a lower value of birth_rhomax because we
    #  used an approximate plasma separatrix.  Now we
    #  use the real equilibrium, so we can go closer to
    #  the edge
        
    birth_rhomax = 0.998
    birth_rhomin = 0.
    
    try:
        if settings["birth_rhomax"]:
            birth_rhomax = settings["birth_rhomax"]
    except:
        dummy = 0.

    try:
        if settings["birth_rhomin"]:
            birth_rhomin = settings["birth_rhomin"]
    except:
        dummy = 0.
            
    weight              = np.zeros((Nmrk))
    weight_energy_array = np.zeros((Nmrk))
    weight_qich_array   = np.zeros((Nmrk))
    cperp_array         = np.zeros((Nmrk))
    cpar_array          = np.zeros((Nmrk))
    ctot_array          = np.zeros((Nmrk))

    #  define 'candidate' R and Z values that are spread
    #  randomly over the entire domain of the equilibrium
        
    print("")
    
    print("  marker sets:  computing rho of markers (3)")
    print("")

    t_before = clock.time()

    imark = 0    # marker counter

    c_inflate    = 16.0
    if(Nmrk < 1000):
        c_inflate = 20.
    Nmrk_inflate = int(c_inflate * Nmrk)
    print("   ... number of markers and inflated number: ", Nmrk, Nmrk_inflate)
    
    #R_temp    = np.zeros(Nmrk_inflate)
    #z_temp    = np.zeros(Nmrk_inflate)

    # until 5/3/2022 we got the values for R_temp and z_temp within the
    # while loop, but that is slower by a factor of 75 than getting
    # them all at once

    try:
        rgrid_min = settings["rgrid_min"]
        rgrid_max = settings["rgrid_max"]
        zgrid_min = settings["zgrid_min"]
        zgrid_max = settings["zgrid_max"]
        R_temp     = rgrid_min + (rgrid_max - rgrid_min) * np.random.rand(Nmrk_inflate)
        z_temp     = zgrid_min + (zgrid_max - zgrid_min) * np.random.rand(Nmrk_inflate)
    except:
        R_temp     = psi_rmin + (psi_rmax - psi_rmin) * np.random.rand(Nmrk_inflate)
        z_temp     = psi_zmin + (psi_zmax - psi_zmin) * np.random.rand(Nmrk_inflate)


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  OVERRIDE THIS BECAUSE ALEX'S NEW EQULIBRIA
    #  COVER TOO MUCH TERRITORY    5/3/2022

    #R_temp     = 1.2  + 1.3 * np.random.rand(int(3.5*Nmrk_inflate))
    #z_temp     = -1.3 + 2.6 * np.random.rand(int(3.5*Nmrk_inflate))
    
    # rhos_temp  =  np.zeros(int(3.5*Nmrk_inflate))  # this is overridden below
    
    zlcfs_max  = np.max(zlcfs)
    zlcfs_min  = np.min(zlcfs)

    time_before = clock.time()
    rhos_temp   = rho_interpolator(R_temp, z_temp, grid=False)    # 5/11/2022:  do them all at once
    
    ictr = 0

    t_before = clock.time()
    ii_good  = (rhos_temp<= birth_rhomax) & (rhos_temp >= birth_rhomin) & ( z_temp<= zlcfs_max) & ( z_temp >= zlcfs_min)
    
    R_temp    =    R_temp[ii_good]
    z_temp    =    z_temp[ii_good]
    rhos_temp = rhos_temp[ii_good]

    if(R_temp.size < Nmrk):
        print("   ... define_prt_markers_03:  Nmrk, size, ratio = ", Nmrk, R_temp.size,  R_temp.size/Nmrk)
        print("                               NOT ENOUGH!")
        sys.exit()
    else:
        print("   ... define_prt_markers_03:  Nmrk, size = ", Nmrk, R_temp.size, "  (ok)"
        )
    #R_temp    =    R_temp[0:Nmrk_inflate]
    #z_temp    =    z_temp[0:Nmrk_inflate]
    #rhos_temp = rhos_temp[0:Nmrk_inflate]
    
       
    good_size = R_temp.size


    rho_biggest = np.max(rhos_temp)
    # ---------------------------------------------------
    #  paranoia check 9/3/2020

    #rho_biggest = 0.
    #for imark in range(Nmrk):
    #   this_rho = rhogeq_interp(R_temp[imark], z_temp[imark])
    #   rho_biggest = np.max([this_rho, rho_biggest])

    print("\n \n biggest rho of candidate markers: ",rho_biggest)
    if (rho_biggest > 1.00):
        pdb.set_trace()

    print("   ... we are at the end of the common-software portion of marker_03 \n")
    
    if set == 2:     # He3 tail
        
        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 3   * np.ones(ids.shape)
        charge = 2   * np.ones(ids.shape)
        anum   = 3   * np.ones(ids.shape)
        znum   = 2   * np.ones(ids.shape)
        phi    = 360 * np.random.rand(Nmrk)
        time   = 0.  * np.ones(ids.shape)

        energy = np.zeros(Nmrk)
        pitch  = np.zeros(Nmrk)
        gamma  = np.zeros(Nmrk)
        v      = np.zeros(Nmrk)
        vR     = np.zeros(Nmrk)
        vphi   = np.zeros(Nmrk)
        vz     = np.zeros(Nmrk)
        R      = np.zeros(Nmrk)
        z      = np.zeros(Nmrk)
        rhos   = np.zeros(Nmrk)

        cperp_max      = settings["cperp_max"]
        cpar_max       = settings["cpar_max"]
        do_goose       = settings["do_goose"]
        goose_factor   = settings["goose"]
        cperp_g        = settings["cperp_g"]
        fn_profiles    = settings["fn_profiles"]
        vpar_ratio_max = settings["vpar_ratio_max"]

        aa_profiles = proc.read_sparc_profiles_new(fn_profiles)

        tperp_rf  = aa_profiles["tperp_rf"]    # eV
        tpar_rf   = aa_profiles["tpar_rf"]
        rho_array = aa_profiles["rhosqrt"]
        qich      = aa_profiles["qich"]

        ctr   = 0
        imark = 0

        # we originally computed a marker's weight based on the nhe3 density
        # but following meeting with John and Pablo on 11/15 we
        # base the weight on the RF power density multiplied by zone
        # volume.  The 4th-order fit was computed manually (pdb.set_trace)
        # in process_v1c_profiles.py and the fit is shown in a figure
        # generated by that script.

        number_culled    = 0
        number_increased = 0
        
        while True:
            
           #print("ctr, imark, rhos_temp.size: ", ctr, imark, rhos_temp.size)
            
           rho = rhos_temp[imark]

           weight_qich = np.interp(rho, rho_array, qich)
           print(" rho, weight_qich: ", rho, weight_qich)
           
           #weight_density = 0.065 * rho - 0.014 * rho**2     # eyeball fit to nhe3 * zone volume
           #
           #if(weight_density <0.):
           #   print("   ... warning:  weight_density was negative. set to zero.")
           #   weight_density = 0.
              
           #coeffs = [-43.27721434, 119.23620643, -108.71233891, 28.61521171, -0.27592309]
           #
           #weight_pich =    coeffs[0] * rho**4   \
           #               + coeffs[1] * rho**4   \
           #               + coeffs[2] * rho**3   \
           #               + coeffs[3] * rho**2   \
           #               + coeffs[4] * rho      \
           #               + coeffs[5]
 
           tperp = np.interp(rho, rho_array, tperp_rf)
           tpar  = np.interp(rho, rho_array, tpar_rf)

           vpar_over_vperp_nominal = np.sqrt(tpar/tperp)

           cperp       = cperp_max * np.random.rand()
           this_eperp  = cperp * tperp
           this_vR     = np.sqrt(2. * electron_charge * this_eperp / (mass[0] * amu_mass))    # m/s

           # the parallel velocity is allowed to be some fraction of the perp velocity
           # up to some maximum ratio defined by the user

           vphi_nominal = np.sqrt(2. * electron_charge * tpar  / (mass[0] * amu_mass))
           vphi_rand    = np.random.rand()
           this_vphi    = this_vR * vpar_ratio_max * vpar_over_vperp_nominal * vphi_rand

           #originally cpar was taken randomly, now we compute it from this_vphi
           
           cpar  = (this_vphi/vphi_nominal)**2

           this_epar   = cpar  * tpar
           this_etot   = this_eperp + this_epar                                                 # eV

           this_v      = np.sqrt(2. * electron_charge * this_etot  / (mass[0] * amu_mass))    # m/s
           this_vz     = 0.

           vphi_sign = np.random.rand()
           if(vphi_sign <= 0.5):
                   this_vphi = -1. * this_vphi

           weight_energy =  np.exp( -1.*(cperp + cpar))

           q = np.random.rand()
           # pdb.set_trace()
           #  we skip a marker only if (1) goosing is turned on; AND (2) the candidate marker's value of cperp
           #  is bigger than the threshold cperp_g; AND (3) q > goose_factor

           if (do_goose !=1) or (cperp >= cperp_g) or ( q <= goose_factor):

               my_text = "accepted"
               
               if(do_goose == 1) and (cperp <= cperp_g) and  (q <= goose_factor):
                   #print("   ... upped weight: %3d %2d %6.3f  %6.3f %6.3f %6.3f " % (ctr, do_goose, cperp, cperp_g, q, goose_factor))
                   weight_energy = weight_energy / goose_factor
                   number_increased = number_increased + 1
                   my_text = "increased"

               R[ctr]      = R_temp[imark]
               z[ctr]      = z_temp[imark]
               rhos[ctr]   = rhos_temp[imark]
               v[ctr]      = this_v
               vR[ctr]     = this_vR
               vphi[ctr]   = this_vphi
               vz[ctr]     = this_vz
               energy[ctr] = this_etot
               pitch[ctr]  = vphi[ctr] / v[ctr]
               
               weight[ctr]              = weight_energy * weight_qich
               weight_energy_array[ctr] = weight_energy
               weight_qich_array[ctr]   = weight_qich

               cperp_array[ctr] = cperp
               cpar_array[ctr]  = cpar

               ctot_array[ctr]  = cperp+cpar
               
               print(" %6d  %6.3f %6.3f %6.3f %13.5e %13.5e %13.5e " % (ctr, rhos[ctr], cperp, cpar, weight_energy, weight_qich, weight[ctr]))     
               ctr += 1

               if (ctr >= Nmrk):
                   print(" we have enough markers!")
                   break
           else:
               #print("   ... culled:       %3d %2d %6.3f  %6.3f %6.3f %6.3f"% (ctr, do_goose, cperp, cperp_g, q, goose_factor))
               number_culled = number_culled + 1
               my_text = "culled"


           imark = imark + 1

        if (ctr < Nmrk):
            print("   ... marker_sets: insufficient number of markers so I must stop.")
            exit()
               
        # pdb.set_trace()

        # -------------------------------------------------------
        #
        fn_marker = stub + "combined_marker_plots.plt"
        with PdfPages(fn_marker) as pdf:

            pitch_derived = vphi/v
            print("   ... starting plots \n")
            plt.close()
            fig=plt.figure(figsize=(7., 5.))
            plt.plot(R, pitch_derived, 'ro', ms=1, rasterized=do_rasterized)
            plt.xlabel('Rmajor')
            plt.title('pitch versus Rmajor')
            pdf.savefig(fn)
            plt.close()
            plt.clf()

            plt.close()
            fig=plt.figure(figsize=(7., 5.))
            plt.plot(z, pitch_derived, 'ro', ms=1, rasterized=do_rasterized)
            plt.xlabel('Rmajor')
            plt.title('pitch versus z')
            pdf.savefig()
            plt.close()
            plt.clf()
        
            # -------------------------------------------------------
            #   weights vs rho
        
            plt.close()
            fig=plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight, 'ro', ms=1, rasterized=do_rasterized)
            plt.yscale('log')
            plt.ylim(1.e-5,100.)
            plt.xlim(0,1)
            plt.xlabel('rho')
            plt.title('total marker weight')
            pdf.savefig()
            plt.close()
            plt.clf()
        
            fig=plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight_qich_array, 'ro', ms=1, rasterized=do_rasterized)
            plt.xlim(0,1)
            plt.ylim(0,10)
            plt.xlabel('rho')
            plt.title('weight for RF power deposition')
            pdf.savefig()
            plt.close()

            fig=plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight_qich_array, 'ro', ms=1, rasterized=do_rasterized)
            plt.xlim(0,1)
            plt.ylim(0.01,10)
            plt.yscale('log')
            plt.xlabel('rho')
            plt.title('weight for RF power deposition')
            filename = stub + '_RF_weight_power_rho_log.pdf'
            pdf.savefig()
            plt.close()

            fig=plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight_energy_array, 'ro', ms=1, rasterized=do_rasterized)
            plt.xlim(0,1)
            plt.ylim(1.e-3, 3.)
            plt.xlabel('rho')
            plt.title('weight for energe')
            pdf.savefig()
            plt.close()

            fig=plt.figure(figsize=(7., 5.))
            plt.plot(vphi, vR, 'ro', ms=2, rasterized=do_rasterized)
            plt.xlabel('Vphi')
            plt.ylabel('vR')
            plt.savefig(filename)
            plt.close()
        
           #print("... %3d %2d %6.3f  %6.3f %6.3f %6.3f %s " % (ctr, do_goose, cperp, cperp_g, q, goose_factor, my_text))
            fraction_culled = (1.0*number_culled) / (Nmrk+number_culled)
            print("Number culled: ", number_culled)
            print("Number increased: ", number_increased)
            print("Fraction of markers culled: ", fraction_culled)
            total_weight = np.sum(weight)
            weight = weight / total_weight
        
    elif set == 3:     # He3 tail

        #  all markers at a given radius have the same eperp
        #  weighting is simply proportional to local deposited
        #  RF power.
        #  nominal ratio of v_par/v_perp is set by parallel
        #  to perpendicular temperature ratio
        #
        #  we distribute the markers equally in v_par / v_perp
        #  over 'npitch' values
        #
        #  so if Nmrk = 1000 and npitch = 6 you will get only
        #  166 positions and Nmrk will be re-defined to be
        #  6*166 = 996.
      
        npitch      = settings["npitch"]
        tperp_mult  = settings["tperp_mult"]
        fn_profiles = settings["fn_profiles"]

        nposition  = Nmrk // npitch
        Nmrk       = nposition * npitch
        
        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 3   * np.ones(ids.shape)
        charge = 2   * np.ones(ids.shape)
        anum   = 3   * np.ones(ids.shape)
        znum   = 2   * np.ones(ids.shape)
        phi    = 360 * np.random.rand(Nmrk)
        time   = 0.  * np.ones(ids.shape)

        v      = np.zeros(Nmrk)
        vR     = np.zeros(Nmrk)
        vphi   = np.zeros(Nmrk)
        vz     = np.zeros(Nmrk)
        R      = np.zeros(Nmrk)
        z      = np.zeros(Nmrk)
        rhos   = np.zeros(Nmrk)
        weight = np.zeros(Nmrk)
        pitch  = np.zeros(Nmrk)
        
        aa_profiles = proc.read_sparc_profiles_new(fn_profiles)

        tperp_rf  = aa_profiles["tperp_rf"]    # eV
        tpar_rf   = aa_profiles["tpar_rf"]
        rho_array = aa_profiles["rhosqrt"]
        qich      = aa_profiles["qich"]

        ctr = 0

        vphi_multipliers = np.linspace(-1,1,npitch)
        
        for ipos in range(nposition):
        
            for ip in range(npitch):

                R[ctr]      = R_temp[ipos]
                z[ctr]      = z_temp[ipos]
                rhos[ctr]   = rhos_temp[ipos]

                rho         = rhos[ctr]
                
                weight[ctr] = np.interp(rho, rho_array, qich)

                tpar   = np.interp(rho, rho_array, tpar_rf)
                tperp  = np.interp(rho, rho_array, tperp_rf)

                vpar_over_vperp = np.sqrt(tpar/tperp)

                tperp = tperp * tperp_mult

                vR[ctr]   = np.sqrt(2. * electron_charge * tperp / (mass[0] * amu_mass))    # m/s  
                vphi[ctr] = vR[ctr] * vpar_over_vperp * vphi_multipliers[ip]
                vz[ctr]   = 0.

                vtot = np.sqrt( vR[ctr]**2 + vphi[ctr]**2 + vz[ctr]**2)
                pitch[ctr] = vphi[ctr] / vtot

                ctr += 1
                
        weight_sum = np.sum(weight)
        weight     = weight / weight_sum

        # pdb.set_trace()
        
        plt.close()
        plt.clf()
        fig = plt.figure(figsize=(7., 5.))
        plt.plot(rhos, weight, 'ro', ms=1)
        plt.yscale('log')
        plt.xlim(0,1)
        plt.xlabel('rho')
        plt.title('marker weight')
        filename = stub + '_RF_weight_rho_log.pdf'
        plt.savefig(filename)
        plt.close()
        plt.clf()

        fig=plt.figure(figsize=(7., 5.))
        plt.plot(rhos, weight, 'ro', ms=1)
        plt.ylim(bottom=0.)
        plt.xlim(0,1)
        plt.xlabel('rho')
        plt.title('marker weight')
        filename = stub + '_RF_weight_rho_linear.pdf'
        plt.savefig(filename)
        plt.close()
        plt.clf()

        fig=plt.figure(figsize=(7., 5.))   
        plt.plot(vphi, vR, 'ro', ms=2)
        #plt.figure(figsize=(7., 5.))                
        plt.xlabel('Vphi')
        plt.ylabel('vR')
        #plt.show()
        filename = stub + '_vR_vs_vphi.pdf'
        plt.savefig(filename)

        plt.close()
        plt.clf()
        
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    elif set == 11:     # He3 tail  9/11/22

     #  weight markers by local nHe3 and Rmajor

     #  itype_pitch ... specifies how this routine will construct an
     #  array of birth pitch values
     #
     #   1    single pitch value, set by ratio of perp and parallel temps
     #   2    single pitch value, set by  settings["fixed_pitch"]
     #   3    npitch pitch values, that span the range [0, ratio parallel/perp temp] NOT SUPPORTED YET

        # ++++++++++++++++++++++++++++++++++++++++++++
        #  populate other arrays needed by ASCOT
        
        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 3   * np.ones(ids.shape)
        charge = 2   * np.ones(ids.shape)
        anum   = 3   * np.ones(ids.shape)
        znum   = 2   * np.ones(ids.shape)
        phi    = 360 * np.random.rand(Nmrk)
        time   = 0.  * np.ones(ids.shape)

        # ++++++++++++++++++++++++++++++++++++++
        
        itype_pitch = settings["itype_pitch"]

        if(itype_pitch == 1):
            npitch = 1
        elif(itype_pitch == 2):
            pitch_fixed  = settings["pitch_fixed"]
            npitch = 1
        elif(itype_pitch == 3):
            npitch      = settings["npitch"]
            print("   ... sorry, itype_pitch==3 is not supported yet")
            sys.exit()
        else:
            print("   ... marker sets:  itype_pitch = ", itype_pitch, " is not supported.")
            sys.exit()
            
        tperp_mult   = settings["tperp_mult"]
        fn_profiles  = settings["fn_profiles"]
        fn_pitch_geq = settings["fn_pitch_geq"]   # provides data on Bfield so we
                                                 # can properly convert from v_parallel
                                                 # to R, phi, Z etc

        print("   ... about to call a5 = Ascotpy(fn)")
        a5_pitch     = Ascotpy(fn_pitch_geq)
        print("   ... about to a5_pich.init")
        a5_pitch.init(bfield=True)
        print("   ... have completed a5_pitch.init")

        aa_profiles = proc.read_sparc_profiles_new(fn_profiles)
        
        idensity      = aa_profiles["idensity"]
        anums_profile = aa_profiles["anum"]
        znums_profile = aa_profiles["znum"]

        index_He3 = ( anums_profile == 3.) & (znums_profile == 2.)

        nHe3 = idensity[:,index_He3]
        if(nHe3.size == 0):
            print("    I cannot find data for the 3He species so I must quit.")
            sys.exit()
        nHe3 = nHe3.flatten()
            
        tperp_rf  = aa_profiles["tperp_rf"]    # eV
        tpar_rf   = aa_profiles["tpar_rf"]
        rho_array = aa_profiles["rhosqrt"]
        qich      = aa_profiles["qich"]

        ctr = 0
        
    
        tpars             =  np.interp(rhos_temp, rho_array, tpar_rf)
        tperps            =  np.interp(rhos_temp, rho_array, tperp_rf)
        gyro_angles       =  np.random.rand(rhos_temp.size) * 2. * np.pi


        #  calculation of weights:  for maxwellian, see memo "rf_tail_09.pdf"
        #  we impose an artificial minimum on cpars to avoid the possibility of getting, by
        #  pure chance, a value very close to zero that would skew the weights
        
        try:
            cmax = settings["cmax"]
        except:
            cmax = 3.
            print("   ...define_prt_markers_03 (11):  settings[cmax] not defined, will use value = 3")

    
        try:
            cpar_min = settings["cpar_min"]
            print("   ... setting cpar_min to: ", cpar_min)
        except:
            cpar_min = 0.01
            
        cperps = cmax                       * np.random.rand(rhos_temp.size)   # uniform distribution:  [0,  cmax]
        cpars  = cpar_min + (cmax-cpar_min) * np.random.rand(rhos_temp.size)   # uniform distribution:  [0.1,cmax]

        eperps_kev = tperps * cperps / 1000.  # for plotting only
        epars_kev  = tpars  * cpars  / 1000.  # for plotting only
        
        weights_maxwellian = np.sqrt(1./cpars) * np.exp( -1.*(cperps + cpars))
        weights_position  =  np.interp(rhos_temp, rho_array, nHe3) * R_temp
        
        weight = np.multiply(weights_position, weights_maxwellian)
        weight = weight / np.sum(weight)

        vparallels  = np.sqrt(2. * electron_charge * tpars  * cpars  / (mass[0] * amu_mass))    # m/s  ... check arithmetic
        vperps      = np.sqrt(2. * electron_charge * tperps * cperps / (mass[0] * amu_mass))    # m/s  ... check arithmetic
        vtots       = np.sqrt(vparallels**2 + vperps**2)

        #  but we must allow v_parallel to be both positive and negative.  so far it is only positive

        plus_minus_random = np.random.rand(vparallels.size)
        plus_minus = np.ones(vparallels.size)
        ii_negative = (plus_minus_random <= 0.5)
        plus_minus[ii_negative] = -1

        nn_positive = plus_minus[ (plus_minus >0)].size
        nn_negative = plus_minus[ (plus_minus <0)].size

        print("   ... number of positive and negative v_parallels: ", nn_positive, nn_negative)
        # pdb.set_trace()
        vparallels = vparallels * plus_minus
        
        # pdb.set_trace(header='after calc of weight') 
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  read the 3D magnetic field from an earlier ASCOT simulation

        phigrid = np.zeros(R_temp.size)
        
        bPhi, bR, bZ, bfield_3D  = CB.compute_bfield_arrays(a5_pitch, R_temp, phigrid, z_temp)

        btots      = np.sqrt( bPhi**2 + bR**2 + bZ**2)
        bR_hats    = bR   / btots
        bPhi_hats  = bPhi / btots
        bZ_hats    = bZ   / btots

        vperp_hats = np.transpose(CV.compute_vperp_hat(bR_hats, bPhi_hats, bZ_hats, gyro_angles))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  finally, compute the velocity components
        #
        #  V = v_parallel x b_hat + v_perp x vperp_hat     [R, phi, Z]
        #  see  vparallel_conversion.pptx   9/14/22
        
        if(itype_pitch == 1):         # get pitch from t_parallel and t_perp

            vR   = vparallels *   bR_hats  + vperps * vperp_hats[:,0]
            vphi = vparallels * bPhi_hats  + vperps * vperp_hats[:,1]
            vz   = vparallels *   bZ_hats  + vperps * vperp_hats[:,2]

            pitch = vparallels/vtots
            thetas = np.arccos(pitch) * 180./np.pi
            
        elif(itype_pitch == 2):        # get pitch from the user

            vparallels = pitch_fixed * vtots
            vperps     = np.sqrt(vtots**2 - vparallels**2)
            
            vR   = vparallels * bR_hats   + vperps * vperp_hats[:,0]
            vphi = vparallels * bPhi_hats + vperps * vperp_hats[:,1]

            vz   = vparallels * bZ_hats   + vperps * vperp_hats[:,2]

            pitch = vparallels/vtots
       
        weight_sum = np.sum(weight)
        weight     = weight / weight_sum

        # ++++++++++++++++++++++++++++++++++++++++++++
        #  populate other arrays needed by ASCOT
        
        R           =    R_temp[0:Nmrk]
        z           =    z_temp[0:Nmrk]
        rhos        = rhos_temp[0:Nmrk]
        weight      =    weight[0:Nmrk]
        vR          =        vR[0:Nmrk]
        vz          =        vz[0:Nmrk]
        vphi        =      vphi[0:Nmrk]
        pitch       =     pitch[0:Nmrk]
        eperps_kev = eperps_kev[0:Nmrk]
        epars_kev  =  epars_kev[0:Nmrk]
        weights_maxwellian = weights_maxwellian[0:Nmrk]
        weights_position   =   weights_position[0:Nmrk]
        thetas             =   thetas[0:Nmrk]
        
        
        proton_mass = 1.67e-27

        vtots     = vphi**2 + vR**2 + vz**2
        etots_kev = 0.5 * proton_mass * anum * vtots / (1000. * 1.602e-19)

        
        # mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  plots
        
        fn_stub = fn_hdf5.split('.')[0]
        filename_marker = fn_stub + '_markerplots.pdf'

        with PdfPages(filename_marker) as pdf:

            fn_stub = fn_hdf5.split('.')[0]
            filename_marker = fn_stub + '_markerplots.pdf'

            plt.close('all')
            plt.clf()

            #pdb.set_trace()

            plt.title(" marker position weights")
            plt.xlabel("rho_poloidal")
            plt.plot(rhos, weights_position, 'bo', ms=2, rasterized=True)
            plt.ylim(bottom=0.)
            pdf.savefig()
            plt.close()
            print("   ... have finished plot:  marker position-weights vs rho")

            plt.title(" marker maxwellian weights")
            plt.xlabel("rho_poloidal")
            plt.plot(rhos, weights_maxwellian, 'bo', ms=2, rasterized=True)
            plt.yscale('log')
            pdf.savefig()
            plt.close()
            print("   ... have finished plot:  marker maxwellian-weights vs rho")

            plt.title(" marker total weights")
            plt.xlabel("rho_poloidal")
            plt.plot(rhos, weight, 'bo', ms=2, rasterized=True)
            plt.yscale('log')
            pdf.savefig()
            plt.close()
            print("   ... have finished plot:  marker maxwellian-weights vs rho")
            
            plt.title(" histogram of birth eperp keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(eperps_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k')
            plt.hist(eperps_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            plt.hist(eperps_kev[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g')           
            plt.hist(eperps_kev[ (rhos>0.75)],                bins=30, histtype='step', color='c')
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth eperp keV")

            plt.title(" histogram of birth eperp keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(eperps_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k', log=True)
            plt.hist(eperps_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', log=True)
            plt.hist(eperps_kev[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', log=True)           
            plt.hist(eperps_kev[ (rhos>0.75)],                bins=30, histtype='step', color='c', log=True)
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: (log) histogram of birth eperp keV")

            plt.title(" histogram of birth eperp keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(eperps_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k')
            plt.hist(eperps_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: another histogram of birth eperp keV")

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth epar keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(epars_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k')
            plt.hist(epars_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            plt.hist(epars_kev[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g')           
            plt.hist(epars_kev[ (rhos>0.75)],                bins=30, histtype='step', color='c')
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth epar keV")

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth epar keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(epars_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k')
            plt.hist(epars_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: another histogram of birth epar keV")

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth keV (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(etots_kev[ (rhos<0.25)],              bins=30, histtype='step', color='k')
            plt.hist(etots_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            plt.hist(etots_kev[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g')           
            plt.hist(etots_kev[ (rhos>0.75)],              bins=30, histtype='step', color='c')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of WEIGHTD birth keV density (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("[keV]")
            plt.hist(etots_kev[ (rhos<0.25)],                bins=30, histtype='step', color='k', log=True,  density=True, weights = weight[ (rhos<0.25)])
            plt.hist(etots_kev[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', log=True,  density=True, weights = weight[ (rhos>0.25) & (rhos< 0.50)])
            plt.hist(etots_kev[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', log=True,  density=True, weights = weight[ (rhos>0.50) & (rhos< 0.75)])           
            plt.hist(etots_kev[ (rhos>0.75)],                bins=30, histtype='step', color='c', log=True,  density=True, weights = weight[ (rhos>0.75)])
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of WEIGHTD birth keV density (k-r-g = 0.1/0.2/0.3")
            plt.xlabel("[keV]")
            plt.hist(etots_kev[ (rhos<0.10)],                bins=30, histtype='step', color='k', log=True,  density=True, weights = weight[ (rhos<0.10)])
            plt.hist(etots_kev[ (rhos>0.10) & (rhos< 0.20)], bins=30, histtype='step', color='r', log=True,  density=True, weights = weight[ (rhos>0.10) & (rhos< 0.20)])
            plt.hist(etots_kev[ (rhos>0.20) & (rhos< 0.30)], bins=30, histtype='step', color='g', log=True,  density=True, weights = weight[ (rhos>0.20) & (rhos< 0.30)])
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth pitch (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("pitch")
            plt.hist(pitch[ (rhos<0.25)],                bins=30, histtype='step', color='k')
            plt.hist(pitch[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r')
            plt.hist(pitch[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g')           
            plt.hist(pitch[ (rhos>0.75)],                bins=30, histtype='step', color='c')
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth pitch")

            #pdb.set_trace
            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of WEIGHTED birth pitch (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("pitch")
            plt.hist(pitch[ (rhos<0.25)],                bins=30, histtype='step', color='k', weights=weight[ (rhos<0.25)])
            plt.hist(pitch[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', weights=weight[ (rhos>0.25) & (rhos< 0.50)])
            plt.hist(pitch[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', weights=weight[ (rhos>0.50) & (rhos< 0.75)])           
            plt.hist(pitch[ (rhos>0.75)],                bins=30, histtype='step', color='c', weights=weight[ (rhos>0.75)])
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth pitch")

            plt.figure(figsize=(8.,6.))
            plt.title(" WEIGHTED pitch angle theta(k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel(" theta")
            plt.hist(thetas[ (rhos<0.25)],                bins=30, histtype='step', color='k', weights=weight[ (rhos<0.25)])
            plt.hist(thetas[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', weights=weight[ (rhos>0.25) & (rhos< 0.50)])
            plt.hist(thetas[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', weights=weight[ (rhos>0.50) & (rhos< 0.75)])           
            plt.hist(thetas[ (rhos>0.75)],                bins=30, histtype='step', color='c', weights=weight[ (rhos>0.75)])
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of pitch angle theta")

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth pitch (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("pitch")
            plt.hist(pitch[ (rhos<0.25)],                bins=30, histtype='step', color='k', density=True)
            plt.hist(pitch[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', density=True)
            plt.hist(pitch[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', density=True)           
            plt.hist(pitch[ (rhos>0.75)],                bins=30, histtype='step', color='c', density=True)
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth pitch (density)")

            plt.figure(figsize=(8.,6.))
            plt.title(" histogram of birth pitch (k-r-g-c = rho/0.25/0.5/0.75)")
            plt.xlabel("pitch")
            plt.hist(pitch[ (rhos<0.25)],                bins=30, histtype='step', color='k', density=True, weights=weight[ (rhos<0.25)])
            plt.hist(pitch[ (rhos>0.25) & (rhos< 0.50)], bins=30, histtype='step', color='r', density=True, weights=weight[ (rhos>0.25) & (rhos< 0.50)])
            plt.hist(pitch[ (rhos>0.50) & (rhos< 0.75)], bins=30, histtype='step', color='g', density=True, weights=weight[ (rhos>0.50) & (rhos< 0.75)])           
            plt.hist(pitch[ (rhos>0.75)],                bins=30, histtype='step', color='c', density=True, weights=weight[ (rhos>0.75)])
            pdf.savefig()
            plt.close()
            print("   ... have finished creating: histogram of birth pitch (density, weighted)")
                     
            fig = plt.figure(figsize=(8.,6.))
            plt.plot(rhos, pitch, 'bo', ms=1, rasterized=True)
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Pitch vs rho')
            plt.tight_layout(pad=1)
            plt.grid('True')
            pdf.savefig()
            plt.close()

            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')           
            ii = (z < -0.5) 
            this_line, = plt.plot(rhos[ii], pitch[ii], 'ko', ms=1, label=" Z < -0.50", rasterized=True)
            my_labels.append(this_line)
            ii = (z > -0.5) & (z < -0.25)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'ro', ms=1, label=" -0.50 < Z < -0.25", rasterized=True)
            my_labels.append(this_line)
            ii = (z > -0.25) & (z < 0.00)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'o', ms=1, color='orange', label=" -0.25 < Z < -0.00", rasterized=True)
            my_labels.append(this_line)
            ii = (z > 0) & (z < 0.25)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'go', ms=1, label=" 0.00 < Z < 0.25", rasterized=True)
            my_labels.append(this_line)
            ii = (z > 0.25) & (z < 0.50)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'co', ms=1, label=" 0.25 < Z < 0.50", rasterized=True)
            my_labels.append(this_line)
            ii = (z > 0.50) 
            this_line, = plt.plot(rhos[ii], pitch[ii], 'bo', ms=1, label=" 0.50 < Z", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            pdf.savefig()
            plt.close()
            
            # ++++++++++++++++++++++++++++++++++++++
            #  individual curves

            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))
            ii = (z < -0.5) 
            this_line, = plt.plot(rhos[ii], pitch[ii], 'ko', ms=3, label=" Z < -0.50", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()
            plt.close()


            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))
            ii = (z > -0.5) & (z < -0.25)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'ro', ms=3, label=" -0.50 < Z < -0.25", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()
            plt.close()

            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))
            ii = (z > -0.25) & (z < 0.00)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'o', ms=3, color='orange', label=" -0.25 < Z < -0.00", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()

            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))
            ii = (z > 0) & (z < 0.25)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'go', ms=3, label=" 0.00 < Z < 0.25", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()
            plt.close()
            
            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))

            ii = (z > 0.25) & (z < 0.50)
            this_line, = plt.plot(rhos[ii], pitch[ii], 'co', ms=3, label=" 0.25 < Z < 0.50", rasterized=True)
            my_labels.append(this_line) 
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()
            plt.close()
            
            my_labels = []
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('rho_poloidal')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs rho_poloidal')
            plt.ylim((0.15,0.52))
            ii = (z > 0.50) 
            this_line, = plt.plot(rhos[ii], pitch[ii], 'bo', ms=3, label=" 0.50 < Z", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="upper left", fontsize=9)
            plt.tight_layout(pad=1)
            plt.xlim((0.,1.05))
            pdf.savefig()
            plt.close()
            #  individual curves (end)
            # ++++++++++++++++++++++++++++++++++++++++++++++++

            
            my_labels = []
            plt.clf()
            fig = plt.figure(figsize=(8.,6.))
            plt.xlabel('Rmajor [m]')
            plt.ylabel('pitch')
            plt.title('Pitch vs Rmajor')
            plt.tight_layout(pad=1)
            ii = (z<=-0.75)
            this_line, = plt.plot(R[ii], pitch[ii], 'ko', ms=1, label="Z < -0.75", rasterized=True)
            my_labels.append(this_line)
            ii = (z>-0.75) & (z<-0.5)
            this_line, = plt.plot(R[ii], pitch[ii], 'ro', ms=1, label=" -0.75 < Z < -0.5", rasterized=True)
            my_labels.append(this_line)
            ii = (z>-0.5) & (z<-0.25)
            this_line, = plt.plot(R[ii], pitch[ii], 'o', color='orange', ms=1, label=" -0.5 < Z < -0.25", rasterized=True)
            my_labels.append(this_line)
            ii = (z>-0.25) & (z<0.)
            this_line, = plt.plot(R[ii], pitch[ii], 'yo',  ms=1, label=" -0.25 < Z < 0.", rasterized=True)
            my_labels.append(this_line)
            ii = (z>0.) & (z<0.25)
            this_line, = plt.plot(R[ii], pitch[ii], 'go',  ms=1, label=" 0. < Z < 0.25", rasterized=True)
            my_labels.append(this_line)
            ii = (z>0.25) & (z<0.50)
            this_line, = plt.plot(R[ii], pitch[ii], 'co',  ms=1, label=" 0.25 < Z < 0.50", rasterized=True)
            my_labels.append(this_line)
            ii = (z>0.50) & (z<0.75)
            this_line, = plt.plot(R[ii], pitch[ii], 'bo',  ms=1, label=" 0.50 < Z < 0.75", rasterized=True)
            my_labels.append(this_line)
            ii = (z>0.75)
            this_line, = plt.plot(R[ii], pitch[ii], 'o',  color='magenta', ms=1, label=" 0.75 <Z ", rasterized=True)
            my_labels.append(this_line)
            plt.legend(handles=my_labels, loc="lower right", fontsize=9)
            pdf.savefig()

            
            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(R, z, 'bo', ms=1, rasterized=True)
            plt.xlabel('Rmajor [m]')
            plt.ylabel('Elevation [m]')
            plt.title('Marker birth positions')
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(R, pitch, 'bo', ms=1, rasterized=True)
            plt.xlabel('Rmajor [m]')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs Rmajor')
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(z, pitch, 'bo', ms=1, rasterized=True)
            plt.xlabel('Elevation [m]')
            plt.ylabel('pitch')
            plt.title('Marker pitch vs elevation')
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(R, vR, 'bo', ms=1, rasterized=True)
            plt.xlabel('Rmajor [m]')
            plt.ylabel('m/s')
            plt.title('Marker vR vs Rmajor')
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(R, vphi, 'bo', ms=1, rasterized=True)
            plt.xlabel('Rmajor [m]')
            plt.title('Marker vphi vs Rmajor', rasterized=True)
            plt.tight_layout(pad=2)
            plt.ylabel('m/s')
            pdf.savefig()

            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(4.,7.))
            plt.plot(R, vz, 'bo', ms=1, rasterized=True)
            plt.xlabel('Rmajor [m]')
            plt.ylabel('m/s')
            plt.title('Marker vz vs Rmajor')
            plt.tight_layout(pad=1)
            pdf.savefig()
            
            plt.close()
            plt.clf()
            fig = plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight, 'ro', ms=1, rasterized=True)
            plt.yscale('log')
            plt.xlim(0,1.05)
            plt.xlabel('rho')
            plt.title('marker weight')
            plt.grid(axis='both', alpha=0.75)
            plt.tight_layout(pad=2)
            pdf.savefig()
            
            plt.close()
            plt.clf()
            fig=plt.figure(figsize=(7., 5.))
            plt.plot(rhos, weight, 'ro', ms=1, rasterized=True)
            plt.ylim(bottom=0.)
            plt.xlim(0,1.05)
            plt.xlabel('rho')
            plt.grid(axis='both', alpha=0.75)
            plt.title('marker weight')
            plt.tight_layout(pad=2)
            pdf.savefig()
            
            plt.close()
            plt.clf()
            fig=plt.figure(figsize=(7., 5.))   
            plt.plot(vphi, vR, 'ro', ms=2, rasterized=True)       
            plt.xlabel('Vphi')
            plt.ylabel('vR')
            plt.title("vR vs vPhi")
            plt.tight_layout(pad=1)
            pdf.savefig()

                        
            plt.close()
            plt.clf()
            fig=plt.figure(figsize=(7., 5.))   
            plt.plot(vphi, vz, 'ro', ms=2, rasterized=True)       
            plt.xlabel('Vphi')
            plt.ylabel('vZ')
            plt.title("vZ vs vPhi")
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig=plt.figure(figsize=(7., 5.))   
            plt.plot(vphi, np.sqrt(vR**2 + vz**2), 'ro', ms=2, rasterized=True)       
            plt.xlabel('Vphi')
            plt.ylabel('')
            plt.title("(vz**2 + vR**2)**0.5 vs vPhi")
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()
            fig=plt.figure(figsize=(7., 5.))   
            plt.plot(pitch, vphi/(np.sqrt(vR**2 + vz**2+ vphi**2)), 'bo', ms=2, rasterized=True)       
            plt.xlabel('pitch')
            plt.ylabel('vphi/vtot')
            plt.title("vphi/vtot vs pitch")
            plt.tight_layout(pad=1)
            pdf.savefig()

            plt.close()
            plt.clf()

    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif set == 1:

         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         R     =     R_temp[0:Nmrk]
         z     =     z_temp[0:Nmrk]
         rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c
        
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = np.interp(rhos[imark], rho_array, alpha_source)
             
             #if rhos[imark] <= 0.6:
             #    weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
             #else:
             #    weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)
             #
             #if (weight[imark] <= 0.):
             #    weight[imark] = 1.e-6

         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
         vR   = np.sqrt(1-pitch*pitch) * v
         vphi = pitch * v
         vz   = 0.* v           # why vz=0 identically?

         v_poloidal = np.sqrt(vR**2 + vz**2)

    
    # ===========================================================================
    
    elif set == 5:

         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         R     =     R_temp[0:Nmrk]
         z     =     z_temp[0:Nmrk]
         rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c
        
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = np.interp(rhos[imark], rho_array, alpha_source)
             
             #if rhos[imark] <= 0.6:
             #    weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
             #else:
             #    weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)
             #
             #if (weight[imark] <= 0.):
             #    weight[imark] = 1.e-6

         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)

         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v
        

         # realization 1/23/2020 ... must make velocity random in
         # poloidal direction also

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)
         
        # ===========================================================================
        # gadzooks, had the wrong distribution in pitch angle
        
    elif set == 6:

         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         R     =     R_temp[0:Nmrk]
         z     =     z_temp[0:Nmrk]
         rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c
        
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = np.interp(rhos[imark], rho_array, alpha_source)
             
             #if rhos[imark] <= 0.6:
             #    weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
             #else:
             #    weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)
             #
             #if (weight[imark] <= 0.):
             #    weight[imark] = 1.e-6

         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         # pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)  # removed 1/26/2020

         nn_pitch        = 10* Nmrk
         pitch_candidate = -1 + 2 * np.random.rand(nn_pitch)
         pitch_weights   = np.sqrt(1-pitch_candidate**2)
         rr_pitch        = np.random.rand(nn_pitch)
         pitch_out       = np.zeros(nn_pitch)
         mm_pitch        = 0
         
         for kk in range(nn_pitch):
             if(rr_pitch[kk] <= pitch_weights[kk]):
                pitch_out[mm_pitch] = pitch_candidate[kk]
                mm_pitch += 1
         pitch_out = pitch_out[0:mm_pitch-1]
         kk_good = (pitch_out >= pitch_min) & (pitch_out <= pitch_max)
         pitch_out = pitch_out[kk_good]
         mm_good   = pitch_out.size
         if(mm_good < Nmrk):
             print("  I could not find enough good pitch angles")
             exit()
         else:
             pitch = pitch_out[0:Nmrk]
         
         # ----------------------------------------------------------
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
         # pdb.set_trace()
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v
        

         # realization 1/23/2020 ... must make velocity random in
         # poloidal direction also

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)
             
    # -----------------------------------------------------------------
    #   gadzooks, had the wrong ensemble in Rmajor also
    
    elif set == 7:
         
         print("   ... marker_sets_03 (set 7):  nrho_in = ", nrho_in)
         estimated_time = 4.1e-6 * Nmrk
         print("       estimated processing time = ", estimated_time, " minutes")

         #pdb.set_trace()
         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles, nrho_in)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         # logic below gives probability of a marker proportional to its rmajor
         
         print("   ... starting marker probability proportional to Rmajor \n")
         
         mysize = R_temp.size             # added 1.5 4/16/2021
         myrand = np.random.rand(mysize)

         good_R    = np.zeros(mysize)
         good_z    = np.zeros(mysize)
         good_rhos = np.zeros(mysize)

         R_big = psi_rmax   # must be bigger than any rmajor. was = 4. until 2/19/20
         mm_count = 0
         
         for kk in range(mysize):

             if( (R_temp[kk]/R_big) >= myrand[kk]):
                 good_R[mm_count]       = R_temp[kk]
                 good_z[mm_count]       = z_temp[kk]
                 good_rhos[mm_count] = rhos_temp[kk]
                 mm_count += 1
         if(mm_count < Nmrk):
             print(" insufficient r-weighted markers:  Nmrk, mm_count: ", Nmrk, mm_count)
             exit()

         R    = good_R[0:Nmrk]
         z    = good_z[0:Nmrk]
         rhos = good_rhos[0:Nmrk]
         
         #R     =     R_temp[0:Nmrk]
         #z     =     z_temp[0:Nmrk]
         #rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c
         
         print("   ... computing marker weights ")
         
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = np.interp(rhos[imark], rho_array, alpha_source)
             
             #if rhos[imark] <= 0.6:
             #    weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
             #else:
             #    weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)
             #
             #if (weight[imark] <= 0.):
             #    weight[imark] = 1.e-6
         print("   ... marker weight computation finished \n")
         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
         print("   ... about to create some arrays ")
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
               print("  marker_sets/define_prt_markers_03 (set=7):  setting alpha energy to: ", e_alpha)
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         # pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)  # removed 1/26/2020
         
         pitch = compute_random_pitch(Nmrk, pitch_min, pitch_max)

         
         
         #gamma = 1 + energy * const["elementary charge"][0]               \
         #  / ( const["alpha particle mass"][0]                          \
         #       * np.power(const["speed of light in vacuum"][0],2) )

         alpha_particle_mass = 6.645e-27    # scipy.const changed via pip circa 3/9/2021
         #  was const_elementary_charge until 2/16/2022
         
         gamma = 1 + energy * const["elementary charge"][0]  / ( alpha_particle_mass \
                                                * np.power(const["speed of light in vacuum"][0],2) )
         
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
         # pdb.set_trace()
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v
   

         # realization 1/23/2020 ... must make velocity random in
         # poloidal direction also

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)

         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         # sds 1/15/2023
         #
         #  new section to generate more markers near center
         #  ... this may be useful when running ASCOT to compute
         #      the alpha distribution function
         #
         #  basic idea:  "split" all markers into multiple markers;
         #  the number Nd of daughter markers is a function of rho_poloidal
         #  of the parent marker.
         #
         #  Let rho_crit = 1./settings["ncenter_split"]
         #  then   Nd(rho) = np.max(1, int(1./(rho + rho_crit))
         #
         #  so for example if you want to split the markers by a factor
         #  of 5 at the center, reducing to 1 at the edge, you would
         #  specify  ncenter_split = 5.  and then rho_crit = 0.20
         #
         #  the positions of daughter markers will be identical to that
         #  of the parent markers, as will vphi and vpoloidal.  But we
         #  will assign random velocity gyro-angles to the daughter
         #  markers, so their values of vR and vz will differ from
         #  the parent marker.
         
         #  Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time

         try:
             do_split_markers = settings["do_split_markers"]
         except:
             do_split_markers = 0

         if(do_split_markers == 1):

             Nmrk_before_split = Nmrk
             print("   ... start marker-split process at time: ",clock.ctime(clock.time()))
             
             rho_crit = 1./settings["ncenter_split"]
             print("   ... marker_sets.py/define_prt_markers_03 (set == 7):  splitting markers with rho_crit = %7.3f"%(rho_crit))
             
             Nmrk_d   = 0
             
             ids_d    =  np.zeros(0)
             rhos_d   =  np.zeros(0)
             mass_d   =  np.zeros(0)
             charge_d =  np.zeros(0)
             R_d      =  np.zeros(0)
             phi_d    =  np.zeros(0)
             z_d      =  np.zeros(0)
             vR_d     =  np.zeros(0)
             vphi_d   =  np.zeros(0)
             vz_d     =  np.zeros(0)
             anum_d   =  np.zeros(0)
             znum_d   =  np.zeros(0)
             weight_d =  np.zeros(0)
             time_d   =  np.zeros(0)
             Nd_array =  np.zeros(0)  # this used for plotting and debugging only

             jctr = 0

 
             # do not forget that ids[0] = 1
             
             rhos_parent   = rhos    # for overplotting with split-marker ensemble
             weight_parent = weight  # for overplotting with split-marker ensemble
             
             for jm in range(Nmrk):     # Loop over all parent markers, generating 'daughter' markers for each

                 if(jm == 500*(jm//500)):
                     print("   ... starting parent marker: ", jm)
                     
                 Nd = np.max((1, int(1./(rho_crit + rhos[jm]))))    # number of daughter markers for this parent marker
                 
                 if(rho_crit == 1.):   # ncenter_split=1. this is mostly for debugging
                     Nd= 1
                     print("   ... marker_sets_center_weight/define_prt_markers_03:  setting Nd=1")
                     
                 #  Nd = 2    # special debug case  1q/16/23
                 
                 new_ids = jctr + 1 +  (np.linspace(0, Nd-1, Nd)).astype(int)    # need to provide unique IDs for all markers
                 jctr   += Nd
                             
                 this_weight = weight[jm] / Nd
                             
                 ids_d    =  np.append(ids_d,    new_ids)
                 
                 rhos_d   =  np.append(rhos_d,   rhos[jm]    * np.ones(Nd))        
                 mass_d   =  np.append(mass_d,   mass[jm]    * np.ones(Nd))
                 charge_d =  np.append(charge_d, charge[jm]  * np.ones(Nd))
                 R_d      =  np.append(R_d,      R[jm]       * np.ones(Nd))     
                 phi_d    =  np.append(phi_d,    phi[jm]     * np.ones(Nd))
                 z_d      =  np.append(z_d,      z[jm]       * np.ones(Nd))
                 anum_d   =  np.append(anum_d,   anum[jm]    * np.ones(Nd))
                 znum_d   =  np.append(znum_d,   znum[jm]    * np.ones(Nd))
                 weight_d =  np.append(weight_d, this_weight * np.ones(Nd))
                 time_d   =  np.append(time_d,   time[jm]    * np.ones(Nd))
                 Nd_array =  np.append(Nd_array, Nd          * np.ones(Nd))
                 
                 vtot_scalar     = np.sqrt(vphi[jm]**2 + v_poloidal[jm]**2)        # 2/8/23:  randomize entire velocity vector
                 pitch_array     = compute_random_pitch(Nd, pitch_min, pitch_max)  # array of Nd random pitches
                 vphi_local      = vtot_scalar * pitch_array
                 vpoloidal_local = vtot_scalar * np.sqrt(1.-pitch_array * pitch_array)
                 
                 vphi_d   =  np.append(vphi_d,   vphi_local)               
                             
                 gyro_angles = (2. * np.pi) * np.random.rand(Nd)    # daughter markers have random gyro-angle
                 new_vR  = vpoloidal_local * np.cos(gyro_angles)
                 new_vz  = vpoloidal_local * np.sin(gyro_angles)                             

                 vR_d = np.append(vR_d, new_vR)
                 vz_d = np.append(vz_d, new_vz)
                             
             Nmrk = jctr
             #pdb.set_trace()
                                       
             # ++++++++++++++++++++++++++++++
             #  sanity checks
             
             total_weight = np.sum(weight_d)
             if(np.abs(total_weight - 1.)>1.e-4):
                 print("   ... sorry, total weight after splitting markers is: ", total_weight)
                 sys.exit()
                             
             assert Nmrk == ids_d.size,     "Nmrk and ids_d not same size"
             assert Nmrk == mass_d.size,    "Nmrk and mass_d not same size"
             assert Nmrk == charge_d.size,  "Nmrk and charge_d not same size"
             assert Nmrk == R_d.size,       "Nmrk and R_d not same size"
             assert Nmrk == phi_d.size,     "Nmrk and phi_d not same size"
             assert Nmrk == z_d.size,       "Nmrk and z_d not same size"
             assert Nmrk == anum_d.size,  "Nmrk and anum_d not same size"
             assert Nmrk == znum_d.size,    "Nmrk and znum_d not same size"
             assert Nmrk == weight_d.size,  "Nmrk and weight_d not same size"
             assert Nmrk == time_d.size,    "Nmrk and time_d not same size"
             assert Nmrk == vphi_d.size,    "Nmrk and vphi_d not same size"
             assert Nmrk == vR_d.size,      "Nmrk and vr_d not same size"
             assert Nmrk == vz_d.size,      "Nmrk and vz_d not same size"
             assert Nmrk == Nd_array.size,  "Nmrk and Nd_array not same size"

             # shuffle the daughter markers' indices, then reduce
             # to user-specified number of markers (Nmrk_max_extended)
                             
             ids_random = np.arange(Nmrk)
             np.random.shuffle(ids_random)  # changes contents of ids_random
             #pdb.set_trace(header="almost done")
             Nmrk_extended = np.min((Nmrk, settings["Nmrk_max_extended"]))

             rhos    =  np.squeeze(rhos_d[ids_random][0:Nmrk_extended])
             mass   =   np.squeeze(mass_d[ids_random][0:Nmrk_extended])
             charge = np.squeeze(charge_d[ids_random][0:Nmrk_extended])
             R      =      np.squeeze(R_d[ids_random][0:Nmrk_extended])
             phi    =    np.squeeze(phi_d[ids_random][0:Nmrk_extended])
             z      =      np.squeeze(z_d[ids_random][0:Nmrk_extended])
             anum   =   np.squeeze(anum_d[ids_random][0:Nmrk_extended])
             znum   =   np.squeeze(znum_d[ids_random][0:Nmrk_extended])
             weight = np.squeeze(weight_d[ids_random][0:Nmrk_extended])
             time   =   np.squeeze(time_d[ids_random][0:Nmrk_extended])
             vphi   =   np.squeeze(vphi_d[ids_random][0:Nmrk_extended])
             vR     =     np.squeeze(vR_d[ids_random][0:Nmrk_extended])
             vz     =     np.squeeze(vz_d[ids_random][0:Nmrk_extended])
             Nd_array     =     np.squeeze(Nd_array[ids_random][0:Nmrk_extended])

             ids    = 1 + np.arange(Nmrk_extended)    # just in case ASCOT expects markers = 1,2, 3, ...
             vtot   = np.sqrt(vR**2 + vphi**2 + vz**2)
             pitch  = vphi/vtot  # this needed for plots below

             weight = weight / np.sum(weight)
             Nmrk   = Nmrk_extended
             #pdb.set_trace(header="here we are")
             fn_daughter = stub + "_split_markers.pdf"
             with PdfPages(fn_daughter) as pdf:
                           
                 plt.close()
                 plt.figure(figsize=(8.,6.))
                 plt.plot(rhos, Nd_array, 'bo', ms=2, rasterized=True)
                 plt.xlabel('rho_poloidal')
                 plt.ylim(bottom=0.)
                 plt.xlim((0.,1.))
                 my_title = "Number of daughter markers, rho_crit = %7.4f"%(rho_crit)
                 plt.title(my_title)
                 pdf.savefig()

                 plt.close()
                 plt.figure(figsize=(8.,6.))
                 plt.hist(rhos, bins=30,  color='b', histtype='step')
                 plt.hist(rhos_parent, bins=30, color='k', histtype='step')
                 plt.xlabel("rho_poloidal")
                 plt.ylim(bottom=0.)
                 my_title = "Unweighted rhos, rho_crit = %7.4f"%(rho_crit) + "  k/b = parent/split"
                 plt.title(my_title)
                 pdf.savefig()
                 
                 plt.close()
                 plt.figure(figsize=(8.,6.))
                 plt.hist(rhos, bins=30, weights=weight, color='b', histtype='step')
                 plt.hist(rhos_parent, bins=30, weights=weight_parent, color='b', histtype='step')
                 plt.xlabel("rho_poloidal")
                 my_title = "weighted rhos, rho_crit = %7.4f"%(rho_crit) + "  k/b = parent/split"
                 plt.title(my_title)
                 pdf.savefig()

                 plt.close()
                 plt.figure(figsize=(4.5,7.))
                 plt.plot(R, z,'bo', ms=0.5, rasterized=True)
                 plt.xlabel("Rmajor [m]")
                 plt.ylabel("Z [m]")
                 my_title = "Marker [R,Z] after splitting with rho_crit = %7.4f"%(rho_crit)
                 plt.title(my_title)
                 pdf.savefig()
             print("   ... number markers before and after marker splitting: ",Nmrk_before_split, Nmrk)   
            
             
    # -----------------------------------------------------------------
    #  do not cull markers for Rmajor and add Rmajor to weighting
    
    elif set == 8:

         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         vol_zone      = aa_profiles["vol_zone"]

         S_tot = 0.
         for jj in range(alpha_source.size):
             S_tot = S_tot + alpha_source[jj] * vol_zone[jj]
         print(" total alpha emission rate: ", 1.e18*S_tot)
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()


         good_R     = R_temp
         good_z     = z_temp
         good_rhos  = rhos_temp

         R    = good_R[0:Nmrk]
         z    = good_z[0:Nmrk]
         rhos = good_rhos[0:Nmrk]
         
         #  define the marker weights based on the local alpha source rate
         
        
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = R[imark] * np.interp(rhos[imark], rho_array, alpha_source)
             
         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         # pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)  # removed 1/26/2020

         nn_pitch        = 10* Nmrk
         pitch_candidate = -1 + 2 * np.random.rand(nn_pitch)
         pitch_weights   = np.sqrt(1-pitch_candidate**2)
         rr_pitch        = np.random.rand(nn_pitch)
         pitch_out       = np.zeros(nn_pitch)
         mm_pitch        = 0
         
         for kk in range(nn_pitch):
             if(rr_pitch[kk] <= pitch_weights[kk]):
                pitch_out[mm_pitch] = pitch_candidate[kk]
                mm_pitch += 1
         pitch_out = pitch_out[0:mm_pitch-1]
         kk_good = (pitch_out >= pitch_min) & (pitch_out <= pitch_max)
         pitch_out = pitch_out[kk_good]
         mm_good   = pitch_out.size
         if(mm_good < Nmrk):
             print("  I could not find enough good pitch angles")
             exit()
         else:
             pitch = pitch_out[0:Nmrk]
         
         # ----------------------------------------------------------
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
         # pdb.set_trace()
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v
        

         # realization 1/23/2020 ... must make velocity random in
         # poloidal direction also

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)
 
    # ----------------------------------------------------------------

    elif set == 9:

         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         alpha_source  = aa_profiles["alpha_source"]/1.e18
         rho_array     = aa_profiles["rhosqrt"]
         rho_pol       = aa_profiles["rho_pol"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         # logic below gives probability of a marker proportional to its rmajor
         
         mysize = R_temp.size
         myrand = np.random.rand(mysize)

         good_R    = np.zeros(mysize)
         good_z    = np.zeros(mysize)
         good_rhos = np.zeros(mysize)

         R_big = psi_rmax   # must be bigger than any rmajor. was = 4. until 2/19/20
         mm_count = 0
         
         for kk in range(mysize):

             if( (R_temp[kk]/R_big) >= myrand[kk]):
                 good_R[mm_count]       = R_temp[kk]
                 good_z[mm_count]       = z_temp[kk]
                 good_rhos[mm_count]    = rhos_temp[kk]
                 mm_count += 1
         if(mm_count < Nmrk):
             print(" insufficient r-weighted markers")
             exit()
         # ------------------------------

         qq = mm_count-1
         
         candidate_source = np.zeros(qq)
         

         for kk in range(qq):
             #pdb.set_trace()
             candidate_source[kk] = np.interp(good_rhos[kk], rho_pol, alpha_source)

         source_max = np.max(candidate_source)
         source_sum = np.sum(candidate_source)

         candidate_probability = candidate_source / source_max

         mm_blessed = 0
         blessed_R  = np.zeros(qq)
         blessed_Z  = np.zeros(qq)
         blessed_rho = np.zeros(qq)
         blessed_source = np.zeros(qq)
                                              
         my_random = np.random.random(qq)

         for kk in range(qq):
                                              
             if (candidate_probability[kk] >= my_random[kk]):
                                              
                 blessed_R[mm_blessed]      = good_R[kk]
                 blessed_Z[mm_blessed]      = good_z[kk]
                 blessed_rho[mm_blessed]    = good_rhos[kk]
                 blessed_source[mm_blessed] = candidate_source[kk]
                                              
                 mm_blessed += 1
        
         blessed_R   = blessed_R[0:mm_blessed-1]
         blessed_Z   = blessed_Z[0:mm_blessed-1]
         blessed_rho = blessed_rho[0:mm_blessed-1]
         blessed_source = blessed_source[0:mm_blessed-1]

         source_sum   = np.sum(blessed_source)                                   
         weight       = S_tot * source_max / source_sum
         weight_array = weight * np.ones(mm_blessed-1)
                                              
         blessed_total = weight * np.sum(blessed_source)
         print("   blessed_total = ", blessed_total)
         print("   number of blessed points = ", mm_blessed-1)
                                              
         # 
         # ------------------------------------------------
         #  histogram 

         
         plt.figure(figsize=(7.,5.))
         plt.hist(blessed_rho, bins=50, weights=weight_array, rwidth=1,color='b', histtype='edge')
                          
         plt.title('scheme-1')
         plt.xlabel('rho')
         plt.ylabel('')
         plt.savefig(stub + '_marker_scheme_1.pdf')
         plt.close()
         plt.clf()

         exit()                                   

         #   ------------------------------------------------
         R    = good_R[0:Nmrk]
         z    = good_z[0:Nmrk]
         rhos = good_rhos[0:Nmrk]
         
         #R     =     R_temp[0:Nmrk]
         #z     =     z_temp[0:Nmrk]
         #rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c
        
         for imark in range(Nmrk):

             # until 11/16/19 we got weight from alpha_source * volume but
             # this is incorrect because we now obtain markers from a full
             # 2D spatial grid.
             
             weight[imark] = np.interp(rhos[imark], rho_array, alpha_source)
             
             #if rhos[imark] <= 0.6:
             #    weight[imark] = -0.00553 + 0.597 * rhos[imark] -1.295*(rhos[imark]**2) + 0.7557*(rhos[imark]**3)
             #else:
             #    weight[imark] = 0.1827 - 0.2777*rhos[imark] + 0.0963*(rhos[imark]**2)
             #
             #if (weight[imark] <= 0.):
             #    weight[imark] = 1.e-6

         total_weight = np.sum(weight)
         weight = weight / total_weight
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)

         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

              
         # pitch  = 0.999-1.998*np.random.rand(Nmrk)   # changed 12/7/19

         # pitch = pitch_max - (pitch_max - pitch_min)* np.random.rand(Nmrk)  # removed 1/26/2020

         nn_pitch        = 10* Nmrk
         pitch_candidate = -1 + 2 * np.random.rand(nn_pitch)
         pitch_weights   = np.sqrt(1-pitch_candidate**2)
         rr_pitch        = np.random.rand(nn_pitch)
         pitch_out       = np.zeros(nn_pitch)
         mm_pitch        = 0
         
         for kk in range(nn_pitch):
             if(rr_pitch[kk] <= pitch_weights[kk]):
                pitch_out[mm_pitch] = pitch_candidate[kk]
                mm_pitch += 1
         pitch_out = pitch_out[0:mm_pitch-1]
         kk_good = (pitch_out >= pitch_min) & (pitch_out <= pitch_max)
         pitch_out = pitch_out[kk_good]
         mm_good   = pitch_out.size
         if(mm_good < Nmrk):
             print("  I could not find enough good pitch angles")
             exit()
         else:
             pitch = pitch_out[0:Nmrk]
         
         # ----------------------------------------------------------
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
         # pdb.set_trace()
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v
        

         # realization 1/23/2020 ... must make velocity random in
         # poloidal direction also

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)
  
    # -----------------------------------------------------------------
    elif set == 4:

        # fixed grid of particles.  no randomness
        
         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         #alpha_source  = aa_profiles["alpha_source"]/1.e18
         #rho_array     = aa_profiles["rhosqrt"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         R     =  np.linspace(1.2, 2.4,  50)
         z     =  np.linspace(-1.0, 1.0, 50)
         pitch =  np.linspace(-0.4, 0.4,  9)

         Nmrk_local = 50*50*9

         rr_candidate       = np.zeros(Nmrk_local)
         zz_candidate       = np.zeros(Nmrk_local)
         pitch_candidate    = np.zeros(Nmrk_local)
         rho_birth          = np.zeros(Nmrk_local)
         rho_birth_new      = np.zeros(Nmrk_local)
         
         mm = 0
         
         for ii in range(50):
             for jj in range(50):
                 for kk in range(9):
                     
                     rr_candidate[mm]       = R[ii]
                     zz_candidate[mm]       = z[jj]
                     pitch_candidate[mm]    = pitch[kk]
                     rho                    = rhogeq_interp(rr_candidate[mm], zz_candidate[mm])     # interpolate to get rho

                     
                     rho_birth[mm]     = rho
                     

                     mm += 1
         #pdb.set_trace()
         ii_good = (rho_birth <= birth_rhomax) &  (rho_birth >= birth_rhomin)

         R     =    rr_candidate[ii_good]
         z     =    zz_candidate[ii_good]
         pitch = pitch_candidate[ii_good]
         rhos  =       rho_birth[ii_good]

         Nmrk = R.size
         print("   ... number of good markers = ", Nmrk)

         weight = (1./Nmrk) * np.ones(Nmrk)

         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = 4     * np.ones(ids.shape)
         charge = 2     * np.ones(ids.shape)
         anum   = 4     * np.ones(ids.shape)
         znum   = 2     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)

         try:                              # 12/3/2019
            if settings["my_ealpha"]:
               e_alpha = settings["my_ealpha"]
         except:
            e_alpha = 3.5e6

         energy = e_alpha * np.ones(ids.shape)
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        
         vR   = np.sqrt(1-pitch*pitch) * v
         vphi = pitch * v
         vz   = 0.* v
         v_poloidal = vR

         
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   from a Maxwellian of thermal ions
    
    elif set == 10:
        
         m           = settings["m"]    # maximum energy to consider is local Ti multiplied by m
         marker_mass = settings["marker_mass"]
         min_e_over_ti = m/Nmrk
         xx         = np.linspace(min_e_over_ti, m, Nmrk)
         
         weight     = (2./np.sqrt(np.pi)) * (m/Nmrk) * (xx**0.5) * np.exp(-xx)
         sum_weight  = np.sum(weight)
         norm_factor = 1./sum_weight
         print(" sum of weights = ", sum_weight, " So multiply weights by ", norm_factor)
         weight = weight * norm_factor
         total_weight = np.sum(weight)
         if (abs(total_weight -1.) > 1.e-5):
             print("   ... total_weight = ", total_weight, " ... this should not happen")
             sys.exit()
         
         fn_profiles   = settings["fn_profiles"]
         aa_profiles   = proc.read_sparc_profiles_new(fn_profiles)

         rho_array     = aa_profiles["rhosqrt"]
         ion_densities = aa_profiles["idensity"]
         ni            = ion_densities[:,0] + ion_densities[:,1]
         Ti            = aa_profiles["itemperature"]
         
         if(good_size < Nmrk):
             print("marker_sets:  good_size, Nmrk: ", good_size, Nmrk," insufficient points")
             sys.exit()

         # logic below gives probability of a marker proportional to its rmajor
         
         mysize = R_temp.size
         myrand = np.random.rand(mysize)

         good_R    = np.zeros(mysize)
         good_z    = np.zeros(mysize)
         good_rhos = np.zeros(mysize)

         R_big = psi_rmax   # must be bigger than any rmajor. was = 4. until 2/19/20
         mm_count = 0
         
         for kk in range(mysize):

             if( (R_temp[kk]/R_big) >= myrand[kk]):
                 good_R[mm_count]       = R_temp[kk]
                 good_z[mm_count]       = z_temp[kk]
                 good_rhos[mm_count]    = rhos_temp[kk]
                 mm_count += 1
                 
         if(mm_count < Nmrk):
             print(" insufficient r-weighted markers")
             exit()

         R    = good_R[0:Nmrk]
         z    = good_z[0:Nmrk]
         rhos = good_rhos[0:Nmrk]
         
         #R     =     R_temp[0:Nmrk]
         #z     =     z_temp[0:Nmrk]
         #rhos  =  rhos_temp[0:Nmrk]

         #  define the marker weights based on the local alpha source rate
         #  the following code is only applicable to v1c

         Ti_marker = np.zeros(Nmrk)
         energy    = np.zeros(Nmrk)
         ni_marker = np.zeros(Nmrk)
         
         for imark in range(Nmrk):
             
             Ti_marker[imark] = np.interp(rhos[imark], rho_array, Ti)
             energy[imark]    = Ti_marker[imark] * xx[imark]
             
             ni_marker[imark] = np.interp(rhos[imark], rho_array, ni)
             weight[imark]    = weight[imark] * ni_marker[imark]

         total_weight = np.sum(weight)
         weight       = weight / total_weight
         
         e_over_ti = energy/Ti_marker
         
         ids    = np.linspace(1,Nmrk,Nmrk)
        
         mass   = marker_mass     * np.ones(ids.shape)
         charge = 1     * np.ones(ids.shape)
         anum   = marker_mass     * np.ones(ids.shape)
         znum   = 1     * np.ones(ids.shape)
         phi    = 360   * np.random.rand(Nmrk)
         time   = 0.    * np.ones(ids.shape)


         # ++++++++++++++++++++++++++++++++++++++++++++++++++
         #  assign pitch angles with cosine distribution
         try:
             if settings["my_min_pitch"]:
                pitch_min = settings["my_min_pitch"]
         except:
             pitch_min = -0.999

         try:
             if settings["my_max_pitch"]:
                 pitch_max = settings["my_max_pitch"]
         except:
             pitch_max = 0.999

         nn_pitch        = 10* Nmrk
         pitch_candidate = -1 + 2 * np.random.rand(nn_pitch)
         pitch_weights   = np.sqrt(1-pitch_candidate**2)
         rr_pitch        = np.random.rand(nn_pitch)
         pitch_out       = np.zeros(nn_pitch)
         mm_pitch        = 0
         
         for kk in range(nn_pitch):
             if(rr_pitch[kk] <= pitch_weights[kk]):
                pitch_out[mm_pitch] = pitch_candidate[kk]
                mm_pitch += 1
         pitch_out = pitch_out[0:mm_pitch-1]
         kk_good = (pitch_out >= pitch_min) & (pitch_out <= pitch_max)
         pitch_out = pitch_out[kk_good]
         mm_good   = pitch_out.size
         if(mm_good < Nmrk):
             print("  I could not find enough good pitch angles")
             exit()
         else:
             pitch = pitch_out[0:Nmrk]
         
         # ----------------------------------------------------------
         
         gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        
         v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
         # pdb.set_trace()
         v_poloidal   = np.sqrt(1-pitch*pitch) * v
         vphi         = pitch * v

         gyro_angles = (2. * np.pi) * np.random.rand(Nmrk)
         
         vR = v_poloidal * np.cos(gyro_angles)
         vz = v_poloidal * np.sin(gyro_angles)
         
  
             
                            
         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                             
         plt.close(fig=None)
         plt.clf()
         plt.figure(figsize=(7.,5.))          
         plt.plot(e_over_ti,weight, 'ro', ms=1)
         plt.title('weight vs E_over_Ti')
         plt.savefig(stub + '_weight_E_over_Ti.pdf')
         
         plt.close(fig=None)
         plt.clf()
         plt.figure(figsize=(7.,5.))          
         plt.plot(e_over_ti, 'ro', ms=1)
         plt.title('Energy over local Ti')
         plt.savefig(stub + '_E_over_Ti.pdf')
             
         plt.close()
         plt.figure(figsize=(7.,5.))
         plt.plot(energy, v, 'ro')
         plt.title('velocity versus energy')
         plt.savefig(stub + '_vel_vs_energy.pdf')

         plt.close()
         plt.figure(figsize=(7.,5.))
         plt.plot(rho_array, Ti, 'ro')
         plt.title('Ion temperature vs rho')
         plt.xlabel('rho_pol')
         plt.ylabel('eV')
         plt.xlim((0.85, 1.))
         plt.savefig(stub + '_Ti_profile.pdf')

         plt.close()
         plt.figure(figsize=(7.,5.))
         plt.plot(rho_array, ni, 'ro')
         plt.title('ion density profile vs rho')
         plt.xlim((0.85, 1.))
         plt.xlabel('rho_pol')
         plt.ylabel('m^-3')
         plt.savefig(stub + '_ni_profile.pdf')

         plt.close()
         plt.figure(figsize=(7.,5.))
         plt.plot(rhos, weight, 'ro')
         plt.title('marker weight vs rho')
         plt.xlim((0.85, 1.))
         plt.xlabel('rho_pol')
         plt.ylabel('m^-3')
         plt.savefig(stub + '_marker_weights.pdf')

         plt.close()
         plt.figure(figsize=(7.,5.))
         plt.plot(ni_marker, weight, 'ro')
         plt.title('marker weight vs density')
         plt.xlabel('weight')
         plt.xlabel('m^-3')
         plt.savefig(stub + '_weight_vs_density.pdf')
    
    # ++++++++++++++++++++++++++++++++++++++++++++++
    #   make some common plots

    
    print("   ... starting common plots")

    filename_inputs = stub + '_marker_inputs.pdf'
    with PdfPages(filename_inputs) as pdf:
    
        nn = Nmrk

        try:
            if(settings['nplot_max']):
                nn = settings["nplot_max"]
        except:
            xx = 1.  # dummy
           
    
        plt.close(fig=None)
        plt.clf()
        plt.plot(rhos[0:nn], weight[0:nn], 'ro', ms=1, rasterized=True)
        plt.xlim([0.,1.])
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('rho [sqrt(norm poloidal flux)]')
        plt.savefig(stub + '_marker_weights_set_1.pdf')
        plt.ylim(bottom=0.)
        plt.tight_layout(pad=1)
        plt.close(fig=None)  # first one is corrupted
        plt.clf()
    
        plt.plot(rhos[0:nn], weight[0:nn], 'ro', ms=1, rasterized=1)
        xx_fake = np.linspace(0.,1., 100)
        yy_fake = np.zeros(100)
        plt.plot(xx_fake, yy_fake, 'k-')
        plt.xlim([0.,1.])
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('rho [sqrt(norm poloidal flux)]')
        pdf.savefig()
        #plt.savefig(stub + '_marker_weights_set_1.pdf')
    
        plt.close(fig=None)
        plt.clf()


        # ------------------------------------------------
        #  histogram of rmajor

        mmm = (np.abs(z[0:nn]) < 0.1)
        plt.figure(figsize=(7.,5.))
        plt.hist(R[0:nn][mmm], bins=50, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker Rmajor (abs(z)<0.1)')
        plt.xlabel('Rmajor')
        plt.ylabel('')
        plt.savefig(stub + '_marker_rmajor_hist.pdf')
        plt.close()
        plt.clf()

        # ------------------------------------------------
        #  histogram of z

        plt.figure(figsize=(7.,5.))
        plt.hist(z[0:nn], bins=100, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker Z')
        plt.xlabel('Z')
        plt.ylabel('')
        #plt.savefig(stub + '_marker_Z_hist.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()
    
        # ------------------------------------------------
        #  histogram of pitch angles

        plt.figure(figsize=(7.,5.))
        plt.hist(pitch[0:nn], bins=100, rwidth=1,color='c') #yyyy
        plt.title('Distribution of marker pitch angles')
        plt.xlabel('pitch angle')
        plt.ylabel('')
        #plt.savefig(stub + '_marker_pitch_hist.pdf')
        pdf.savefig()
        plt.close()
        plt.clf() 
                
        # -------------------------------------------------------------
        #  velocity vector directions

        #fig=plt.figure(figsize=(6., 6.))
        #vr_direction = vR[0:nn]/v_poloidal[0:nn]
        #vz_direction = vz[0:nn]/v_poloidal[0:nn] 
        #plt.plot(vr_direction, vz_direction, 'ro', ms=1,rasterized=1)
        #plt.xlim([-1.1,1.1])
        #plt.ylim([-1.1,1.1])
        #plt.xlabel('vR/v_poloidal')
        #plt.ylabel('vz/v_poloidal')
        #plt.title(' poloidal velocity direction')
        #plt.tight_layout()
        #plt.savefig(stub + '_marker_vpoloidal_dirn.pdf')
        #pdf.savefig()
        #plt.close()
        #plt.clf() 
    
        # -------------------------------------------------------------
        
        z_over_r = (psi_zmax - psi_zmin) / (psi_rmax - psi_rmin)
        xsize = 5.
        ysize = xsize * z_over_r
        fig=plt.figure(figsize=(xsize, ysize))
        plt.axis([psi_rmin, psi_rmax, psi_zmin, psi_zmax])
        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=True)
        plt.plot(R[0:nn],z[0:nn], 'ro', ms=0.4,zorder=2, rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        rho_contours_2 = [1.02, 1.04]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        plt.tight_layout(pad=1)
        #plt.savefig(stub + '_marker_rz_birth.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro', ms=3., fillstyle='none',zorder=2, rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
       #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((1.4,1.6))
        plt.ylim((0.90, 1.15))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        #plt.savefig(stub + '_marker_rz_birth_aa.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro', zorder=2, ms=3, fillstyle='none', rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((2.4,2.6))
        plt.ylim((-0.1,0.1))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        #plt.savefig(stub + '_marker_rz_birth_bb.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=1)
        plt.plot(R[0:nn],z[0:nn], 'ro',  ms=3, fillstyle='none',zorder=2,rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
        plt.clabel(cs, fontsize=10)
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
        plt.clabel(cs, fontsize=10)
        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.xlim((2.28,2.40))
        plt.ylim((0.35, 0.45))
        plt.title(' marker birth locations')
        plt.grid(axis='both', alpha=0.75)
        #plt.savefig(stub + '_marker_rz_birth_cc.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

    
        z_over_r = (psi_zmax - psi_zmin) / (psi_rmax - psi_rmin)
        xsize = 5.
        ysize = xsize * z_over_r
        fig=plt.figure(figsize=(xsize, ysize))
        plt.axis([psi_rmin, psi_rmax, psi_zmin, psi_zmax])
        #plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=2, rasterized=1)
    
        rho_contours = np.linspace(0.1, 1.0, 10)
        #rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        #pdb.set_trace()
        cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=1.0, colors='g',zorder=1)
        plt.clabel(cs)

        plt.xlabel('R [m]')
        plt.ylabel('z [m]')
        plt.title(' contours of rho-pol')
        plt.tight_layout(pad=2)
        #plt.savefig(stub + '_equilibrium_10.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.close(fig=None)
        plt.plot(R[0:nn], pitch[0:nn], 'ro', ms=2,rasterized=True)
        plt.xlabel('Rmajor [m]')
        plt.title('pitch of markers')
        #plt.savefig(stub + '_pitch_Rmajor.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()


        plt.plot(R[0:nn], pitch[0:nn], 'ro', ms=2,rasterized=True)
        plt.xlabel('Rmajor [m]')
        plt.title('pitch of markers')
        #plt.savefig(stub + '_pitch_Rmajor.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()

        plt.close(fig=None)
        plt.plot(z[0:nn], pitch[0:nn], 'ro', ms=2, rasterized=True)
        plt.xlabel('elevation [m]')
        plt.title('pitch of markers')
        #plt.savefig(stub + '_pitch_Z.pdf')
        pdf.savefig()
        plt.close()
        plt.clf()
        
    print("   ... I finished plots, now writing marker information to .h5 file. \n")
    #pdb.set_trace()
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    out={}
    out['z'] = z
    out['R'] = R 
    out['markers'] = Nmrk

    print('   ... I have computed and written initial conditions of markers.')
    print("   ... total time to create markers = ", clock.time() - t_very_start, " sec \n")
    
    return out


def define_prt_markers_04(fn_hdf5, set, Nrho, rhomin, rhomax, Npitch, pitch_min, pitch_max,  desc=None):
    """
    define_prt_markers: 
       hdf5_fn     name of hdf5 file
       set         set number = 1, 2, 3, ...
       desc        description
    """

    Nmrk = Nrho * Npitch

    R     = np.zeros(Nmrk)
    pitch = np.zeros(Nmrk)

    print("   ... define_prt_markers_04:  Nrho, Npitch, Nmrk: ", Nrho, Npitch, Nmrk)
    
    if set==1:
       
        R0     = 1.85
        aa     = 0.57
        Rstart = R0 + rhomin * aa   
        Rend   = R0 + rhomax * aa

        nn = 0
        
        for irho in range(Nrho):
            
            for ipitch in range(Npitch):

                if(Nrho == 1):
                    this_R = Rstart
                else:
                    this_R = Rstart + (Rend-Rstart)*irho/(Nrho-1)

                R[nn] = this_R

                if(Npitch ==1):
                    this_pitch = pitch_min
                else:
                    this_pitch = pitch_min + (pitch_max-pitch_min)*ipitch/(Npitch-1)

                pitch[nn] = this_pitch
                nn = nn + 1
            

        z      = np.zeros((Nmrk))
        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
                
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?                 

    print("")
    print("=======================================================")
    print("")
    print("    values for marker-particle set ", set)
    print("")
    print("   Nmrk: {0:6d}".format(Nmrk))
    print("   mass:   ", mass)
    print("   charge: ", charge)
    print("   anum:   ", anum)
    print("   znum:   ", znum)
    print("   phi:    ", phi)
    print("   weight: ", weight)
    print("   time:   ", time)
    print("   energy: ", energy)
    print("   pitch:  ", pitch)
    print("   theta:  ", theta)
    print("   gamma:  ", gamma)
    print("   v:      ", v)
    print("   vR:     ", vR)
    print("   vphi:   ", vphi)
    print("   vz:     ", vz)
    print("   R:      ", R)
    print("   z:      ", z)
    print("")

    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    out={}
    out['z'] = z
    out['R'] = R 
    out['markers'] = Nmrk
    
    return out


def define_prt_markers_05(fn_hdf5, fn_geqdsk, fn_hdf5_parent, set, end_condition, settings, desc=None):
    """    copies endstate of parent run into new ensemble of birth-markers

    set-1:  only copy markers that hit the wall or the LCFS
    set-2:  copy all markers
    set-3:  copy markers from a *list* of parent files

    end_condition:  (mandatory for set-1)
        "wall_only"        --> copy only markers that hit the wall
        "rhomax_only"      --> copy only markers whose rho exceeds rhomax
        "wall_and_rhomax"  -->  copy markers that either hit wall or exceed rhomax

    settings["vmult"]  --> multiply all velocities by vmult  (optional)
    settings["tmin"]   --> for set=1, discard markers with t < tmin   4/12/2021
    settings["tmax"]   --> for set=1, discard markers with t > tmax   4/12/2021
    """
    print("\n inside define_prt_markers_05: set = ", set)
    print("                               fn_hdf5_parent = ", fn_hdf5_parent)
    
    
    #  copies endstate of parent run into markers for daughter run
    #  sds 12/18/2019

    do_rasterized = True
    #geq_strings  = fn_geqdsk.split('.')

    geq_strings  = fn_hdf5.split('.')
    stub         = geq_strings[0] 
    
    # ---------------------------------------------
    #   get equilibrium and (R,Z) of LCFS

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    eq_index = settings["index"]
    rlcfs    =  gg.equilibria[eq_index].rlcfs
    zlcfs    =  gg.equilibria[eq_index].zlcfs

    # -------------------------------------------------------
          
    if(set == 1):
    
        new_or_old = process_ascot.old_or_new(fn_hdf5_parent)
    
        if(new_or_old == 0):
            aa = process_ascot.myread_hdf5(fn_hdf5_parent)
        elif(new_or_old == 1):
            aa = process_ascot.myread_hdf5_new(fn_hdf5_parent)
        
        # aa = process_ascot.myread_hdf5(fn_hdf5_parent)

        endcond = aa["endcond"]
        
        if(end_condition == "wall_and_rhomax"):
            ii_end = (endcond == 8) ^ (endcond == 32) ^ (endcond==544)
            name_end = " wall-hit and rhomax"
        elif (end_condition == "wall_only"):
            ii_end = (endcond == 8)
            name_end = "wall-hit only"
        elif (end_condition == "rhomax_only"):
            ii_end = (endcond == 32) ^ (endcond==544)
            name_end = "rhomax only"
        else:
            print(" marker_sets:  name_end must be rhomax_only, wall-hit only, or wall-and_rhomax")
            exit()

        if(settings['sim_type'] == 'go'):   ##  new 10/7/2020

            print("   ... as instructed, I will use: rprt, zprt, phiprt\n")
            time   =   aa["time"][ii_end]
            anum   =   aa["anum"][ii_end]
            charge =   aa["charge"][ii_end]
            mass   =   aa["mass"][ii_end]
            R      =   aa["rprt_end"][ii_end]    ## new
            phi    =   aa["phiprt_end"][ii_end] % 360. ## new
            z      =   aa["zprt_end"][ii_end]    ## new    
            vR     =   aa["vr"][ii_end]          
            vphi   =   aa["vphi"][ii_end]        
            vz     =   aa["vz"][ii_end]          

            znum   =    aa["znum"][ii_end]        
            weight =    aa["weight"][ii_end]
        
            r_ini       = aa["r_ini"][ii_end]
            z_ini       = aa["z_ini"][ii_end]
            rprt_ini    = aa["rprt_ini"][ii_end]
            zprt_ini    = aa["zprt_ini"][ii_end]
        
            R_marker    = aa["marker_r"][ii_end]
            z_marker    = aa["marker_z"][ii_end]
            phi_marker  = aa["marker_phi"][ii_end] % 360.
            vR_marker   = aa["marker_vr"][ii_end]
            vphi_marker = aa["marker_vphi"][ii_end]
            vz_marker   = aa["marker_vz"][ii_end]

        else:

            time   =    aa["time"][ii_end]
            anum   =    aa["anum"][ii_end]
            charge =  aa["charge"][ii_end]
            mass   =    aa["mass"][ii_end]
            R      =   aa["r_end"][ii_end]
            phi    = aa["phi_end"][ii_end]     % 360.   # added 10/18/2020
            z      =   aa["z_end"][ii_end]
            vR     =      aa["vr"][ii_end]
            vphi   =    aa["vphi"][ii_end]
            vz     =      aa["vz"][ii_end]
            znum   =    aa["znum"][ii_end]
            weight =  aa["weight"][ii_end]
        
            r_ini       = aa["r_ini"][ii_end]
            z_ini       = aa["z_ini"][ii_end]
        
            R_marker    = aa["marker_r"][ii_end]
            z_marker    = aa["marker_z"][ii_end]
            phi_marker  = aa["marker_phi"][ii_end]  %360.  # added 10/18/2020
            vR_marker   = aa["marker_vr"][ii_end]
            vphi_marker = aa["marker_vphi"][ii_end]
            vz_marker   = aa["marker_vz"][ii_end]

        # ++++++++++++++++++++++++++++++++
        #  exclude markers by time
        
        try:
            tmin = settings["tmin"]
        except:
            tmin = -1.
        try:
            tmax = settings["tmax"]
        except:
            tmax = 1.e20

        ntotal = vz_marker.size

        ii_tmin_tmax = (time >= tmin) & (time <= tmax)

        time        =   time[ii_tmin_tmax]
        anum        =   anum[ii_tmin_tmax]
        charge      =   charge[ii_tmin_tmax]
        mass        =   mass[ii_tmin_tmax]   
        R           =   R [ii_tmin_tmax]
        phi         =   phi[ii_tmin_tmax]
        z           =   z[ii_tmin_tmax] 
        vR          =   vR[ii_tmin_tmax]    
        vphi        =   vphi[ii_tmin_tmax]
        vz          =   vz[ii_tmin_tmax]     
        znum        =   znum[ii_tmin_tmax] 
        weight      =   weight[ii_tmin_tmax]
        r_ini       =   r_ini[ii_tmin_tmax]
        z_ini       =   z_ini[ii_tmin_tmax]
        R_marker    =   R_marker[ii_tmin_tmax]
        z_marker    =   z_marker[ii_tmin_tmax]
        phi_marker  =   phi_marker[ii_tmin_tmax]
        vR_marker   =   vR_marker[ii_tmin_tmax]
        vphi_marker =   vphi_marker[ii_tmin_tmax]
        vz_marker   =   vz_marker[ii_tmin_tmax]
 
        nngood = vz_marker.size
        
        print("\n define_prt_markers_05:")
        print("   tmin = ", tmin)
        print("   tmax = ", tmax)
        print("   number markers reduced from ", ntotal, " to ", nngood)
       
        #pdb.set_trace()
        # -------------------------------------------------
        #  optinally, copy data from marker to starting
        #  position if rho_ini > 1
        #  sds 5/24/2020
        #pdb.set_trace()
        if(settings["fix_ini"] !=1):
            print("\n\n define_prt_markers_05:  I will *not* replace ini with marker for rho_ini>1 \n")
        elif(settings["fix_ini"] == 1):

            print("\n\n define_prt_markers_05:  I will replace ini with marker for rho_ini>1 \n")
            rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index)

            nn = r_ini.size
            rho_markers = np.zeros(nn)
            rho_ends    = np.zeros(nn)
            for jj in range(nn):
                rho_markers[jj] = rho_interpolator(R_marker[jj], z_marker[jj])
                rho_ends[jj]    = rho_interpolator(R[jj], z[jj])
            rho_biggest = np.max(rho_markers)
            print("\n\n define_prt_markers_05:  rho_biggest (of markers) = ", rho_biggest)

            
            if(rho_biggest > 1):
                print("\n warning:  some rho values exceed unity: \n")
                iijj = (rho_markers > 1)
                print(rho_markers[iijj])
                print ("")
              
            rho_ini = np.zeros(R.size)
            nn      = rho_ini.size

            for jj in range(nn):
                rho_ini[jj] = rho_interpolator(r_ini[jj], z_ini[jj])

            ii_fix = (rho_ini > 0.999)         # ideally, should be unity
            
            delta_r    = R[ii_fix]    - R_marker[ii_fix]   # debug purposes only
            delta_z    = z[ii_fix]    - z_marker[ii_fix]   
            delta_phi  = phi[ii_fix]  - phi_marker[ii_fix]
            delta_vr   = (vR[ii_fix]   - vR_marker[ii_fix]) / 1.e6
            delta_vz   = (vz[ii_fix]   - vz_marker[ii_fix]) / 1.e6
            delta_vphi = (vphi[ii_fix] - vphi_marker[ii_fix]) / 1.e6
            #pdb.set_trace()
            plt.close()
            plt.figure(figsize=(7.,5.))
            plt.close() 
            plt.plot(delta_r, delta_z, 'bo', ms=2)
            plt.title('delta_z vs delta_r')
            plot_filename = stub + '_delta_z_vs_delta_r.pdf'
            plt.savefig(plot_filename)

            plt.close()  # do it again
            plt.figure(figsize=(7.,5.))
            plt.close() 
            plt.plot(delta_r, delta_z, 'bo', ms=2)
            plt.title('delta_z vs delta_r')
            plot_filename = stub + '_delta_z_vs_delta_r.pdf'
            plt.savefig(plot_filename)
            
            plt.close()
            plt.figure(figsize=(7.,5.))
            plt.plot(delta_r, delta_phi, 'bo', ms=2)
            plt.title('delta_phi vs delta_r')
            plot_filename = stub + '_delta_phi_vs_delta_r.pdf'
            plt.savefig(plot_filename)

            plt.close()
            plt.plot(delta_vphi, delta_vr, 'bo', ms=2)
            plt.title('delta_vr vs delta_vphi (e6)')
            plot_filename = stub + '_delta_vr_vs_delta_vphi.pdf'
            plt.savefig(plot_filename)

            plt.subplot(2,2,4)
            plt.plot(delta_vphi, delta_vz, 'bo', ms=2)
            plt.title('delta_vz vs delta_vphi (e6)')
            plot_filename = stub + '_delta_vs_vs_delta_vphi.pdf'
            plt.tight_layout(pad=1)
            plt.savefig(plot_filename)
            
            #  now overwrite data with marker data
            
            R[ii_fix]    =   R_marker[ii_fix]
            z[ii_fix]    =   z_marker[ii_fix]
            phi[ii_fix]  = phi_marker[ii_fix]

            vR[ii_fix]   =   vR_marker[ii_fix]   
            vphi[ii_fix] = vphi_marker[ii_fix] 
            vz[ii_fix]   =   vz_marker[ii_fix]

            nn_changed   = vz_marker[ii_fix].size
            print("   ... I replaced %5d endpoints with marker points \n" %(nn_changed))
            
            # ------------------------------------------
            #   check that rho <=1 for all
        
            rho_check = np.zeros(vR.size)
            for jj in range(vR.size):
               rho_check[jj] = rho_interpolator(R[jj], z[jj]) 
            jj_check = (rho_check > 1.00)
            bad_rhos = rho_check[jj_check]
            nbad = bad_rhos.size
            if (nbad> 0):
                print("\n \n  define_prt_markers_05: we have ", nbad, " bad rhos > 1 \n")
                worst_one = np.max(bad_rhos)
                print("   largest rho is: ", worst_one)
                #pdb.set_trace()
            else:
                dummy = 0.

        else:
                dummy = 0.

 
        # ----------------------------------------------
        #   1/24/2020:  re-normalize weights to unity

        weight_total = np.sum(weight)
        weight = weight/weight_total

        vtot  = np.sqrt( vR*vR + vphi*vphi + vz*vz)
        pitch = vphi/vtot

        # compute marker energy for plotting only
        
        proton_mass = 1.67e-27
        amu_mass    = 1.66053904e-27
        q_e         = 1.602e-19   
        ekev        = 0.001  * amu_mass * mass * vtot**2/ (2.*1.602e-19)

        
        Nmrk = anum.size
        time = np.zeros(Nmrk)
        ids  = np.linspace(1,Nmrk, Nmrk)

        print("\n  define_prt_markers_05:  parent run is: ", fn_hdf5_parent)
        print(" markers selected from end-conditions:     ", name_end,"\n")
        print(" Number of markers from parent: ", Nmrk)

        
        
        try:
            max_markers = settings["max_markers"]
        
            if (max_markers < mass.size):
                
                ids    =    ids[0:max_markers]
                mass   =   mass[0:max_markers]
                charge = charge[0:max_markers]
                R      =      R[0:max_markers]
                phi    =    phi[0:max_markers]
                z      =      z[0:max_markers]
                vR     =     vR[0:max_markers]
                vphi   =   vphi[0:max_markers]
                vz     =     vz[0:max_markers]
                anum   =   anum[0:max_markers]
                znum   =   znum[0:max_markers]
                weight = weight[0:max_markers]
                time   =   time[0:max_markers]
                pitch  =  pitch[0:max_markers]
                vtot   =   vtot[0:max_markers]
                ekev   =   ekev[0:max_markers]
                
                weight_total = np.sum(weight)
                weight = weight/weight_total      # renormalize
                
                Nmrk   = max_markers
                
                print("   ... as instructed, I have reduced number of parent markers to ", max_markers, "\n")
                
                
        except:
            print("   ... max_markers not present in settings, so I will use all markers")
    
    # -------------------------------------------------------------------------

    elif set == 2:
        
        name_end = "all parent markers"
        
        aa = process_ascot.myread_hdf5_markers(fn_hdf5_parent)

        anum   =  aa["marker_anum"]
        charge =  aa["marker_charge"]
        mass   =  aa["marker_mass"]
        R      =  aa["marker_r"]
        phi    =  aa["marker_phi"]
        z      =  aa["marker_z"]
        vR     =  aa["marker_vr"]
        vphi   =  aa["marker_vphi"]
        vz     =  aa["marker_vz"]
        znum   =  aa["marker_znum"]
        weight =  aa["marker_weight"]
        time   =  aa["marker_time"]
        ids    =  aa["marker_id"]

        # ----------------------------------------------
        #   1/24/2020:  re-normalize weights to unity

        weight_total = np.sum(weight)
        weight = weight/weight_total

        vtot  = np.sqrt( vR*vR + vphi*vphi + vz*vz)
        pitch = vphi/vtot

        # compute marker energy for plotting only
        
        proton_mass = 1.67e-27
        amu_mass    = 1.66053904e-27
        q_e         = 1.602e-19   
        ekev        = 0.001  * amu_mass * mass * vtot**2/ (2.*1.602e-19)

        
        Nmrk = anum.size
 

        print("\n  define_prt_markers_05:  parent run is: ", fn_hdf5_parent)
        print(" markers selected from end-conditions:     ", name_end,"\n")
        print(" Number of markers from parent: ", Nmrk)

    # -------------------------------------------------------------------------
    #    "fn_hdf5_parent" is now a LIST of filenames
    
    elif set == 3:

        anum   =  np.empty(0)
        charge =  np.empty(0)
        mass   =  np.empty(0)
        R      =  np.empty(0)
        phi    =  np.empty(0)
        z      =  np.empty(0)
        vR     =  np.empty(0)
        vphi   =  np.empty(0)
        vz     =  np.empty(0)
        znum   =  np.empty(0) 
        weight =  np.empty(0)
        time   =  np.empty(0)
        ids    =  np.empty(0)

        name_end = "all parent markers"

        for fn in fn_hdf5_parent:
            
            aa = process_ascot.myread_hdf5_markers(fn)

            anum   =  np.concatenate((anum,   aa["marker_anum"]))
            charge =  np.concatenate((charge, aa["marker_charge"]))
            mass   =  np.concatenate((mass,   aa["marker_mass"]))
            R      =  np.concatenate((R,      aa["marker_r"]))
            phi    =  np.concatenate((phi,    aa["marker_phi"]))
            z      =  np.concatenate((z,      aa["marker_z"]))
            vR     =  np.concatenate((vR,     aa["marker_vr"]))
            vphi   =  np.concatenate((vphi,   aa["marker_vphi"]))
            vz     =  np.concatenate((vz,     aa["marker_vz"]))
            znum   =  np.concatenate((znum,   aa["marker_znum"]))
            weight =  np.concatenate((weight, aa["marker_weight"]))  # note: no renormalization of weights
            time   =  np.concatenate((time,   aa["marker_time"]))

            print("   file: ", fn, "   ... sum of weights = ", np.sum(aa["marker_weights"]))
            
        Nmrk = anum.size

        ids = np.linspace(1,Nmrk, Nmrk).astype(int)

        vtot  = np.sqrt( vR*vR + vphi*vphi + vz*vz)
        pitch = vphi/vtot

        # compute marker energy for plotting only
        
        proton_mass = 1.67e-27
        amu_mass    = 1.66053904e-27
        q_e         = 1.602e-19   
        ekev        = 0.001  * amu_mass * mass * vtot**2/ (2.*1.602e-19)

        print("\n  define_prt_markers_05:  parent runs are: ", fn_hdf5_parent)
        print(" markers selected from end-conditions:     ", name_end,"\n")
        print(" Number of markers from parents: ", Nmrk)


            
        fn_hdf5_parent = "(multiple)"   # override name to avoid problems in plot titles ...
        
    # -------------------------------------------------------------------------
    try:
        if(settings["vmult"] > 0):
            vmult = settings["vmult"]
            vR = vR * vmult
            vz = vz * vmult
            vphi = vphi * vmult
            print("\n NOTE!!!  in define_prt_markers_05: I have multiplied the marker velocities by a factor: ", vmult)
    except:
        xdummy = 0.

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    try:
        if (settings["cull_factor"]):

            nn_parent = R.size
            randoms   = np.random.random(nn_parent)
            cull_factor = settings["cull_factor"]
            rcheck    = 1./cull_factor

            ii_good = 0
            
            for jj in range(nn_parent):

                if ( randoms[jj] <= rcheck):

                    R[ii_good]      =  R[jj]
                    z[ii_good]      =  z[jj]
                    phi[ii_good]    =  phi[jj]
                    vR[ii_good]     =  vR[jj]
                    vphi[ii_good]   =  vphi[jj]
                    vz[ii_good]     =  vz[jj]
                    mass[ii_good]   =  mass[jj]
                    charge[ii_good] =  charge[jj]
                    anum[ii_good]   =  anum[jj]
                    znum[ii_good]   =  znum[jj]
                    time[ii_good]   =  time[jj]
                    weight[ii_good] =  weight[jj]   #* cull_factor

                    ii_good = ii_good + 1
                    
            #  shorten the arrays
                    
            R      =  R[0:ii_good]
            z      =  z[0:ii_good]
            phi    =  phi[0:ii_good]
            vR     =  vR[0:ii_good]
            vphi   =  vphi[0:ii_good]
            vz     =  vz[0:ii_good]
            mass   =  mass[0:ii_good]
            charge =  charge[0:ii_good]
            anum   =  anum[0:ii_good]
            znum   =  znum[0:ii_good]
            time   =  time[0:ii_good]
            weight =  weight[0:ii_good]
            weight =  weight / np.sum(weight)

            nn_daughter = R.size
            Nmrk        = nn_daughter
            ids         = np.linspace(1,nn_daughter, nn_daughter).astype(int)
        
            #weight_daughter = np.sum(weight)

            #ratio = weight_parent / weight_daughter

            #weight = weight * weight_parent / weight_daughter
                
            print("   parent:  number of non-lost markers: ", nn_parent)
            print("   daughter:  number of markers:        ", nn_daughter)
    except:
        xdummy = 0.
   # ++++++++++++++++++++++++++++++++++++++++++++++++
    try:
        if (settings["max_markers"]):

            ii_good = settings["max_markers"]
            
            #  shorten the arrays
                    
            R      =  R[0:ii_good]
            z      =  z[0:ii_good]
            phi    =  phi[0:ii_good]
            vR     =  vR[0:ii_good]
            vphi   =  vphi[0:ii_good]
            vz     =  vz[0:ii_good]
            mass   =  mass[0:ii_good]
            charge =  charge[0:ii_good]
            anum   =  anum[0:ii_good]
            znum   =  znum[0:ii_good]
            time   =  time[0:ii_good]
            weight =  weight[0:ii_good]
            weight =  weight / np.sum(weight)

            nn_daughter = R.size
            Nmrk        = nn_daughter
            ids         = np.linspace(1,nn_daughter, nn_daughter).astype(int)
        
            #weight_daughter = np.sum(weight)

            #ratio = weight_parent / weight_daughter

            #weight = weight * weight_parent / weight_daughter
                
            print("   parent:  number of non-lost markers: ", nn_parent)
            print("   daughter:  number of markers:        ", nn_daughter)
    except:
        xdummy = 0.

        
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)
    
    print("   ... define_prt_markers_05:  about to make some plots")
    
    # ----------------------------------------------------------------------------------

    nn = Nmrk

    try:
        if(settings["nplot_max"]):
            if(nn > settings["nplot_max"]):
                nn = settings["nplot_max"]
    except:
        xdummy = 0.
        
    plt.close()

    
    plt.plot(R[0:nn], z[0:nn], 'go', ms=1, fillstyle='none', rasterized=do_rasterized)
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized)
    my_title = "R,Z marker starts from parent: " + fn_hdf5_parent + " Nmrk: " + str(Nmrk)
    plt.title(my_title)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('z [m]')
    my_stub = fn_hdf5.split('.')[0]
    plot_filename = my_stub + '_marker_starts_daughter_RZ.pdf'
    plt.savefig(plot_filename)

    plt.close()
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized, zorder=1)
    plt.plot(R[0:nn], z[0:nn], 'go', ms=1, fillstyle='none', rasterized=do_rasterized)
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized)
    my_title = "R,Z marker starts from parent: " + fn_hdf5_parent + " Nmrk: " + str(Nmrk)
    plt.title(my_title)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('z [m]')
    plot_filename = my_stub + '_marker_starts_daughter_RZ.pdf'
    plt.savefig(plot_filename)

    plt.close()
    plt.plot(R[0:nn], pitch[0:nn], 'go', ms=1, fillstyle='none', rasterized=do_rasterized)
    my_title = "R,pitchangle marker starts from parent: " + fn_hdf5_parent + " Nmrk: " + str(Nmrk)
    plt.title(my_title)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('pitch')
    plot_filename = stub + '_marker_starts_daughter_R_pitch.pdf'
    plt.savefig(plot_filename)

    plt.close()
    plt.plot(z[0:nn], pitch[0:nn], 'go', ms=1, fillstyle='none', rasterized=do_rasterized, zorder=1)
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized)
    my_title = "z,pitchangle marker starts from parent: " + fn_hdf5_parent + " Nmrk: " + str(Nmrk)
    plt.title(my_title)
    plt.xlabel('z [m]')
    plt.ylabel('pitch')
    plot_filename = my_stub + '_marker_starts_daughter_z_pitch.pdf'
    plt.savefig(plot_filename)
    plt.close()

    plt.close()
    plt.plot(R[0:nn], ekev[0:nn], 'go', ms=1, fillstyle='none', rasterized=do_rasterized, zorder=1)
    my_title = "Ekev marker starts from parent: " + fn_hdf5_parent + " Nmrk: " + str(Nmrk)
    plt.title(my_title)
    plt.xlabel('R [m]')
    plt.ylabel('keV')
    plot_filename = my_stub + '_marker_starts_daughter_z_pitch.pdf'
    plt.savefig(plot_filename)
    plt.close()

    # -----------------
    #  borrowed plots
    # -----------------

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
    z_over_r = (psi_zmax - psi_zmin) / (psi_rmax - psi_rmin)
    xsize = 5.
    ysize = xsize * z_over_r
    fig=plt.figure(figsize=(xsize, ysize))
    plt.axis([psi_rmin, psi_rmax, psi_zmin, psi_zmax])

    plt.plot(R[0:nn],z[0:nn], 'ro', ms=0.05,zorder=2)
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=1, zorder=10, rasterized=1)
    rho_contours = np.linspace(0.1, 0.9, 9)
    rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
    rho_contours_2 = [1.02, 1.04]
    contours_2 = [1.02, 1.04]
    #pdb.set_trace()
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
    plt.clabel(cs, fontsize=10)
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
    plt.clabel(cs, fontsize=10)
    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    plt.title(' marker birth locations')
    plt.grid(axis='both', alpha=0.75)
    plt.savefig(stub + '_marker_rz_birth.pdf')
    plt.close()
    plt.clf()

    plt.plot(R[0:nn],z[0:nn], 'ro', ms=1.5, fillstyle='none',zorder=2, rasterized=1)
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=1, zorder=10, rasterized=1)
    rho_contours = np.linspace(0.1, 0.9, 9)
    rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
    contours_2 = [1.02, 1.04]
    #pdb.set_trace()
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
    plt.clabel(cs, fontsize=10)
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
    plt.clabel(cs, fontsize=10)
    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    plt.xlim((1.4,1.6))
    plt.ylim((0.90, 1.15))
    plt.title(' marker birth locations')
    plt.grid(axis='both', alpha=0.75)
    plt.savefig(stub + '_marker_rz_birth_aa.pdf')
    plt.close()
    plt.clf()
    
    plt.plot(R[0:nn],z[0:nn], 'ro', zorder=2, ms=1.5, fillstyle='none')
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=1, zorder=10, rasterized=10)
    rho_contours = np.linspace(0.1, 0.9, 9)
    rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
    contours_2 = [1.02, 1.04]
    #pdb.set_trace()
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
    plt.clabel(cs, fontsize=10)
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
    plt.clabel(cs, fontsize=10)
    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    plt.xlim((2.4,2.6))
    plt.ylim((-0.1,0.1))
    plt.title(' marker birth locations')
    plt.grid(axis='both', alpha=0.75)
    plt.savefig(stub + '_marker_rz_birth_bb.pdf')
    plt.close()
    plt.clf()

    plt.plot(rlcfs, zlcfs, 'k-', linewidth=1, zorder=10, rasterized=1)
    plt.plot(R[0:nn],z[0:nn], 'ro',  ms=1.5, fillstyle='none',zorder=2)
    rho_contours = np.linspace(0.1, 0.9, 9)
    rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
    contours_2 = [1.02, 1.04]
    #pdb.set_trace()
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=0.6, colors='b',zorder=1)
    plt.clabel(cs, fontsize=10)
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours_2, linewidths=0.6, colors='gold',zorder=1)
    plt.clabel(cs, fontsize=10)
    plt.xlabel('R [m]')
    plt.ylabel('z [m]')
    plt.xlim((2.28,2.40))
    plt.ylim((0.35, 0.45))
    plt.title(' marker birth locations')
    plt.grid(axis='both', alpha=0.75)
    plt.savefig(stub + '_marker_rz_birth_cc.pdf')
    plt.close()
    plt.clf()


def define_prt_markers_06(fn_hdf5, fn_geqdsk, fn_markers, set, settings, desc=None):

    #  copies initial state from a flat-ascii file from Gerrit
    #  format of file must be:  R, phi, z, vR, vphi, vz
    #  sds 2/12/2020

    do_rasterized = True
    #geq_strings   = fn_geqdsk.split('.')
    geq_strings   = fn_hdf5.split('.')
    stub          = geq_strings[0]
    
    # ---------------------------------------------
    #   get equilibrium and (R,Z) of LCFS

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    eq_index = settings["index"]
    
    rlcfs    =  gg.equilibria[eq_index].rlcfs
    zlcfs    =  gg.equilibria[eq_index].zlcfs

    rmin = np.min(rlcfs)
    rmax = np.max(rlcfs)
    ymin = np.min(zlcfs)
    ymax = np.max(zlcfs)
    
    # -------------------------------------------------------

    if(set == 1):

        aa = read_any.read_any_file(fn_markers)
        
        R      =  aa[:,0]
        phi    =  aa[:,1]
        z      =  aa[:,2]
        vR     =  aa[:,3]
        vphi   =  aa[:,4]
        vz     =  aa[:,5]
        weight =  aa[:,6]   # still waiting on this one

        Nmrk   = R.size

        #  cannot allow negative weights ... causes ASCOT
        #  to label those markers as 'aborted'
        #  sds 2/17/2020

        Number_corrected = 0
        for jj in range(Nmrk):
            if(weight[jj] < 1.e-9):
                weight[jj] = 1.e-9
                Number_corrected += 1

        print('Marker sets:  number of markers whose weight was made positive: ', Number_corrected)
                
        anum   =  4  *np.ones(Nmrk)
        charge =  2 * np.ones(Nmrk)
        mass   =  4 * np.ones(Nmrk)
        znum   =  2 * np.ones(Nmrk)
        time   =  np.zeros(Nmrk)
        ids    =  np.linspace(1,Nmrk, Nmrk)
        ids    =  ids.astype(int)
            
        weight_total = np.sum(weight)
        weight       = weight/weight_total   # renormalize so that total weight = 1.00

        print(" Number of markers from parent: ", Nmrk)
        
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    # ----------------------------------------------------------------------------------

    xsize = 4.
    ysize = xsize * (ymax-ymin)/(rmax-rmin)
    
    plt.close()
    plt.figure(figsize=(xsize,ysize))
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized)
    plt.plot(R, z, 'go', ms=0.5, fillstyle='none', rasterized=do_rasterized)
    my_title = "R,Z marker starts from parent: " 
    plt.title(my_title)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('z [m]')
    plot_filename = stub + '_marker_starts_daughter_RZ.pdf'
    plt.savefig(plot_filename)

    #  need to do it twice. not sure why

    
    plt.close()
    plt.figure(figsize=(xsize,ysize))
    plt.plot(rlcfs, zlcfs, 'k-', linewidth=2,    rasterized=do_rasterized)
    plt.plot(R, z, 'go', ms=0.5, fillstyle='none', rasterized=do_rasterized)
    my_title = "R,Z marker starts from parent: " 
    plt.title(my_title)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('z [m]')
    plot_filename = stub + '_marker_starts_daughter_RZ.pdf'
    plt.savefig(plot_filename)
 

def define_prt_markers_07(fn_hdf5, set, rhos, pitches, R0, aa,  desc=None):
    """
    define_prt_markers: 
       hdf5_fn     name of hdf5 file
       set         set number = 1, 2, 3, ...
       desc        description
    """

    mystring  = fn_hdf5.split('.')
    stub      = mystring[0]
    
    Nrho   = rhos.size
    Npitch = pitches.size
    Nmrk = Nrho * Npitch
    
    R     = np.zeros(Nmrk)
    pitch = np.zeros(Nmrk)
    
    if set==1:

        nn = 0
        for irho in range(Nrho):
            for ipitch in range(Npitch):
                R[nn] = R0 + rhos[irho]*aa
                pitch[nn] = pitches[ipitch]
                nn += 1

        plt.figure(figsize=(7.,5.))        
        plt.plot(R, pitch, 'ro', ms=3)
        plt.xlabel('Marker Rmajor')
        plt.ylabel('Marker pitch')
        plt.title('Markers constructed by define_prt_markers_07')
        filename = stub + '_marker_rmajor_pitch.pdf'
        plt.savefig(filename)

        z      = np.zeros((Nmrk))
        ids    = np.linspace(1,Nmrk,Nmrk)
        mass   = 4     * np.ones(ids.shape)
        charge = 2     * np.ones(ids.shape)
        anum   = 4     * np.ones(ids.shape)
        znum   = 2     * np.ones(ids.shape)
        phi    = 90    * np.ones(ids.shape)
        weight = 1     * np.ones(ids.shape)
        time   = 0     * np.ones(ids.shape)
        energy = 3.5e6 * np.ones(ids.shape)
                
        theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
        gamma = 1 + energy * const["elementary charge"][0]               \
            / ( const["alpha particle mass"][0]                          \
                * np.power(const["speed of light in vacuum"][0],2) )
        v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
        vR   = np.sqrt(1-pitch*pitch)*v
        vphi = pitch*v
        vz   = 0*v           # why vz=0 identically?                 

    print("")
    print("=======================================================")
    print("")
    print("    values for marker-particle set ", set)
    print("")
    print("   Nmrk: {0:6d}".format(Nmrk))
    print("   mass:   ", mass)
    print("   charge: ", charge)
    print("   anum:   ", anum)
    print("   znum:   ", znum)
    print("   phi:    ", phi)
    print("   weight: ", weight)
    print("   time:   ", time)
    print("   energy: ", energy)
    print("   pitch:  ", pitch)
    print("   theta:  ", theta)
    print("   gamma:  ", gamma)
    print("   v:      ", v)
    print("   vR:     ", vR)
    print("   vphi:   ", vphi)
    print("   vz:     ", vz)
    print("   R:      ", R)
    print("   z:      ", z)
    print("")

    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    out={}
    out['z'] = z
    out['R'] = R 
    out['markers'] = Nmrk
    
    return out


def define_prt_markers_08(fn_hdf5, fn_geqdsk, set, settings, desc=None):
#  sds 12/6/2020
    eq_index  = settings["eq_index"]
    
    pitch_min = settings["pitch_min"]
    pitch_max = settings["pitch_max"]
    rmin      = settings["rmin"]
    rmax      = settings["rmax"]
    zmin      = settings["zmin"]
    zmax      = settings["zmax"]
    rhomax    = settings["rhomax"]
    rhomin    = settings["rhomin"]

    npitch    = settings["npitch"]
    nr        = settings["nr"]
    nz        = settings["nz"]
    ngyro     = settings["ngyro"]

    pitch_array = np.linspace(pitch_min, pitch_max, npitch)
    gyro_array  = np.linspace(0., 2.*np.pi, endpoint=False)
    rr_array    = np.linspace(rmin,rmax,nr)
    zz_array    = np.linspace(zmin,zmax,nz)

    rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index)

    Nmrk = 0

    nmax = nr * nz * npitch * ngyro
    
    R     = np.zeros(nmax)
    z     = np.zeros(nmax)
    pitch = np.zeros(nmax)
    gyro  = np.zeros(nmax)
    
    for ii in range(nr):
        for jj in range(nz):
            for kk in range(npitch):
                for mm in range(ngyro):
                    pdb.set_trace()
                    rho = rho_interpolator(rr_array[ii], zz_array[jj])
                
                    if((rho <= rhomax) and (rho >= rhomin)):
                    
                        R[Nmrk]     =    rr_array[ii]
                        z[Nmrk]     =    zz_array[jj]
                        pitch[Nmrk] = pitch_array[kk]
                        gyro[Nmrk]  =  gyro_array[mm]

                        Nmrk += 1
                        
    print("\n number of candidate markers: ", nmax)
    print(" number of qualified markers: ", Nmrk)
    
    R     =     R[0:Nmrk]
    z     =     z[0:Nmrk]
    pitch = pitch[0:Nmrk]
    gyro  =  gyro[0:Nmrk]
    
    
    mystring  = fn_hdf5.split('.')
    stub      = mystring[0]
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs

    plt.figure(figsize=(4.,7.2))
    plt.plot(rl,zl,'k-')
    plt.plot(R,z,'ro', ms=2)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Z vs R of markers')
    filename = stub + '_marker_RZ.pdf'
    plt.savefig(filename)
               
    plt.figure(figsize=(7.,5.))
                     
    plt.plot(R, pitch, 'ro', ms=3)
    plt.xlabel('Marker Rmajor')
    plt.ylabel('Marker pitch')
    plt.title('Markers constructed by define_prt_markers_08')
    filename = stub + '_marker_R_pitch.pdf'
    plt.savefig(filename)

    ids    = np.linspace(1,Nmrk,Nmrk)
    mass   = 4     * np.ones(ids.shape)
    charge = 2     * np.ones(ids.shape)
    anum   = 4     * np.ones(ids.shape)
    znum   = 2     * np.ones(ids.shape)
    phi    = 90    * np.ones(ids.shape)
    weight = 1     * np.ones(ids.shape)
    time   = 0     * np.ones(ids.shape)
    energy = 3.5e6 * np.ones(ids.shape)
                
    theta  = 2     * np.pi*np.random.rand(Nmrk)     # this is not used

    
    gamma = 1 + energy * const["elementary charge"][0]                 \
            / ( const["alpha particle mass"][0]                        \
                * np.power(const["speed of light in vacuum"][0],2) )
                     
    v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]

    vphi = pitch*v

    vperp = np.sqrt(1-pitch*pitch)*v

    vR = np.zeros(Nmrk)
    vz = np.zeros(Nmrk)

    for ii in range(Nmrk):
                     
        vR[ii] = vperp[ii] * np.sin(gyro[ii])
        vz[ii] = vperp[ii] * np.cos(gyro[ii])
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)


def define_prt_markers_09(fn_hdf5, fn_geqdsk, fn_hdf5_parent, set, end_condition, settings, desc=None):
    """
    copy marker data for markers that did not reach untimely end (or did ...)
    """

    try:
        nmax_markers = settings["nmax_markers"]
    except:
        nmax_markers = 0

    try:
        cull_factor = settings["cull_factor"]
    except:
        cull_factor = 1.
        
    print("\n inside define_prt_markers_09: \n")
    print("    set            = ", set)
    print("    end_condition  = ", end_condition)
    print("    fn_hdf5_parent = ", fn_hdf5_parent)
    print("    nmax_markers   = ", nmax_markers)
    print("    cull_factor    = ", cull_factor)
    print("    note:  if nmax_markers = 0, then cull_factor is ignored")
    
    #  copies markers of parent run into markers for daughter run with culling

    do_rasterized = True
    geq_strings  = fn_hdf5.split('.')
    stub         = geq_strings[0] 
    
    # ---------------------------------------------
    #   get equilibrium and (R,Z) of LCFS

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    eq_index = settings["index"]
    rlcfs    =  gg.equilibria[eq_index].rlcfs
    zlcfs    =  gg.equilibria[eq_index].zlcfs

    # -------------------------------------------------------
          
    if(set == 1):
    
        new_or_old = process_ascot.old_or_new(fn_hdf5_parent)
    
        if(new_or_old == 0):
            aa = process_ascot.myread_hdf5(fn_hdf5_parent)
        elif(new_or_old == 1):
            aa = process_ascot.myread_hdf5_new(fn_hdf5_parent)
        
        # aa = process_ascot.myread_hdf5(fn_hdf5_parent)

        endcond = aa["endcond"]
        
        if(end_condition == "wall_and_rhomax"):
            ii_end = (endcond == 8) ^ (endcond == 32) ^ (endcond==544)
            name_end = " wall-hit and rhomax"
        elif (end_condition == "wall_only"):
            ii_end = (endcond == 8)
            name_end = "wall-hit only"
        elif (end_condition == "rhomax_only"):
            ii_end = (endcond == 32) ^ (endcond==544)
            name_end = "rhomax only"
        elif (end_condition == "survived"):
            ii_end   = (endcond == 2) ^ (endcond == 4) ^ (endcond == 1) ^ (endcond == 256)
        else:
            print(" marker_sets_09:  name_end must be rhomax_only, wall-hit only, wall-and_rhomax or survived")
            exit()
            
        print("    ii_end[0:10] = ", ii_end[0:10], "\n")
       
       #  copy marker data for markers that did not reach untimely end
       
        R_all  = aa["marker_r"]
        nparent_all = R_all.size
        
        R      = aa["marker_r"][ii_end]
        z      = aa["marker_z"][ii_end]
        phi    = aa["marker_phi"][ii_end]  
        vR     = aa["marker_vr"][ii_end]
        vphi   = aa["marker_vphi"][ii_end]
        vz     = aa["marker_vz"][ii_end]
        mass   = aa["mass"][ii_end]
        charge = aa["charge"][ii_end]
        anum   = aa["anum"][ii_end]
        znum   = aa["znum"][ii_end]
        time   = aa["time"][ii_end]
        weight = aa["weight"][ii_end]
        
        time   = np.zeros(znum.size)     # initialize times to zero
        weight_parent = np.sum(weight)
                
        nn_parent = R.size
        randoms   = np.random.random(nn_parent)

        rcheck    = 1./cull_factor

        ii_good = 0
        if(nmax_markers > 0):
           ii_good = nmax_markers
        else:
           ii_good = int(R.size/cull_factor)
           
        #  shorten the arrays
                    
        R      =  R[0:ii_good]
        z      =  z[0:ii_good]
        phi    =  phi[0:ii_good]
        vR     =  vR[0:ii_good]
        vphi   =  vphi[0:ii_good]
        vz     =  vz[0:ii_good]
        mass   =  mass[0:ii_good]
        charge =  charge[0:ii_good]
        anum   =  anum[0:ii_good]
        znum   =  znum[0:ii_good]
        time   =  time[0:ii_good]
        weight =  weight[0:ii_good]
        weight =  weight / np.sum(weight)

        nn_daughter = R.size
        Nmrk        = nn_daughter
        ids         = np.linspace(1,nn_daughter, nn_daughter).astype(int)
        
        print("   parent:    total number of markers:                      ", nparent_all)
        print("   parent:    number of endcond-qualified markers:          ", nn_parent)
        print("   cull factor:                                             ", cull_factor)
        print("   daughter:  number of markers (after cull/nmax_markers):  ", nn_daughter)

        mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    # ++++++++++++++++++++++++++++++++++++
    #   for plots

    vtot         = np.sqrt(vR**2 + vphi**2 + vz**2)
    pitch_approx = vphi/vtot

    stub  = fn_hdf5.split('.')[0]
    nn    = Nmrk

    try:
        if(settings['nplot_max']):
            nn = settings['nplot_max']
    except:
        xdummy = 1.

    rhos = rho.compute_rho(fn_geqdsk, R, z)
    plt.close()
    plt.figure(figsize=(7.,5.))
    plt.hist(rhos,bins=50, histtype='step',density=True, color='r')
    plt.title('histogram of marker rho_poloidal')
    fn_out = stub + '_marker_rhos_09.pdf'
    plt.savefig(fn_out)
               
    plt.close()
    plt.figure(figsize=(4.,7.))
    plt.plot(rlcfs,zlcfs,'k-')
    plt.plot(R[0:nn], z[0:nn], 'ro', ms=0.5)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('Z [m]')
    plt.title('Daughter markers')
    fn_out = stub + '_marker_RZ_09.pdf'
    plt.savefig(fn_out)
    plt.close()

    plt.close()
    plt.figure(figsize=(7.,5.))
    plt.hist(pitch_approx[0:nn], bins=50, histtype='step', color='b')
    plt.title('Distribution of marker vphi/vtot')
    plt.xlabel(' vphi/vtot')
    fn_out = stub + '_pitch_approx_09.pdf'
    plt.savefig(fn_out)

    plt.close()
    plt.figure(figsize=(7.,5.))
    plt.plot(R[0:nn], pitch_approx[0:nn], 'ro', ms=0.5)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('')
    plt.title('pitch_approx vs Rmajor')
    fn_out = stub + '_marker_R_pitch_09.pdf'
    plt.savefig(fn_out)
    plt.close()
  

def define_prt_markers_10(fn_lossmap, fn_weights, fn_daughter, fn_geqdsk, set, settings, desc=None):
    
    """
    usage:  weights = define_prt_markers_10(fn_lossmap, fn_weights, fn_daughter, fn_geqdsk, set, settings, desc=None):

    fn_lossmap   ASCOT output file which will be the basis for the lossmap
    fn_weights   ASCOT input file which has a large ensemble of candidate markers. This
                 file will be the basis for the weights
    fn_daughter  ASCOT input file which we are creating
    fn_geqdsk    equilibrium file  (used only for plots, not for any calculation)
    
    set  ... currently must be 1

 
   unaccounted, norm_weights, index are mandatory, others are optional

    settings["unaccounted"]  = threshold for including markers
    settings["norm_weights"] = 0   ... use raw computed weights
                             = 1   ... renormalize weights so their sum is 1.00
    settings["index"]        =  time index of equilibrium file
    settings["nplot_max"]    = maximum number of points to plot

    settings["rhogrid_min"]  ... default = 0.5
    settings["rhogrid_max"]  ... default = 1.1
    settings["nrho_grid"]    ... default = 61
    settings["nksi_grid"]    ... default = 50
   
 
    """
    
    unaccounted  = settings["unaccounted"]
    norm_weights = settings["norm_weights"]
    
    rhogrid_min = 0.5
    rhogrid_max = 1.1
    nrho_grid   = 60
    rmin        = 1.2
    plotFig     = 1


    stub = fn_daughter.split('.')[0]
    fn_plotfile = stub + '_lossmap.pdf'
        
    try:
        if(settings["rhogrid_min"] > 0.):
            rhogrid_min = settings["rhogrid_min"]
    except:
        dummy=0.

    try:
        if(settings["rhogrid_max"] > 0.):
            rhogrid_max = settings["rhogrid_max"]
    except:
        dummy=0.

    try:
        if(settings["nrho_grid"] > 0.):
            nrho_grid = settings["nrho_grid"]
    except:
        dummy=0.

    try:
        if(settings["nksi_grid"] > 0.):
            nksi_grid = settings["nksi_grid"]
    except:
        dummy=0.

    try:
        if(settings["rmin"] > 0.):
            rmin = settings["rmin"]
    except:
        dummy=0.


    rhogrid = np.linspace(rhogrid_min, rhogrid_max, nrho_grid)
    ksigrid = np.linspace(-1,1,nksi_grid)
    rmin    = 1.2
    plotFig = 1

    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    #  compute weights_new
    
    if (set == 1):
        
        weights_new = loss.plotlossmap(fn_plotfile, fn_lossmap, fn_weights, unaccounted, rhogrid, ksigrid, rmin, plotFig)

    elif (set == 2):

        fnB = settings["fnB"]   # filename containing Bfield (2D or 3D)

        try:
            Nmax = settings["Nmax"]
        except:
            Nmax = 0

        if(Nmax > 0):

           weights_new =   applylossmap(fn_lossmap, fn_weights, unaccounted,           \
                                        rhogrid=rhogrid, ksigrid=ksigrid, rmin=rmin,   \
                                        plotFig=plotFig, Nmax=Nmax, fnB=fnB)
        else:

           weights_new =   applylossmap(fn_lossmap, fn_weights, unaccounted,           \
                                        rhogrid=rhogrid, ksigrid=ksigrid, rmin=rmin,   \
                                        plotFig=plotFig, fnB=fnB)
    else:
           
        print("   marker_sets.define_prt_markers_10 set = ", set, " is not supported.")
        sys.exit()


    print("\n inside define_prt_markers_10 \n")
    print("    set          = ", set)
    print("    unaccounted  = ", unaccounted)
    print("    rhogrid_min  = ", rhogrid_min)
    print("    rhogrid_max  = ", rhogrid_max)
    print("    nrho_grid    = ", nrho_grid)
    print("    nksi_grid    = ", nksi_grid, "\n")
    print("    fn_lossmap   = ", fn_lossmap)
    print("    fn_weights   = ", fn_weights)
    if(set == 2):
        print("    fnB          = ", fnB,  "\n")
        print("    Nmax         = ", Nmax, "\n")
        
    ii = (weights_new > 0)
    
    markers = get.get_markers(fn_weights)
    
    r      =  markers['r'][ii]
    z      =  markers['z'][ii]
    phi    =  markers['phi'][ii]
    vr     =  markers['vr'][ii]
    vphi   =  markers['vphi'][ii]
    vz     =  markers['vz'][ii]
    mass   =  markers['mass'][ii]
    charge =  markers['charge'][ii]
    anum   =  markers['anum'][ii]
    znum   =  markers['znum'][ii]
    time   =  markers['time'][ii]
    weight =  markers['weight'][ii]

    Nmrk = r.size

    time = np.zeros(Nmrk)
    ids  = np.linspace(1,Nmrk, Nmrk).astype(int)

    weight_sum = np.sum(weight)
    print("    sum of weights = ", weight_sum, "\n")
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    try:
            
        max_markers = settings["max_markers"]
        
        if (max_markers < mass.size):
                
            ids    =    ids[0:max_markers]
            mass   =   mass[0:max_markers]
            charge = charge[0:max_markers]
            r      =      r[0:max_markers]
            phi    =    phi[0:max_markers]
            z      =      z[0:max_markers]
            vr     =     vr[0:max_markers]
            vphi   =   vphi[0:max_markers]
            vz     =     vz[0:max_markers]
            anum   =   anum[0:max_markers]
            znum   =   znum[0:max_markers]
            weight = weight[0:max_markers]
            time   =   time[0:max_markers]
            #pitch  =  pitch[0:max_markers]
            #vtot   =   vtot[0:max_markers]
            #ekev   =   ekev[0:max_markers]
                
            #weight_total = np.sum(weight)
            #weight = weight/weight_total      # renormalize
                
            Nmrk   = max_markers
                
            print("   ... as instructed, I have reduced number of parent markers to ", max_markers, "\n")
                
                
    except:
        print("   ... max_markers not present in settings, so I will use all markers")
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    

    
    if (norm_weights == 1):
        weight = weight / np.sum(weight)
        print("    ... have renormalized weights so they sum to unity")
    else:
        print("   ... I have NOT renormalized weights so that they sum to unity")
        
    #  write marker information into the ASCOT input file that we are creating
    
    mrk_prt.write_hdf5(fn_daughter, Nmrk, ids, mass, charge, r, phi, z, vr, vphi, vz, anum, znum, weight, time, desc=desc)

    
    # ++++++++++++++++++++++++++++++++++++++++++++++
    #   make some plots

    vtot = np.sqrt(vphi**2 + vr**2 + vz**2)
    pitch_approx = vphi/vtot

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    #geq_strings  = fn_geqdsk.split('.')
    #stub         = geq_strings[0] + '_'
    #geq_filename = stub + 'equilibrium.pdf'

    eq_index = settings["index"]

    rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index) # 5/24/2020
         
    psi_rmin       =  gg.equilibria[eq_index].rmin
    psi_rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    psi_nr         =  gg.equilibria[eq_index].nw
    psi_zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    psi_zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    psi_nz         =  gg.equilibria[eq_index].nh
        
    psiPolSqrt     =  gg.equilibria[eq_index].psiPolSqrt
    rlcfs          =  gg.equilibria[eq_index].rlcfs
    zlcfs          =  gg.equilibria[eq_index].zlcfs

    geq_rarray = np.linspace(psi_rmin, psi_rmax, psi_nr)
    geq_zarray = np.linspace(psi_zmin, psi_zmax, psi_nz)

    # transpose so that we are on a grid = [R,z]. define a function
    # psiPolSqrt_interp so that we can determine the local
    # value
         
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)   # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))

    rhogeq_interp = interpolate.interp2d(geq_rarray, geq_zarray, psiPolSqrt, kind='cubic')

    #  optionally reduce number of points to plot
    
    nn = Nmrk   # number of points to plot

    try:
        if(settings['nplot_max']):
            nn = settings['nplot_max']
            print("   ... marker_10:  I have reduced the number of points in the plots to ", nn)
    except:
           xdummy = 1.

    rhos    = np.zeros(nn)
    vphi_nn = vphi[0:nn]
    vr_nn   = vr[0:nn]
    vz_nn   = vz[0:nn]
    vtot_nn = np.sqrt(vphi_nn**2 + vr_nn**2 + vz_nn**2)
    pitch   = vphi_nn /vtot_nn   # this is only approximate, should be vpar/vtot

    for jk in range(nn):
        rhos[jk] = rhogeq_interp(r[jk], z[jk])

    plt.close()
    plt.figure(figsize=(8.,5.))
    plt.plot(rhos, pitch, 'ro', ms=0.5)
    plt.xlabel('poloidal rho')
    plt.title('approximate pitch vs rho')
    plt.savefig('marker_rho_pitch_10.pdf')
               
    plt.close()
    plt.figure(figsize=(4.,7.))
    plt.plot(rlcfs,zlcfs,'k-')
    plt.plot(r[0:nn], z[0:nn], 'ro', ms=0.5)
    plt.xlabel('Rmajor [m]')
    plt.ylabel('Z [m]')
    plt.title('marker R Z')
    plt.savefig('marker_RZ_daughter_10.pdf')
    plt.close()
        

def define_prt_markers_11(fn_hdf5, fn_geqdsk, set, settings, desc=None):
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  sds 1/17/2022
#
#  create markers to do detailed study of banana-trapped orbits
#  pitch is fixed (typically a small value) but will allow
#  multiple toroidal angles
    eq_index  = settings["eq_index"]
    
    phi_min    = settings["phi_min"]
    phi_max    = settings["phi_max"]
    pitch_min  = settings["pitch_min"]
    pitch_max  = settings["pitch_max"]
    rmin       = settings["rmin"]
    rmax       = settings["rmax"]
    zmin       = settings["zmin"]
    zmax       = settings["zmax"]
    rhomax     = settings["rhomax"]
    rhomin     = settings["rhomin"]
    gyro_angle = settings["gyro_angle"]
    nphi       = settings["nphi"]
    nr         = settings["nr"]
    nz         = settings["nz"]
    nloops     = settings["nloops"]
    npitch     = settings["npitch"]

    pitch_array = np.linspace(pitch_min, pitch_max, npitch)
    
    phi_array   = np.linspace(0., 20., nphi)
    gyro_array  = np.linspace(0., 2.*np.pi, endpoint=False)
    rr_array    = np.linspace(rmin,rmax,nr)
    zz_array    = np.linspace(zmin,zmax,nz)
    print("   ... define_prt_markers_1, just before call to construct_rho_interpolator")
    #rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index)

    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    print("   ... define_prt_markers_1, just after call to construct_rho_interpolator, type = ",type( rho_interpolator))
    Nmrk = 0

    nmax = nr * nz * nphi * nloops# * ngyro
    
    R     = np.zeros(nmax)
    z     = np.zeros(nmax)
    pitch = np.zeros(nmax)
    gyro  = np.zeros(nmax)
    phi   = np.zeros(nmax)
    rho   = np.zeros(nmax)
    print("   ... inside define_prt_markers_11:  type of rho_interpolator = ", type(rho_interpolator))
    
    for ii in range(nr):
        for jj in range(nz):
            for kk in range(nphi):
                for nn in range(npitch):
                    for mm in range(nloops):   # duplicate markers 'nloops' times
                    
                        rho = rho_interpolator(rr_array[ii], zz_array[jj])
                        #pdb.set_trace(header="insider marker_11")
                        if((rho <= rhomax) and (rho >= rhomin)):
                    
                            R[Nmrk]     =     rr_array[ii]
                            z[Nmrk]     =     zz_array[jj]
                            phi[Nmrk]   =    phi_array[kk]
                            pitch[Nmrk] =  pitch_array[nn]
                            gyro[Nmrk]  =  gyro_angle
                            
                            Nmrk += 1
                        
    print("\n number of candidate markers: ", nmax)
    print(" number of qualified markers: ", Nmrk)
    
    R     =      R[0:Nmrk]
    z     =      z[0:Nmrk]
    phi   =    phi[0:Nmrk]
    pitch =  pitch[0:Nmrk]
    gyro  =   gyro[0:Nmrk]
    
    
    ids    =   np.linspace(1,Nmrk,Nmrk)
    mass   = 4     * np.ones(ids.shape)
    charge = 2     * np.ones(ids.shape)
    anum   = 4     * np.ones(ids.shape)
    znum   = 2     * np.ones(ids.shape)
    weight = 1     * np.ones(ids.shape)
    time   = 0     * np.ones(ids.shape)
    energy = 3.5e6 * np.ones(ids.shape)
    #pdb.set_trace()             
    gamma = 1 + energy * const["elementary charge"][0]                 \
            / ( const["alpha particle mass"][0]                        \
                * np.power(const["speed of light in vacuum"][0],2) )
                     
    v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]

    vphi = pitch*v

    vperp = np.sqrt(1-pitch*pitch)*v

    vR = np.zeros(Nmrk)
    vz = np.zeros(Nmrk)

    for ii in range(Nmrk):
        vR[ii] = vperp[ii] * np.sin(gyro[ii])
        vz[ii] = vperp[ii] * np.cos(gyro[ii])
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    mystring  = fn_hdf5.split('.')
    stub      = mystring[0]
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs

    plt.figure(figsize=(4.,7.2))
    plt.plot(rl,zl,'k-')
    plt.plot(R,z,'ro', ms=2)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Z vs R of markers')
    filename = stub + '_marker_RZ.pdf'
    plt.savefig(filename)
               
    plt.figure(figsize=(7.,5.))
                     
    plt.plot(R, pitch, 'ro', ms=3)
    plt.xlabel('Marker Rmajor')
    plt.ylabel('Marker pitch')
    plt.title('Markers constructed by define_prt_markers_011')
    filename = stub + '_marker_R_pitch.pdf'
    plt.savefig(filename)


def define_prt_markers_12(fn_hdf5, fn_geqdsk, set, settings, desc=None):

    eq_index  = settings["eq_index"]
    
    phi_min    = settings["phi_min"]
    phi_max    = settings["phi_max"]
    pitch_min  = settings["pitch_min"]
    pitch_max  = settings["pitch_max"]
    rmin       = settings["rmin"]
    rmax       = settings["rmax"]
    zmin       = settings["zmin"]
    zmax       = settings["zmax"]
    rhomax     = settings["rhomax"]
    rhomin     = settings["rhomin"]
    gyro_angle = settings["gyro_angle"]
    nphi       = settings["nphi"]
    nr         = settings["nr"]
    nz         = settings["nz"]
    nloops     = settings["nloops"]
    npitch     = settings["npitch"]

    pitch_array = np.linspace(pitch_min, pitch_max, npitch)
    
    phi_array   = np.linspace(0., 20., nphi)
    gyro_array  = np.linspace(0., 2.*np.pi, endpoint=False)
    rr_array    = np.linspace(rmin,rmax,nr)
    zz_array    = np.linspace(zmin,zmax,nz)
    print("   ... marker_sets.define_prt_markers_12:  rr_array = ", rr_array)
    print("   ... marker_sets.define_prt_markers_12:  zz_array = ", zz_array)
    #print("   ... define_prt_markers_1, just before call to construct_rho_interpolator")
    #rho_interpolator = construct_rho_interpolator(fn_geqdsk, eq_index)

    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    print("   ... define_prt_markers_1, just after call to construct_rho_interpolator, type = ",type( rho_interpolator))
    Nmrk = 0

    nmax = nr * nz * nphi * nloops# * ngyro 
    
    R     = np.zeros(nmax)
    z     = np.zeros(nmax)
    pitch = np.zeros(nmax)
    gyro  = np.zeros(nmax)
    phi   = np.zeros(nmax)
    rho   = np.zeros(nmax)
    
    
    for ii in range(nr):
        for jj in range(nz):
            for kk in range(nphi):
                for nn in range(npitch):
                    for mm in range(nloops):   # duplicate markers 'nloops' times
                    
                        rho = rho_interpolator(rr_array[ii], zz_array[jj])
                        #pdb.set_trace(header="insider marker_11")
                        if((rho <= rhomax) and (rho >= rhomin)):
                    
                            R[Nmrk]     =     rr_array[ii]
                            z[Nmrk]     =     zz_array[jj]
                            phi[Nmrk]   =    phi_array[kk]
                            pitch[Nmrk] =  pitch_array[nn]

                            if ( R[Nmrk] <=1.86) & (np.abs(z[Nmrk]<=0.22)):   # special logic
                                pitch[Nmrk] = -0.20                           # to avoid passing orbits
                                                    
                            gyro[Nmrk]  =  gyro_angle
                            
                            Nmrk += 1
                        
    print("\n number of candidate markers: ", nmax)
    print(" number of qualified markers: ", Nmrk)
    
    R     =      R[0:Nmrk]
    z     =      z[0:Nmrk]
    phi   =    phi[0:Nmrk]
    pitch =  pitch[0:Nmrk]
    gyro  =   gyro[0:Nmrk]

    #pdb.set_trace()
    
    ids    =   np.linspace(1,Nmrk,Nmrk)
    mass   = 4     * np.ones(ids.shape)
    charge = 2     * np.ones(ids.shape)
    anum   = 4     * np.ones(ids.shape)
    znum   = 2     * np.ones(ids.shape)
    weight = 1     * np.ones(ids.shape)
    time   = 0     * np.ones(ids.shape)
    energy = 3.5e6 * np.ones(ids.shape)
    #pdb.set_trace()             
    gamma = 1 + energy * const["elementary charge"][0]                 \
            / ( const["alpha particle mass"][0]                        \
                * np.power(const["speed of light in vacuum"][0],2) )
                     
    v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]

    vphi = pitch*v

    vperp = np.sqrt(1-pitch*pitch)*v

    vR = np.zeros(Nmrk)
    vz = np.zeros(Nmrk)

    for ii in range(Nmrk):
        vR[ii] = vperp[ii] * np.sin(gyro[ii])
        vz[ii] = vperp[ii] * np.cos(gyro[ii])
    
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    mystring  = fn_hdf5.split('.')
    stub      = mystring[0]
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs

    plt.figure(figsize=(4.,7.2))
    plt.plot(rl,zl,'k-')
    plt.plot(R,z,'ro', ms=2)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Z vs R of markers')
    filename = stub + '_marker_RZ.pdf'
    plt.savefig(filename)
               
    plt.figure(figsize=(7.,5.))
                     
    plt.plot(R, pitch, 'ro', ms=3)
    plt.xlabel('Marker Rmajor')
    plt.ylabel('Marker pitch')
    plt.title('Markers constructed by define_prt_markers_011')
    filename = stub + '_marker_R_pitch.pdf'
    plt.savefig(filename)
    

def define_prt_markers_13(fn_hdf5, fn_geqdsk, set, settings, fn_parent, desc=None):

    #  sds 2/6/22
    #
    #  this properly sets the intial velocities based on true pitch angle
    #  rather than a specification of vphi/vtot (at last!)
    #
    #  you must provide a 'parent' ascot output file ("fn_parent") from which I can
    #  read the 3D magnetic field.  the magnetic field in this parent should be
    #  identical to the magnetic field you intend to use in this simulation

    # see memo velocity_vectors.pdf for the analysis of how we compute
    # vphi, vz, and vR from vtot and the assumed pitch angle

    
    eq_index  = settings["eq_index"]
    
    phi_min    = settings["phi_min"]
    phi_max    = settings["phi_max"]
    pitch_min  = settings["pitch_min"]
    pitch_max  = settings["pitch_max"]
    rmin       = settings["rmin"]
    rmax       = settings["rmax"]
    zmin       = settings["zmin"]
    zmax       = settings["zmax"]
    rhomax     = settings["rhomax"]
    rhomin     = settings["rhomin"]
    nphi       = settings["nphi"]
    nr         = settings["nr"]
    nz         = settings["nz"]
    nloops     = settings["nloops"]
    npitch     = settings["npitch"]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  get a5 from which we can compute the local B-field
    #  Look first in output directory, then in local directory
    
    stub       = fn_parent.split('_')[0]
    remainder  = fn_parent.split('_')[1]
    remainder  = remainder.split('.')[0]
    
    fn_full= '/project/projectdirs/m3195/ascot/ascot_run_output/' + stub \
             + '_work_' + remainder + '/' + stub + '_' + remainder + '.h5'
    try:
        print("   ... marker_sets.define_prt_markers_13:  about to read B-field from file: ",fn_full)
        a5 = Ascotpy(fn_full)
        print("   ... marker_sets.define_prt_markers_13:  have read B-field from file: ", fn_full)
    except:
        print("   ... marker_sets.define_prt_markers_13:  reading B-field from file: ", fn_parent)
        a5 = Ascotpy(fn)

    print("   ... about to initialize the bfield")
    #pdb.set_trace()
    a5.init(bfield=True)
    print("   ... have completed initialization of bfield")
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=
    
    pitch_array = np.linspace(pitch_min, pitch_max, npitch)
    
    phi_array   = np.linspace(phi_min, phi_max, nphi)
    rr_array    = np.linspace(rmin,    rmax,    nr)
    zz_array    = np.linspace(zmin,    zmax,    nz)
    
    print("   ... marker_sets.define_prt_markers_13:  rr_array = ", rr_array)
    print("   ... marker_sets.define_prt_markers_13:  zz_array = ", zz_array)

    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    print("   ... define_prt_markers_1, just after call to construct_rho_interpolator, type = ",type( rho_interpolator))
    Nmrk = 0

    nmax = nr * nz * nphi * nloops * npitch    # * ngyro 
    
    R     = np.zeros(nmax)
    z     = np.zeros(nmax)
    pitch = np.zeros(nmax)
    phi   = np.zeros(nmax)
    rho   = np.zeros(nmax)
    
    
    for ii in range(nr):
        for jj in range(nz):
            for kk in range(nphi):
                for nn in range(npitch):
                    for mm in range(nloops):   # duplicate markers 'nloops' times
                    
                        rho = rho_interpolator(rr_array[ii], zz_array[jj])
                        
                        if((rho <= rhomax) and (rho >= rhomin)):
                                                
                            R[Nmrk]     =     rr_array[ii]
                            z[Nmrk]     =     zz_array[jj]
                            phi[Nmrk]   =    phi_array[kk]
                            pitch[Nmrk] =  pitch_array[nn]          
                        
                            
                            Nmrk += 1
                        
    print("\n number of candidate markers: ", nmax)
    print(" number of qualified markers: ", Nmrk)
    
    R     =      R[0:Nmrk]
    z     =      z[0:Nmrk]
    phi   =    phi[0:Nmrk]
    pitch =  pitch[0:Nmrk]

    #pdb.set_trace()
    
    ids    =   np.linspace(1,Nmrk,Nmrk)
    mass   = 4     * np.ones(ids.shape)
    charge = 2     * np.ones(ids.shape)
    anum   = 4     * np.ones(ids.shape)
    znum   = 2     * np.ones(ids.shape)
    weight = 1     * np.ones(ids.shape)
    time   = 0     * np.ones(ids.shape)
    energy = 3.5e6 * np.ones(ids.shape)
               
    gamma = 1 + energy * const["elementary charge"][0]                 \
            / ( const["alpha particle mass"][0]                        \
                * np.power(const["speed of light in vacuum"][0],2) )
                     
    v    = np.sqrt(1-1/(gamma*gamma))*const["speed of light in vacuum"][0]
    
    vperp     = np.sqrt(1-pitch*pitch)*v
    vparallel = pitch*v
    
    vphi      = np.zeros(Nmrk)
    vR        = np.zeros(Nmrk)
    vz        = np.zeros(Nmrk)
    v_check   = np.zeros(Nmrk)
    
    
    for jm in range(Nmrk):

        br, bphi, bz = VB.compute_bfield(a5, R[jm], phi[jm], z[jm])
        b = np.sqrt(br**2 + bphi**2 + bz**2)
        
        ff = np.sqrt(1. + (bz/bphi)**2)

        vphi[jm] = vparallel[jm]*(bphi/b) - (bz/bphi)*vperp[jm]/ff
        vz[jm]   = vparallel[jm]*(bz/b)  + vperp[jm]/ff
        vR[jm]   = vparallel[jm] * (br/b)

        v_check[jm] = np.sqrt(vphi[jm]**2 + vz[jm]**2 + vR[jm]**2)

    v_ratios = v_check / v
    if (np.max(np.abs(v_ratios -1) > 1.e-5)):
        print("   ... marker_sets.define_prt_markers_13:  computed velocities do not sum to vtot")
        pdb.set_trace()
        
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, R, phi, z, vR, vphi, vz, anum, znum, weight, time, desc=desc)

    mystring  = fn_hdf5.split('.')
    stub      = mystring[0]
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs

    plt.figure(figsize=(4.,7.2))
    plt.plot(rl,zl,'k-')
    plt.plot(R,z,'ro', ms=2)
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('Z vs R of markers')
    filename = stub + '_marker_RZ.pdf'
    plt.savefig(filename)
               
    plt.figure(figsize=(7.,5.))
                     
    plt.plot(R, pitch, 'ro', ms=3)
    plt.xlabel('Marker Rmajor')
    plt.ylabel('Marker pitch')
    plt.title('Markers constructed by define_prt_markers_013')
    filename = stub + '_marker_R_pitch.pdf'
    plt.savefig(filename)
    

def define_prt_markers_14(fn_hdf5, fn_reverse_markers, set, settings, desc=None):

    # read birth positions and velocities from a file generated by
    # triangulate_torus_16.py or a later version

    nrepeat = settings["nrepeat"]    # number of identical markers at each position
    
    aa = RR.read_any_file(fn_reverse_markers)

    ictrs         = aa[:,0]
    start_rmajors = aa[:,1]
    start_phis    = aa[:,2]
    start_zs      = aa[:,3]
    start_vrs     = aa[:,4]
    start_vphis   = aa[:,5]
    start_vzs     = aa[:,6]

    nn = start_rmajors.size

    nn_out = nn * nrepeat

    rmajors = np.zeros(nn_out)
    phis    = np.zeros(nn_out)
    zs      = np.zeros(nn_out)
    vrs     = np.zeros(nn_out)
    vphis   = np.zeros(nn_out)
    vzs     = np.zeros(nn_out)

    ictr = 0

    for jm in range(nn):
        for jp in range(nrepeat):
            
            rmajors[ictr] = start_rmajors[jm]
            phis[ictr]    =    start_phis[jm]
            zs[ictr]      =      start_zs[jm]
            vrs[ictr]     =     start_vrs[jm]
            vphis[ictr]   =   start_vphis[jm]
            vzs[ictr]     =     start_vzs[jm]

            ictr +=1

    Nmrk = nn_out
     
    ids    =   np.linspace(1,Nmrk,Nmrk)
    mass   = 4     * np.ones(ids.shape)
    charge = 2     * np.ones(ids.shape)
    anum   = 4     * np.ones(ids.shape)
    znum   = 2     * np.ones(ids.shape)
    weight = 1     * np.ones(ids.shape)
    time   = 0     * np.ones(ids.shape)
     
    mrk_prt.write_hdf5(fn_hdf5, Nmrk, ids, mass, charge, rmajors, phis, zs, \
                       vrs, vphis, vzs, anum, znum, weight, time, desc=desc)
     
 # -------------------------------------------------------------------------


def define_gc_markers(hdf5_fn, set, Nmrk, desc=None):
    """
    define_gc_markers: 
       hdf5_fn     name of hdf5 file
       set         set number = 1, 2, 3, ...
       Nmrk        number of markers
       desc        description
 
    """
    
    if (set < 1) or (set > 1):
        print("define_gc_markers:  set must be one or two")
        return None

    #  set-1:  four deeply-trapped alphas
    
    if set == 1:
        
        ids_list    = np.linspace(1,Nmrk, Nmrk)
        for item in ids_list:
            item = int(item)
        ids    = np.array(ids_list)
        
        weight = np.ones(Nmrk)
        pitch  = 0.2     * np.array(np.ones(Nmrk))
        mass   = 4.      * np.ones(Nmrk)
        charge = 2.      * np.ones(Nmrk)
        anum   = 1.      * np.ones(Nmrk)
        znum   = 1.      * np.ones(Nmrk)
        time   = 0.      * np.ones(Nmrk)

        R0     = 1.65
        aa     = 0.50
        Rstart = R0 + aa/(2.*(Nmrk+1))    # e.g. rho = 0.1
        Rend   = R0 - aa/(2.*(Nmrk+1))    # e.g. rho = 0.9
        R      = np.linspace(Rstart, Rend, Nmrk)
        
        phi    = 90.     * np.ones(Nmrk)
        z      = 0       * np.ones(Nmrk)
        zeta   = 2       * np.ones(Nmrk)
        energy = 3.5e6   * np.ones(Nmrk)

    mrk.write_hdf5(hdf5_fn, Nmrk, ids, mass, charge,              \
                   R, phi, z, energy, pitch, zeta,                \
                   anum, znum, weight, time, desc=desc)       

    return None
        

def construct_full_filename(end_condition, delta_time, filename_in):
    
    stub          = filename_in.split('_')[0]
    remainder     = filename_in.split('_')[1]
    remainder     = remainder.split('.')[0]
    
    this_filename = '/project/projectdirs/m3195/ascot/ascot_run_output/' + stub + '_work_' + remainder + '/' + stub + '_' + remainder + '.h5'

    return this_filename


def split_markers(end_condition, delta_time, mult, fn_hdf5, fn_parent):
    """
    split_markers(end_condition, delta_time, mult, fn_hdf5, fn_parent)

    end_condition = wall_and_rhomax, wall_only,or  rhomax_only (a string)
    delta_time    = how far back in time to go when extracting orbit data
    mult          = how many sibling-markers to be created for each parent marker
    fn_hdf5       = filename of ASCOT input file you are writing into
    fn_parent     = filename of parent orbits  

    this module does not return anything
    """
    ff = h5py.File(fn_parent)
    rr = ff['results']

    # +++++++++++++++++++++++++++++++
    #  get orbit data
    
    results_keys = list(rr.keys())
    results      = rr[results_keys[0]]

    orbits  = results["orbit"]

    r        = np.array(orbits["r"])
    z        = np.array(orbits["z"])
    phi      = np.array(orbits["phi"]) % 360.

    pr       = np.array(orbits["pr"])
    pz       = np.array(orbits["pz"])
    pphi     = np.array(orbits["pphi"])
    
    ids_orbs = np.array(orbits["ids"]).astype("int")
    time     = np.array(orbits["time"])
    
    # ++++++++++++++++++++++++++++++
    #   get marker data

    tt          = ff['marker']
    marker_keys = list(tt.keys())   
    markers     = tt[marker_keys[0]]
    
    mass        = np.array(markers["mass"])[:,0]          # amu
    charge      = np.array(markers["charge"])[:,0]
    anum        = np.array(markers["anum"])[:,0]
    znum        = np.array(markers["znum"])[:,0]
    ids_marker  = np.array(markers["id"])[:,0]

    # ++++++++++++++++++++++++++++++
    #   get endstate data

    endstate  = results["endstate"]
        
    ids_end     = np.array(endstate["ids"]).astype(int)
    time_end    = np.array(endstate["time"])
    weight      = np.array(endstate["weight"])
    endcond     = np.array(endstate["endcond"])
    prprt_end   = np.array(endstate["prprt"])
    pzprt_end   = np.array(endstate["pzprt"])
    pphiprt_end = np.array(endstate["pphiprt"])

    ekev_end = np.zeros(time_end.size)

    # pdb.set_trace()

        

    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    #  check that ids in orbits and endstate are the same
    
    ids_unique = np.array(np.unique(ids_orbs)).astype(int)
    for ii in range(ids_end.size):
        if (ids_unique[ii] != ids_end[ii]):
            print(" big problem in back_in_time!")
            sys.exit

    # ++++++++++++++++++++++++++
    #  compute velocities

    Nmrk = ids_end.size

    vr          = np.zeros(ids_orbs.size)
    vz          = np.zeros(ids_orbs.size)
    vphi        = np.zeros(ids_orbs.size)
    ekev        = np.zeros(ids_orbs.size)
    proton_mass = const.physical_constants["proton mass"][0]     # kg


    for ii in range(Nmrk):    # loop over markers

        jj = (ids_orbs == ii+1)

        vr[jj]   = pr[jj]   / ( mass[ii] * proton_mass)
        vz[jj]   = pz[jj]   / ( mass[ii] * proton_mass)
        vphi[jj] = pphi[jj] / ( mass[ii] * proton_mass)
        ekev[jj] = 0.5 * mass[ii] * proton_mass * (vr[jj]**2 + vz[jj]**2 + vphi[jj]**2) / (1.602e-19 * 1000)

    # find indices of markers that are lost
    
    if(end_condition == "wall_and_rhomax"):
        ii_end = (endcond == 8) ^ (endcond == 32) ^ (endcond==544)
        name_end = " wall-hit and rhomax"
    elif (end_condition == "wall_only"):
        ii_end = (endcond == 8)
        name_end = "wall-hit only"
    elif (end_condition == "rhomax_only"):
        ii_end = (endcond == 32) ^ (endcond==544)
        name_end = "rhomax only"
    else:
        print(" split_markers:  name_end must be rhomax_only, wall-hit only, or wall_and_rhomax")
        exit()

    bad_ones = 0

    rout    = np.zeros(Nmrk)
    zout    = np.zeros(Nmrk)
    phiout  = np.zeros(Nmrk)
    vrout   = np.zeros(Nmrk)
    vzout   = np.zeros(Nmrk)
    vphiout = np.zeros(Nmrk)
    ekevout = np.zeros(Nmrk)
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++
    #  compute position and velocity at backward-time-step
    #  note:  per emails early March 2021, the time is
    #         not a monotonic function of index
    
    for kk in range(Nmrk):

        jj = (ids_orbs == kk+1)

        time_1d = time[jj]

        r_1d     =     r[jj]
        z_1d     =     z[jj]
        phi_1d   =   phi[jj]
        vr_1d    =    vr[jj]
        vz_1d    =    vz[jj]
        vphi_1d  =  vphi[jj]
        ekev_1d  =  ekev[jj]

        desired_time = np.max(time_1d) - delta_time

        if(desired_time < np.min(time_1d)):    # cannot go back far enough in time
           bad_ones += 1
           
        orbit_index = np.ndarray.argmin( np.abs(time_1d - desired_time))
        
        rout[kk]    =     r_1d[orbit_index]
        zout[kk]    =     z_1d[orbit_index]
        phiout[kk]  =   phi_1d[orbit_index]
        vrout[kk]   =    vr_1d[orbit_index]
        vzout[kk]   =    vz_1d[orbit_index]
        vphiout[kk] =  vphi_1d[orbit_index]
        ekevout[kk] =  ekev_1d[orbit_index]
                        
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   retain only those markers which meet the end conditions
                        
    rout_shifted    =     rout[ii_end]
    zout_shifted    =     zout[ii_end]
    phiout_shifted  =   phiout[ii_end]
    vrout_shifted   =    vrout[ii_end]
    vzout_shifted   =    vzout[ii_end]
    vphiout_shifted =  vphiout[ii_end]
    weight_shifted  =   weight[ii_end]
    mass_shifted    =     mass[ii_end]
    charge_shifted  =   charge[ii_end]
    anum_shifted    =     anum[ii_end]
    znum_shifted    =     znum[ii_end]

    Nmrk_shifted = anum_shifted.size
    # create empty arrays for the ensemble of 'split' markers
    
    rout_split    =  np.zeros(mult*Nmrk_shifted)
    zout_split    =  np.zeros(mult*Nmrk_shifted)
    phiout_split  =  np.zeros(mult*Nmrk_shifted)
    vrout_split   =  np.zeros(mult*Nmrk_shifted)
    vzout_split   =  np.zeros(mult*Nmrk_shifted)
    vphiout_split =  np.zeros(mult*Nmrk_shifted)
    weight_split  =  np.zeros(mult*Nmrk_shifted)
    mass_split    =  np.zeros(mult*Nmrk_shifted)
    charge_split  =  np.zeros(mult*Nmrk_shifted)
    anum_split    =  np.zeros(mult*Nmrk_shifted)
    znum_split    =  np.zeros(mult*Nmrk_shifted)

    kk = 0
    
    for ii in range(Nmrk_shifted):
        for jj in range(mult):
          
            rout_split[kk]     =    rout_shifted[ii]
            zout_split[kk]     =    zout_shifted[ii]
            phiout_split[kk]   =  phiout_shifted[ii]
            vrout_split[kk]    =   vrout_shifted[ii]
            vzout_split[kk]    =   vzout_shifted[ii]
            vphiout_split[kk]  = vphiout_shifted[ii]
            weight_split[kk]   =  weight_shifted[ii]
            mass_split[kk]     =    mass_shifted[ii]
            charge_split[kk]   =  charge_shifted[ii]
            anum_split[kk]     =    anum_shifted[ii]
            znum_split[kk]     =    znum_shifted[ii]

            kk += 1


    weight_split = weight_split / mult
     
    Nmrk_split = anum_split.size
    ids_split  = np.linspace(1, Nmrk_split, Nmrk_split).astype(int)
    time_split = np.zeros(Nmrk_split)
     
    print("   split_markers:  Nmrk_parent, Nmrk_shifted, Nmrk_split = ", Nmrk, Nmrk_shifted, Nmrk_split)
    print("   number markers could not go back in time far enough: ", bad_ones)

    vtot = np.sqrt( vrout_split**2 + vzout_split*2 + vphiout_split**2)
    etot_kev = 0.5 * mass_split[0]* (vtot**2) * (proton_mass/1.602e-19) / 1.e3

    print("   maximum split-marker energy (keV):  ", np.max(etot_kev), "\n")
    
    mrk_prt.write_hdf5(fn_hdf5,       \
                       Nmrk_split,    \
                       ids_split,     \
                       mass_split,    \
                       charge_split,  \
                       rout_split,    \
                       phiout_split,  \
                       zout_split,    \
                       vrout_split,   \
                       vphiout_split, \
                       vzout_split,   \
                       anum_split,    \
                       znum_split,    \
                       weight_split,  \
                       time_split,    \
                       desc='split')

