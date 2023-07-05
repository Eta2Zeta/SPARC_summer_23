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
    charge                 = settings["q"]
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
    psiPolSqrt =  gg.equilibria[eq_index].psiPolSqrt    #Sqrt of the The poloidal magnetic flux psi? 
    
    #  All the properties of gg.equilibira[eq_index]
    #'bcentr', 'case', 'current', 'dr', 'dz', 'ffprime', 'fpol', 'limitr', 'nbbbs', 'nh', 'nw', 'pprime', 'pres', 
    #'psiPolSqrt', 'psirz', 'qpsi', 'rGrid', 'rcentr', 'rdim', 'rlcfs', 'rlim', 'rmax', 'rmaxis', 'rmin', 'setData', 
    #'sibry', 'simag', 'time', 'zGrid', 'zdim', 'zlcfs', 'zlim', 'zmax', 'zmaxis', 'zmid', 'zmin'
    

    #  rhogeq_transpose_2d is an array on [R,Z] on which the equilibrium is
    #  defined.  Using this array, we can compute BZ and BR (as generated by
    #  the plasma itself at any [R,Z] point in the plasma.
    
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)         # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))
    geq_rarray          = np.linspace(Rmin, Rmax, psi_nr)
    geq_zarray          = np.linspace(Zmin, Zmax, psi_nz)
    

    
    # f = (Zmax-Zmin)/(Rmax-Rmin) 

 
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

    # If min/max of R and Z are not speficied in the settings, take the equilibrium values
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
    
    # Find the step size for the R gird and Z grid
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

    # Constructing arrays to store 5D parameters for each marker
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

    print("The 1000th girdpoint has the value of" + marker_Rs[1000],marker_Zs[1000],marker_pitches[1000],marker_phis[1000],marker_Rhos[1000] + "from the original code")
    # Attempting a faster way to implement this loop: 


    # Constructing 5D meshgrids for each marker
    R_grid, Z_grid, pitch_grid, phi_grid, gyro_grid = np.meshgrid(
        R_array, Z_array, pitch_array, phi_array, gyro_array, indexing='ij'
    )

    # Flattening the 5D arrays to 1D arrays
    marker_Rs      = R_grid.flatten()
    marker_Zs      = Z_grid.flatten()
    marker_pitches = pitch_grid.flatten()
    marker_phis    = phi_grid.flatten()
    marker_gyros   = gyro_grid.flatten()



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
        weight               = weight / np.sum(weight)
        
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
        fig=plt.figure(figsize=(xsize, ysize))
        plt.axis([Rmin, Rmax, Zmin, Zmax])
        plt.plot(rlcfs, zlcfs, 'k-', linewidth=2, zorder=1, rasterized=True)
        plt.plot(R[0:nn],z[0:nn], 'ro', ms=0.4,zorder=2, rasterized=True)
        rho_contours = np.linspace(0.1, 0.9, 9)
        rho_contours = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.]
        rho_contours_2 = [1.02, 1.04]
        contours_2 = [1.02, 1.04]
        #pdb.set_trace()
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

