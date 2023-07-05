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
