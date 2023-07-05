def define_prt_markers_uniform(settings):
    mpl.rcParams['image.composite_image'] = True    
    time_before = clock.time()
    proton_mass     = 1.67e-27
    amu_mass        = 1.66053904e-27
    electron_charge = 1.602e-19
    set = settings["set"]
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
    bfield_single_file     = settings["bfield_single_file"]      
    bfield_single_filename = settings["bfield_single_filename"]
    birth_rhomin           = settings["birth_rhomin"]            
    birth_rhomax           = settings["birth_rhomax"]            
    Nmrk                   = settings["Nmrk"]                    
    Pitch_min              = settings["Pitch_min"]               
    Pitch_max              = settings["Pitch_max"]
    Phi_min                = settings["Phi_min"]
    Phi_max                = settings["Phi_max"]
    Ekev                   = settings["Ekev"]             
    mass_amu               = settings["mass_amu"]         
    q                      = settings["q"]                
    eq_index               = settings["index"]            
    randomize_gyro         = settings["randomize_gyro"]   
    gyro_angle_fixed       = settings["gyro_fixed"]       
    fn_geqdsk              = settings["fn_geqdsk"]        
    fn_hdf5                = settings["fn_hdf5"]          
    fn_profiles            = settings["fn_profiles"]      
    Npitch                 = settings["Npitch"]           
    Nphi                   = settings["Nphi"]             
    Ngyro                  = settings["Ngyro"]            
    Nrho_profiles          = settings["Nrho_profiles"]    
    Nmrk_original  = Nmrk
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    geq_strings  = fn_geqdsk.split('.')
    stub         = geq_strings[0] + '_'
    geq_filename = stub + 'equilibrium.pdf'
    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    gg.plotEquilibrium(eq_index)
    plt.savefig(geq_filename)
    Rmin       =  gg.equilibria[eq_index].rmin
    Rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    Zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    Zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    rlcfs      =  gg.equilibria[eq_index].rlcfs
    zlcfs      =  gg.equilibria[eq_index].zlcfs
    psi_nr     =  gg.equilibria[eq_index].nw
    psi_nz     =  gg.equilibria[eq_index].nh
    psiPolSqrt =  gg.equilibria[eq_index].psiPolSqrt
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)         
    geq_rarray          = np.linspace(Rmin, Rmax, psi_nr)
    geq_zarray          = np.linspace(Zmin, Zmax, psi_nz)
    NRZ        = int(Nmrk/(Npitch*Nphi*Ngyro))                
    NR         = int( np.sqrt(NRZ*(Rmax-Rmin)/(Zmax-Zmin)) )  
    NZ         = int( np.sqrt(NRZ*(Zmax-Zmin)/(Rmax-Rmin)) )  
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
    marker_Rhos =  rho_interpolator(marker_Rs, marker_Zs, grid=False)
    ii_good  =   (marker_Rhos <=  birth_rhomax)    \
               & (marker_Rhos >=  birth_rhomin)    \
               & (marker_Zs   <=  np.max(zlcfs))   \
               & (marker_Zs   >=  np.min(zlcfs))
    xx = 2.
    Nmrk = marker_Rhos[ii_good].size
    print("   ... number of candidate markers, number of qualified markers: ", Nmrk_big, Nmrk)
    aa_profiles   = proc.read_sparc_profiles_new(fn_profiles, Nrho_profiles)
    alpha_source  = aa_profiles["alpha_source"]/1.e18
    rho_array     = aa_profiles["rhosqrt"]
    R           =      marker_Rs[ii_good]       
    z           =      marker_Zs[ii_good]
    rhos        =    marker_Rhos[ii_good]
    phi         =    marker_phis[ii_good]
    pitches     = marker_pitches[ii_good]
    gyro_angles =   marker_gyros[ii_good]
    if(set == 1):    
       vtot       = np.sqrt(2. * electron_charge * Ekev * 1000. / AMU_MASS *PROTON_MASS)    
    elif(set==2):    
       E_joules                   = Ekev * 1000. * ELECTRON_CHARGE
       mass_electron_relativistic = E_joules / (LIGHT_SPEED**2)
       vtot                       = LIGHT_SPEED * np.sqrt( 1. - (ELECTRON_MASS/mass_electron_relativistic)**2)
       print("  define_prt_markers_uniform:  vtot, c, vtot/c: ", vtot/1.e8, LIGHT_SPEED/1.e8, vtot/LIGHT_SPEED)
    vtots      = vtot       * np.ones(Nmrk)
    vphi      = np.zeros(vtots.size)
    vR        = np.zeros(vtots.size)
    vz        = np.zeros(vtots.size)
    vparallel = np.zeros(vtots.size)   
    vperp     = np.zeros(vtots.size)   
    if(bfield_single_file):
        BB_3D = proc.read_sparc_bfield_3d(bfield_single_filename)
        bhats = compute_bhats(R, phi, z, BB_3D)
    else:
         RR_bgrid, ZZ_bgrid, BTor_grid, BR_grid, BZ_grid, BTot_grid, Psi_dummy = WG.wrapper_geqdskfile(fn_geqdsk)
         if( btor_multiplier !=1.0):
             print("   ... marker_sets/define_prt_markers_uniform:  I will multiply Btor by factor: ", btor_multiplier)
             BTor_grid = BTor_grid * btor_multiplier
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
        velocity_vector_R_phi_Z = CVV.construct_velocity_vector(bhats[jj], vtots[jj], pitches[jj], gyro_angles[jj])
        vR[jj]        = velocity_vector_R_phi_Z[0]
        vphi[jj]      = velocity_vector_R_phi_Z[1]
        vz[jj]        = velocity_vector_R_phi_Z[2]
        vparallel[jj] = pitches[jj] * vtots[jj]
        vperp[jj]     = np.sqrt(vtots[jj]**2 - vparallel[jj]**2)
    if (set == 1):
        weights_alpha_source = np.interp(rhos, rho_array, alpha_source)
        weights_pitch        = np.sqrt(1. - pitches*pitches)
        weight               = R * weights_alpha_source * weights_pitch
        weight               = weight / np.total(weight)
    elif (set == 2):    
        weight = R
        weight = weight/np.sum(weight)
    ids    = np.linspace(1,Nmrk,Nmrk).astype(int)    
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
    vtot_reconstructed      = np.zeros(Nmrk)
    vparallel_reconstructed = np.zeros(Nmrk)
    vperp_reconstructed     = np.zeros(Nmrk)
    pitch_reconstructed     = np.zeros(Nmrk)
    zhat = np.array((0,0.,1.))     
    for jk in range(Nmrk):
        bhat                   = bhats[jk]                       
        hhat                   = VA.cross_product(bhat, zhat)    
        khat                   = VA.cross_product(hhat, bhat)    
        hhat = VA.vector_hat(hhat)   
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
        xx = 1.  
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
        plt.title('Marker weights (marker set 1)')
        plt.xlabel('Rmajor [m]')
        plt.ylim(bottom=0.)
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        plt.grid("both")
        pdf.savefig()
        plt.close(fig=None)  
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
        mmm = (np.abs(z[0:nn]) < 0.1)
        plt.figure(figsize=(7.,5.))
        plt.hist(R[0:nn][mmm], bins=50, rwidth=1,color='c') 
        plt.title('Distribution of marker Rmajor (abs(z)<0.1)')
        plt.xlabel('Rmajor')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()
        plt.figure(figsize=(7.,5.))
        plt.hist(z[0:nn], bins=100, rwidth=1,color='c') 
        plt.title('Distribution of marker Z')
        plt.xlabel('Z')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf()
        plt.figure(figsize=(7.,5.))
        plt.hist(pitches[0:nn], bins=100, rwidth=1,color='c') 
        plt.title('Distribution of marker pitch angles')
        plt.xlabel('pitch angle')
        plt.ylabel('')
        plt.tight_layout(pad=2)
        sds.graph_label(my_graph_label)
        pdf.savefig()
        plt.close()
        plt.clf() 
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
        rho_contours = np.linspace(0.1, 1.0, 10)
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