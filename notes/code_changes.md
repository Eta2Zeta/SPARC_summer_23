- Improving code for faster computation and better readability
``` python
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

    # Attempting a faster way to implement this loop: 


    # Constructing 5D meshgrids for each marker
    R_grid, Z_grid, pitch_grid, phi_grid, gyro_grid = np.meshgrid(
        R_array, Z_array, pitch_array, phi_array, gyro_array, indexing='ij'
    )

    # Flattening the 5D arrays to 1D arrays
    marker_Rs_new      = R_grid.flatten()
    marker_Zs_new      = Z_grid.flatten()
    marker_pitches_new = pitch_grid.flatten()
    marker_phis_new    = phi_grid.flatten()
    marker_gyros_new   = gyro_grid.flatten()

    print("Are the Rs the same? ", np.array_equal(marker_Rs_new, marker_Rs))
    print("Are the Zs the same? ", np.array_equal(marker_Zs_new, marker_Zs))
    print("Are the pitches the same? ", np.array_equal(marker_pitches_new, marker_pitches))
    print("Are the phis the same? ", np.array_equal(marker_phis_new, marker_phis))
    print("Are the gyros the same? ", np.array_equal(marker_gyros_new, marker_gyros))

```    # Constructing arrays to store 5D parameters for each marker
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

    # Attempting a faster way to implement this loop: 


    # Constructing 5D meshgrids for each marker
    R_grid, Z_grid, pitch_grid, phi_grid, gyro_grid = np.meshgrid(
        R_array, Z_array, pitch_array, phi_array, gyro_array, indexing='ij'
    )

    # Flattening the 5D arrays to 1D arrays
    marker_Rs_new      = R_grid.flatten()
    marker_Zs_new      = Z_grid.flatten()
    marker_pitches_new = pitch_grid.flatten()
    marker_phis_new    = phi_grid.flatten()
    marker_gyros_new   = gyro_grid.flatten()

    print("Are the Rs the same? ", np.array_equal(marker_Rs_new, marker_Rs))
    print("Are the Zs the same? ", np.array_equal(marker_Zs_new, marker_Zs))
    print("Are the pitches the same? ", np.array_equal(marker_pitches_new, marker_pitches))
    print("Are the phis the same? ", np.array_equal(marker_phis_new, marker_phis))
    print("Are the gyros the same? ", np.array_equal(marker_gyros_new, marker_gyros))
```
The output appears to be: 
```
Are the Rs the same?  True
Are the Zs the same?  True
Are the pitches the same?  True
Are the phis the same?  True
Are the gyros the same?  True
```

- One other major change I did was to restructure the imports so it looks cleaner 

- In the group_go_files, I made some prompt bold for easier readability

- In pnp_losses.py I added a line to print out the total cpu time for all markers. 

- In get_ascot_AT.py I added some comments for readability 