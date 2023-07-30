# Documentation for Marker_sets

## `define_prt_markers_uniform(settings)` Function
Purpose of this module:  generate an ensemble of markers that is distributed uniformily (rather than randomly) in phase space. But this is not quite true ...

To be precise, this module generates an ensemble of markers that is distributed uniformily in [R,Z] space and uniformily in velocity-direction. Alas, being uniformily in [R,Z] space is not quite the same as being distributed uniformily in 3D space.  The reason for this is that the volume element in toroidal coordinates is $dV = (2 \pi R d\phi) * dR * dZ$.

So if we really wanted to construct a marker ensemble that is distributed uniformily in 3D space, we would have to figure out how to properly deal with that.  Presumably, there is a simple way to do that (maybe a homework assignment for an energetic student?) but instead, this module simply constructs the marker ensemble that is uniformily distributed in [R,Z] space, and then adds an additional weighting factor that is proportional to Rmajor.
### Input
`settings` (dictionary): A set of dictionary key and value pairs that specify how the ensemble of markers should be constructed on a fixed grid. Below are the values required for the key and value paris.    

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
#### General Settings
##### `Set`
- `set=1`   fixed grid distribution for non relativistic alpha particles
- `set=2`   fixed grid distribution for relativistic electrons

##### `Marker Option`
There are two options for user to define a set of markers. 
1. Define the total number of markers, then the number of markers in r and z dimensions are determined by the total number of markers Nmrk and the number of markers in other dimensions and the center scaling factor: `Npitch`, `Ngyro`, `Nphi`, and `Maxmax_Npitch_scale_factor`. The reason where `Nr` and `Nz` here are the dependent variables is that `Npitch` plays a often more significant role in determining the particle interactions in the plasma, we therefore want to insure the value of `Npitch` stays fixed. The reason why we care about the total number of markers is that the CPU time is proportional to the number of markers. Another crucial reason why we want to pass in the total number of markers is when we compare the result of ASCOT5 to other simulation programs like spiral, we want to insure we have similar levels of statistics. It is thus often needed to have similar number of makrers. 

The down side of this option is when the total number of markers are defined to be too small. The the grid resolution in the R and Z directions will be too coarse. We therefore offer user another option to spefify the values of number of markers in all dimensions directly and then calculate the final number of markers. 

2. Define the number of markers by defining the number of markers in every dimension plus the `Maxmax_Npitch_scale_factor`. The downside of this approach is that there is no control over the total number of markers and the total number of markers can often end up too large for the pursuit of fine distribution in the marker grid. 

- `Marker_option = 1` if we want to pass in the total number of markers and calculate the needed NZ and NR
- `Marker_option = 2` if we want to pass in Nr and Nz sperately, which eventually spitout the final total number of markers
#### Ensemble size
##### `Nmrk`

## Algorithm calculating the number of `Nr` and `Nz` needed to make the number of total markers closest to the user specified value
The algorithm currently is located in hz_utilities named `get_NRNZ`.
This algorithm starts by guessing the rough number of `Nr` and `Nz` needed by doing a simple math. Then it iteratively find the optimal solution: 
1. We can do a simple estimation of the volumn of a cone + a cylinder for the total number of markers and from there get the formula for `Nr` and `Nz`. The volumn for the cone is $\frac{1}{3} \frac{Nr}{2} \frac{Nz}{2} \pi \text{Npitch} \text{Scale_factor} = \frac{1}{3} \frac{\pi}{2} \text{Scale_factor} \text{Nr}^2 \text{Npitch}$ whereas for the cylinder it is $ \frac{1}{3} \frac{\pi}{2} \text{Scale_factor} \text{Nr}^2 \text{Npitch}$, we realize after extracting out the common factors, we get in the parenthesis $\frac{Scale_factor}{3} + 1$. After some algebra, we get $\text{Nmrk} = \frac{\pi}{2} \text{Nr}^2 \text{Npitch} \text{Ngyro} \text{Nphi} (\frac{Sale_factor}{3} + 1)$, form which we can estimate the `Nr` and `Nz` for our first guess. In the actual code, we use a much more emperical constant instead of $\frac{\pi}{2}$ due to all the approximations that we made. Now the constant we add in the front is 0.65. 
2. We gradually increase/degress the `Nr` or `Nz` to calcuate the new total number of markers, depending on if the currently calculated `Nmrk` is too low or too high. 
3. We record the values of `Nr` and `Nz` where we have gotten the closest to the desired `Nmrk`, once we have see by further increasing or decresing `Nr` or `Nz`, the new total number of markers is father apart the desired `Nmrk` than our current "best" `Nr` and `Nz` values, we return the "best" `Nr` and `Nz` values. 

Note, since we only increase or decrse the `Nr` or `Nz` by one each time, when the number of markers is incrediblly large, the algorithm may take a long time to run. However, in any reasonable range for the number of markers that a cluster can calculate, the algorithm is able to finish in a reasonable time. 

## Artifically adding more makrers in pitch dimension in the center of the plasma 

- The arithmetics of linearly scaling the number of pitches as a function of $\rho$. 
  - Using rho_interpolator to get the rho at each RZ gird point. 
  - Make the gridRZ_Rhos_scale_pre as a function of rho: gridRZ_Rhos_scale_pre $(\rho) = 1 - \rho$. So that it is 1 at the center and 0 on the edge. 
  - Make this linear function scale with our max_Npitch_scale_factor $m$ and called it gridRZ_Rhos_scale_pre2: gridRZ_Rhos_scale_pre2 $(\rho) = m \times (1 - \rho)$. 
  - A `np.ceil` function is used to get the ceiling integers for the `gridRZ_Rhos_scale_pre3`. This is important because notice that there will not be any RZ gridpoint that lies at exactly at $\rho = 0$ nor exactly at $\rho = 1$. Therefore, the minimum value for the `gridRZ_Rhos_scale_pre3` will be 1 and the maximum number will be $m$. 
  - Finally multiplying the `gridRZ_Rhos_scale_pre3` with the base value for `Npitches` to get `pitch_array_scaled`. This will correctly make the number of pitch the base `Npitches` on the edge and $m \times \text{Npitches}$ at the center. 
- Looping to create the 5D grid in R, z, pitches, phi, and gyro_angles
  - First loop through `pitch_array_scaled` and add one to the `RZ_grid_index` each time. This is because we want the value of `pitch_array_scaled` at each of the `RZ_grid_index` since each of this value is the number of `Npitch` we should have. 
    - Loop through an array size of the `pitch_array_scaled[RZ_grid_index]`. Simultaneously define the `pitch_array` to be evenly spaced between the `Pitch_min` and `Pitch_max`. 
      - Loop through both phi and gyro_anles respectively
        - Store the 5 dimension of coordinates in 5 arrays. Also store the maximum number of pitches at each 5D grid point in `Npitches_scales`. 


# Documentation for geqdsk

A geqdsk file (also called EFIT G-EQDSK file) is a type of output file from the EFIT code (Equilibrium Fitting Code), which is used in plasma physics and nuclear fusion research to reconstruct the equilibrium state of a plasma in a tokamak device.

The geqdsk file contains data about the geometry and magnetic field of the plasma. Specifically, it includes information about the poloidal flux function, magnetic field data, current profile, pressure profile, and the boundary shape of the plasma.

The format of the geqdsk file is quite specific and has a particular structure. Generally, the file begins with header information followed by 2D arrays of grid data. This data is typically read using specialized routines in scientific software or custom scripts.

This file is a critical input for many analysis and modeling codes in fusion research as it provides a lot of important information about the plasma state in the tokamak device.


## The equilibrium class
``` python
        #Quantities read from the geqdsk file
        
        #A name for the particular equilibirum
        self.case           = None
        #Number of horizontal R grid points
        self.nw             = None
        #Number of vertical Z grid points
        self.nh             = None
        #Horizontal dimension in meters of computational box
        self.rdim           = None
        #Vertical dimension in meters of computational box
        self.zdim           = None
        #R in meters of vacuum toroidal magnetic field BCENTR
        self.rcentr         = None
        #Minimum R in meters of rectangular computational box
        self.rmin           = None
        #Z of center of computational box in meters
        self.zmid           = None
        #R of magnetic axis in meters
        self.rmaxis         = None
        #Z of magnetic axis in meters
        self.zmaxis         = None
        #poloidal flux at magnetic axis in Weber/rad
        self.simag          = None
        #poloidal flux at the plasma boundary in Weber/rad
        self.sibry          = None
        #Vacuum toroidal magnetic field in Tesla at RCENTR
        self.bcentr         = None
        #Plasma current in Ampere
        self.current        = None
        #Poloidal current function in m-T, F = RBT on flux grid
        self.fpol           = None
        #Plasma pressure in nt / m 2 on uniform flux grid
        self.pres           = None
        #FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid
        self.ffprime        = None
        #P'(psi) in (nt/m2)/(Weber/rad) on uniform flux grid
        self.pprime         = None
        #Poloidal flux in Weber / rad on the rectangular grid points
        self.psirz          = None
        #q values on uniform flux grid from axis to boundary
        self.qpsi           = None
        #Number of points in last closed flux survace
        self.nbbbs          = None
        #Number of points in the limiter
        self.limitr         = None
        #R points of the last closed flux surface
        self.rlcfs          = None
        #Z ponts of the last closed flux surface
        self.zlcfs          = None
        #R points of the limiter
        self.rlim           = None
        #Z points of the limiter
        self.zlim           = None

        #Calculated quantities

        #The maximum radius of the domain
        self.rmax           = None
        #The radial resolution
        self.dr             = None
        #The minimum z of the domain
        self.zmin           = None
        #The maximum z of the domain
        self.zmax           = None
        #The vertical resolution
        self.dz             = None
        #The r points of the computational grid
        self.rGrid          = None
        #The z points of the computational grid
        self.zGrid          = None
        #The square root of the normalized poloidal flux
        self.psiPolSqrt     = None
```