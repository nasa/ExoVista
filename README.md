# ExoVista v2.5
October 2024
Alex Howe & Chris Stark
NASA Goddard Space Flight Center

ExoVista is a hybrid Python/C++ software package based on an earlier IDL/C iteration that generates synthetic exoplanetary systems. ExoVista models exoplanet atmospheres in reflected light, stellar spectra using Kurucz stellar atmosphere models, and debris disks in scattered light using realistic spatial distributions and optical properties. Planets can be drawn from measured/extrapolated Kepler occurrence rates (Dulz et al. 2020) and are checked for basic stability criteria; debris disks are dynamically quasi-self-consistent with the underlying planetary system. All bodies are integrated with a Bulirsch-Stoer integrator to determine their barycentric velocities, positions, and orbits. The output product is a multi-extension fits file that contains all of the information necessary to generate a spectral data cube of the system for direct imaging simulations of coronagraphs/starshades, as well as position/velocity data for simulation of RV, astrometric, and transit (pending) data sets.

To run the main modules of ExoVista, you must have a Python interpreter (Python 3.8 or higher recommended). You will also need to have installed, in addition to the “standard” suite of Python modules, the Python packages scipy, astropy, and cython. The multiprocessing package is also needed if you wish to use ExoVista with parallel processing, although ExoVista can run as a serial code without it.

ExoVista also requires a C++ compiler.
For Linux users, g++ is usually available.
For Mac OS users, it is recommended to install Apple’s XCode to obtain a compiler.
For Windows users, it is recommended to use the Microsoft Visual C/C++ compiler.
    Other tools such as those found in MinGW or Cygwin may work, but these have not been tested.

To install ExoVista, download the current version of the ExoVista package from the Github repository into the desired directory on your local machine.

Open a terminal window, navigate to the “src” subdirectory in the directory containing the ExoVista code, and compile the disk imaging module by typing the following command:

    python setup.py build_ext --inplace
	
Cython should automatically call your C++ compiler and generate the files “wrapImage.cpp” and “wrapIntegrator.cpp” on your machine in the src subdirectory. If this fails, you may need to uncomment and edit the following lines in “setup.py” to reflect your local compiler environment (see the Cython documentation for more information):

    import os
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'

Once the wrapImage and wrapIntegrator modules are built successfully, you will be ready to run all ExoVista modules.

Notices and Disclaimers:

“Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  No copyright is claimed in the United States under Title 17, U.S. Code.  All Other Rights Reserved.”
 
Disclaimer:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.