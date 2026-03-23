import numpy as np
import pandas as pd
import os
from exovista import load_stars
from exovista import generate_starspots
from exovista import generate_planets
from exovista import generate_disks
from exovista import generate_scene
from exovista import read_solarsystem
from exovista import Settings
from exovista import make_starmap

# ExoVista v2.4

# Generates a universe of simulated planetary systems based on a stellar target list.

parallel = True
maxcores = 1000000
try: import multiprocessing
except:
    print('multiprocessing module not available. Continuing with serial processing.')
    parallel = False    

if __name__ == '__main__':

    settings = Settings.Settings(timemax=10.0, output_dir='output') # "standard" configuration
    
    #target_list_file = 'master_target_list-usingDR2-50_pc.txt'
    target_list_file = 'target_list8.txt'
    stars, nexozodis = load_stars.load_stars(target_list_file)
    print('\n{0:d} stars in model ranges.'.format(len(stars)))
    
    spots = generate_starspots.generate_spots(stars, settings)
    
    planets, albedos = generate_planets.generate_planets(stars, settings, usebins=True, addearth=True)
    disks, compositions = generate_disks.generate_disks(stars, planets, settings, nexozodis=nexozodis)
    print('Generating scenes. (This may take a while.)')
    
    if parallel:
        cores = min(maxcores,os.cpu_count())
        cores = min(cores,len(stars))
        percore = int(np.ceil(len(stars)/cores))
        if percore > 1 and percore*(cores-1) >= len(stars): percore -= 1
        
        pool = multiprocessing.Pool(cores)
        
        inputs = []
        for i in range(0,cores):
            imin = i*percore
            imax = (i+1)*percore
            inputs.append([stars.iloc[imin:imax],spots.iloc[imin:imax],planets[imin:imax],disks[imin:imax],albedos[imin:imax],compositions[imin:imax],settings])
            
        pool.starmap(generate_scene.generate_scene, [inputs[j][:] for j in range(0,cores)])
        pool.close()
        pool.join()
    else:
        generate_scene.generate_scene(stars,spots,planets,disks,albedos,compositions,settings)

    print('Done')

'''
Notices and Disclaimers

“Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  No copyright is claimed in the United States under Title 17, U.S. Code.  All Other Rights Reserved.”
 
Disclaimer:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
'''
