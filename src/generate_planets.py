import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from scipy.stats import norm
from src.constants import pllabel, maxnplanets
from src import Settings
import matplotlib.pyplot as plt

settings = Settings.Settings()

# Read in bins for planet types.
fbound = open('planetbins.dat','r')
line = fbound.readline().strip()
if line!='Radius':
    print('Error: wrong format in planetbins.dat')
    exit()
rbound = np.array(fbound.readline().split()).astype(float)

atype = fbound.readline().strip()
alines = fbound.readlines()
if len(alines)!=len(rbound)-1:
    print('Error: wrong format in planetbins.dat')
    exit()
alen = len(alines[0].split())
abound = np.zeros((len(alines),alen))
for i in range(0,len(alines)):
    abound[i] = np.array(alines[i].split()).astype(float)
if atype=='Flux': abound = 1./np.sqrt(abound)


def mass_to_radius(M, rng, a=None, randrad=False):
    # Calculates radius in Earth radii based on Chen & Kipping (2017)
    if a is None: a = np.full(M.shape,1.)
    
    R = 1.008 * M**0.279
    i = np.where(M<=2.04)[0]
    if randrad: R[i] *= rng.lognormal(sigma = np.log(1.0403), size = len(i))
    
    j = np.where(M>2.04)[0]
    R[j] = 1.008 * 2.04**0.279 * (M[j]/2.04)**0.589
    i = [k for k in range(0,len(M)) if M[k]>2.04 and M[k]<=131.6]
    if randrad: R[i] *= rng.lognormal(sigma = np.log(1.146), size = len(i))
    
    j = np.where(M>131.6)[0]
    R[j] = 1.008 * 2.04**0.279 * (131.6/2.04)**0.589 * (M[j]/131.6)**-0.044
    if randrad:
        R[j] *= rng.lognormal(sigma = np.log(1.0737), size = len(j))
        i = [k for k in range(0,len(M)) if R[k]>14.31 and a[k]>0.091]
        R[i] = 14.31
    
    return R


def radius_to_mass(R):
    # Calculates mass in Earth masses based on Chen & Kipping (2017)
    # WARNING: does not return valid results for Saturns and up.
    # Not monotonic for R>12.22 R_E
    # Returns the value on the lower-mass branch.
    M = np.zeros(len(R))
    
    i = [j for j in range(0,len(R)) if R[j]<=1.008 * 2.04**0.279]
    M[i] = (R[i]/1.008)**(1./0.279)
    i = [j for j in range(0,len(R)) if R[j]>1.008 * 2.04**0.279]
    M[i] = (R[i]/(1.008 * 2.04**0.279))**(1./0.589) * 2.04
    #i = [j for j in range(0,len(R)) if R[j]>12.22] # R(M) is not monotonic in this range.
    #M[i] = -1.
    i = [j for j in range(0,len(R)) if R[j]>1.008 * 2.04**0.279 * 131.6**0.589]
    M[i] = (R[i]/(1.008 * 2.04**0.279 * (131.6/2.04)**0.589))**(1./-0.044) * 131.6

    return M


def generate_planets(stars, settings, bound='', nomcdraw=False, addearth=False, usebins=False, subdivide=1):
    print('Generating planets...')
    settings = settings
    rng = np.random.default_rng(seed=settings.seed)
    
    subdivide = max(1,min(subdivide,10))
    
    if settings.emax < settings.emin:
        print('Error: emax must be greater than emin.')
        return
    
    if settings.imax < settings.imin:
        print('Error: imax must be greater than imin.')
        return

    cossysimin = np.cos(settings.sysimin * np.pi/180.)
    cossysimax = np.cos(settings.sysimax * np.pi/180.)
    
    nstars = len(stars)
    
    # Check that "stars" has the necessary tags
    if 'Lstar' not in stars.head():
        print('Error: stars input must be a DataFrame including column \"Lstar\"')
        exit()
        return

    if usebins and settings.randrad:
        print('Error: usebins keyword is not compatible with the randomized mass-radius relation (randrad)')
        exit()
        return
    
    # Define planet structure entry, as described below
    # Default semi-major axis set to 1.e20 for sorting purposes.
    plorb = np.zeros((nstars,maxnplanets,8))
    plorb[:,:,pllabel.index('a')] = 1.e20
    
    '''
    M: mass of planet in Earth masses
    R: radius of planet in Earth radii
    a: semi-major axis (physical, not projected) in AU
    e: eccentricity (physical, not projected)
    i: inclination relative to system midplane
    longnode: longitude of the ascending node
    argperi: arg. of pericenter
    meananom: mean anomaly of the planet at t=0

    plalbedo: relative path to geometric albedo file
    '''
    
    # Add position angle and inclination of host stars if needed.
    if 'PA' not in stars.head():
        temp = np.zeros(nstars)
        stars['PA'] = temp
    if 'I' not in stars.head():
        temp = np.zeros(nstars)
        stars['I'] = temp
  
    # Randomly assign system midplane orientation
    stars['PA'] = rng.random(nstars) * (settings.sysPAmax-settings.sysPAmin) + settings.sysPAmin

    stars['I'] = np.arccos((rng.random(nstars) * (cossysimin-cossysimax) + cossysimax)) * 180./np.pi
    temprand = rng.random(nstars)
    j = np.where(temprand < 0.5)[0]
    stars['I'].loc[j] -= 180.
        
    '''
    Monte Carlo draw of planets
    To get "exactly" the right number of planets per star we'd do this:
    expected = round(occrates * nstars)
    But it fails if nstars is small or occurrence rates are small in a given bin.
    So instead, we do a Monte Carlo draw either with larger bins or the whole map.
    '''
    
    # load occurrence rates in mass - semimajor axis space
    occrates, medge, redge, aedge, mbins, rbins, abins = load_occurrence_rates(bound=bound, mass=True, usebins=usebins)
    
    expected = np.zeros((len(abins),len(mbins)))
    randgrid = np.zeros((len(abins),len(mbins),nstars))
    
    if not usebins:
        for ii in range(0,len(abins)):
            for jj in range(0,len(mbins)):
                randgrid[ii,jj] = rng.random(nstars)
                expected[ii,jj] = len(np.where(randgrid[ii,jj] < occrates[ii,jj])[0])
    
    else:        
        mbound = radius_to_mass(rbound)
        mbound[-1] = medge[-1]
        
        for i in range(0,len(mbound)-1):
            try: mmin = np.where(mbins>=mbound[i])[0][0]
            except: mmin = 0
            try: mmax = np.where(mbins>=mbound[i+1])[0][0]
            except: mmax = len(mbins)
            
            for j in range(0,len(abound)-1):
                try: amin = np.where(abins>=abound[i,j])[0][0]
                except: amin = 0
                try: amax = np.where(abins>=abound[i,j+1])[0][0]
                except: amax = len(abins)
                
                expected_in_bin = int(np.round(np.sum(occrates[amin:amax,mmin:mmax])*nstars))
                if expected_in_bin == 0: continue
                
                for ii in range(amin,amax):
                    for jj in range(mmin,mmax):
                        randgrid[ii,jj] = rng.random(nstars)
                        expected[ii,jj] = len(np.where(randgrid[ii,jj] < occrates[ii,jj])[0])
                planets_in_bin = int(np.sum(expected[amin:amax,mmin:mmax]))
                diff = expected_in_bin-planets_in_bin
                
                if diff==0: continue
                elif diff>0:
                    for kk in range(0,diff):
                        ind = np.unravel_index(np.argmax(randgrid[amin:amax,mmin:mmax,:]),(amax-amin,mmax-mmin,nstars))
                        ind2 = np.argmax(randgrid[amin:amax,mmin:mmax,:])
                        randgrid[amin+ind[0],mmin+ind[1],ind[2]] = 0.
                elif diff<0:
                    while diff<0:
                        ind = np.unravel_index(np.argmin(randgrid[amin:amax,mmin:mmax,:]),(amax-amin,mmax-mmin,nstars))
                        if randgrid[amin+ind[0],mmin+ind[1],ind[2]] < occrates[amin+ind[0],mmin+ind[1]]: diff+=1
                        randgrid[amin+ind[0],mmin+ind[1],ind[2]] = 1.
                        
                for ii in range(amin,amax):
                    for jj in range(mmin,mmax):
                        expected[ii,jj] = len(np.where(randgrid[ii,jj] < occrates[ii,jj])[0])

    nexpected = int(np.sum(expected))

    hillsphere_flag = np.full(nstars, True) # improves run time efficiency: calculate new planets only if True.
    nplanets = 0
    n_nochange = 0
    jj = 0
    
    while nplanets < nexpected and n_nochange < 50 and jj < 200:
        print('Iteration number: ',jj)
        #print('{0:d} planets expected.'.format(nexpected))
        prev_nplanets = nplanets
     
        # First, add some random planets based on occurrence rates
        plorb, hillsphere_flag = add_planets(stars, plorb, expected, medge, aedge, hillsphere_flag, settings.randrad, rng)

        # Number of stars with new planets
        nnew = len(np.where(hillsphere_flag)[0])
        #print('{0:d} systems changed.'.format(nnew)) 
        
        # Now remove any unstable ones
        plorb = remove_unstable_planets(stars, plorb, hillsphere_flag)
        
        # Number of stable planets
        nplanets = 0
        for i in range(0,nstars):
            nplanets += len(np.where(plorb[i,:,pllabel.index('R')] > 0)[0])
        #print('{0:d} stable planets.'.format(nplanets))
        
        dnp = nplanets-prev_nplanets
        if dnp<=0: n_nochange += 1   # Ends loop if no increase for 50 iterations.
        else: n_nochange = 0         # (No change gets stuck for small target lists.)
        jj += 1
        #print('No increase for {0:d} iterations.\n'.format(n_nochange))
        
    # Erase semi-major axis of non-existant planets
    for i in range(0,nstars):
        for j in range(0,maxnplanets):
            if plorb[i,j,pllabel.index('R')]==0.: plorb[i,j,pllabel.index('a')]=0.        
        
    # Final number of planets
    nplanets = 0
    for i in range(0,nstars):
        nplanets += len(np.where(plorb[i,:,pllabel.index('R')] > 0)[0])

    print('{0:0.1f} planets expected from occurrence rates'.format(np.sum(occrates)*nstars))
    if not nomcdraw: print('{0:0.1f} planets expected from Monte Carlo draw of occurrence rates'.format(nexpected))
    print('{0:d} planets generated after imposing stability limits.'.format(nplanets))
    
    # Randomly assign all planets an albedo file
    print('Assigning planet types/albedos based albedo_list.csv...')
    albedos = assign_albedo_file(stars, plorb, rng)

    # if True, adds a zero-mass Earth twin at quadrature for yield calculations
    if addearth:
        for i in range(0,nstars):
            inew = len(np.where(plorb[i,:,pllabel.index('R')] > 0)[0])
            anew = np.sqrt(stars['Lstar'].iloc[i])
            plorb[i,inew] = np.array([1.e-6,1.,anew,0.00,0.0005,-11.261,114.208,-102.947])
            albedos[i].append('Earth')
            
    print('generate_planets() done')
    
    return plorb, albedos
    

def load_occurrence_rates(subdivide=1, bound='', mass=True, usebins=False):

    # Note: I turned this into a wrapper for separate read-in functions because
    # I could not follow the "use r and p as variables no matter what quantities we're working with" notation.
    
    '''
    load_occurrence_rates.py
    This procedure generates a 2D array of occurrence rates in
    (R_planet,a_planet) space. The bin size of each pixel is controlled
    by the keyword npix and has constant dln(R) dln(P). Occurrence rates
    returned are integrated over the bin.
    
    By default, the nominal occurrence rates of Dulz & Plavchan
    are returned.
    Two tables are provided.
    The 'Mass' table is in M-a space.
    The 'Radius' table is in R-P space.
    Both require a conversion to get the correct output in R-a space.
    
    Notes:
    - the Dulz & Plavchan occurrence rate files are used to set the grid
    spacing.
    - Dulz & Plavchan occ rates are not analytic, so we must interpolate
    their results to the desired resolution
    '''

    tag = ''
    if mass: tag = 'Mass'
    tag = 'Radius'
    subdivide = min(max(1,subdivide),10) # Keep the occurrence rate grid a reasonable size.
    
    filename = 'occurrence_rates/NominalOcc_' + tag + '.csv'
    if bound=='lower': filename = 'occurrence_rates/PessimisticOcc_' + tag + '.csv'
    if bound=='upper': filename = 'occurrence_rates/OptimisticOcc_'  + tag + '.csv'
    
    return load_occurrence_rates_convolved('occurrence_rates/NominalOcc_', subdivide=subdivide, usebins=usebins)


def load_occurrence_rates_convolved(filepath, subdivide=1, usebins=False):
    # Note: I'm not sure if the routine is actually capable of subdividing in its current state.
    
    # split between RV and transit rates in Dulz et al. (2020)
    rsplit = 10.0
    msplit = (rsplit/(1.008*2.04**0.279))**(1/0.589)*2.04

    # Read in tables from files
    massfile = filepath + 'Mass.csv'
    radfile = filepath + 'Radius.csv'
    
    dataM = pd.read_csv(massfile)
    dataM.columns = dataM.columns.str.replace(' ','')
    
    dataR = pd.read_csv(radfile)
    dataR.columns = dataR.columns.str.replace(' ','')
    
    tempoccrateM = dataM['Occ'].values
    tlenM = len(tempoccrateM)
    
    mmin = dataM['M_min'].values
    m    = dataM['M_mid'].values
    mmax = dataM['M_max'].values
    
    amin = dataM['a_min'].values
    a    = dataM['a_mid'].values
    amax = dataM['a_max'].values
    
    tempoccrateR = dataR['Occ'].values
    tlenR = len(tempoccrateR)
    
    rmin = dataR['R_min'].values
    r    = dataR['R_mid'].values
    rmax = dataR['R_max'].values    
    
    # Create x and y-axis vectors
    mminvec = np.sort(np.unique(mmin))
    mvec    = np.sort(np.unique(m))
    mmaxvec = np.sort(np.unique(mmax))
    
    rminvec = np.sort(np.unique(rmin))
    rvec    = np.sort(np.unique(r))
    rmaxvec = np.sort(np.unique(rmax))
    
    aminvec = np.sort(np.unique(amin))
    avec    = np.sort(np.unique(a))
    amaxvec = np.sort(np.unique(amax))

    # Check that the tables are logarithmically spaced
    # And compute logarithmic pixel size: d^2 occrate / dlna dlnr (or dlnm)
    lnm = np.log(mvec)
    difflnm = lnm[1:len(lnm)] - lnm[0:len(lnm)-1]
    if np.abs(np.max(difflnm)-np.min(difflnm)) / np.abs(np.min(difflnm)) > 0.01:
        print('Mass spacing for occurrence rates must be constant in ln space')
        exit()
    dlnm = difflnm[0]
    
    lnr = np.log(rvec)
    difflnr = lnr[1:len(lnr)] - lnr[0:len(lnr)-1]
    if np.abs(np.max(difflnr)-np.min(difflnr)) / np.abs(np.min(difflnr)) > 0.01:
        print('Radius spacing for occurrence rates must be constant in ln space')
        exit()
    dlnr = difflnr[0]
    
    lna = np.log(avec)
    difflna = lna[1:len(lna)] - lna[0:len(lna)-1]
    if np.abs(np.max(difflna)-np.min(difflna)) / np.abs(np.min(difflna)) > 0.01:
        print('SMA spacing for occurrence rates must be constant in ln space')
        exit()
    dlna = difflna[0]
    
    # Assemble the combined m-space plus r-space output grid with appropriate bin edges
    # Note: default input is 100x100, default output is 100x129
    # Because the m-space and r-space grids don't line up after conversions between the two
    
    npix = len(avec)*subdivide
    mbins = np.exp(np.linspace(np.log(mvec[0]),np.log(mvec[-1]),npix))
    rbins = np.exp(np.linspace(np.log(rvec[0]),np.log(rvec[-1]),npix))
    abins = np.exp(np.linspace(np.log(avec[0]),np.log(avec[-1]),npix))
    
    medge_temp = np.zeros(len(mbins)+1)
    medge_temp[0] = mbins[0] * np.sqrt(mbins[0]/mbins[1])
    medge_temp[-1] = mbins[-1] * np.sqrt(mbins[-1]/mbins[-2])
    medge_temp[1:-1] = np.sqrt(mbins[:-1]*mbins[1:])
    medge_temp = np.append(medge_temp,msplit)
    if usebins: medge_temp = np.append(medge_temp,radius_to_mass(rbound[1:-1]))
    medge_temp.sort()
    mmid_temp = np.sqrt(medge_temp[0:-1]*medge_temp[1:])

    temp_rng = np.random.default_rng() # Dummy variable not used in this function call
    rmid_alt = mass_to_radius(mmid_temp,temp_rng)
    redge_alt = mass_to_radius(medge_temp,temp_rng)
    
    redge_temp = np.zeros(len(rbins)+1)
    redge_temp[0] = rbins[0] * np.sqrt(rbins[0]/rbins[1])
    redge_temp[-1] = rbins[-1] * np.sqrt(rbins[-1]/rbins[-2])
    redge_temp[1:-1] = np.sqrt(rbins[:-1]*rbins[1:])
    redge_temp = np.append(redge_temp,rsplit)
    if usebins: redge_temp = np.append(redge_temp,rbound[1:-1])
    redge_temp.sort()
    rmid_temp = np.sqrt(redge_temp[0:-1]*redge_temp[1:])

    mmid_alt = radius_to_mass(rmid_temp)
    medge_alt = radius_to_mass(redge_temp)
    
    aedge = np.zeros(len(abins)+1)
    aedge[0] = abins[0] * np.sqrt(abins[0]/abins[1])
    aedge[-1] = abins[-1] * np.sqrt(abins[-1]/abins[-2])
    aedge[1:-1] = np.sqrt(abins[:-1]*abins[1:])
    if usebins: aedge = np.append(aedge,np.unique(abound[:,1:-1].flatten()))
    aedge.sort()
    amid = np.sqrt(aedge[0:-1]*aedge[1:])
    asize = aedge[1:]/aedge[0:-1]
    
    risplit = np.where(rmid_temp>=rsplit)[0][0]
    misplit = np.where(mmid_temp>=msplit)[0][0]
    
    mmid  = np.concatenate((mmid_alt[0:risplit],  mmid_temp[misplit:]))
    medge = np.concatenate((medge_alt[0:risplit], medge_temp[misplit:]))
    rmid  = np.concatenate((rmid_temp[0:risplit], rmid_alt[misplit:]))
    redge = np.concatenate((redge_temp[0:risplit],redge_alt[misplit:]))
    
    rsize = redge[1:]/redge[0:-1]
    msize = medge[1:]/medge[0:-1]
    
    # Turn the input occurrence rates into a 2D array of the correct shape
    # Padded with a border for convolution with the dispersion relation
    occrateRpad = np.zeros((len(avec),len(rvec)+20))
    occrateMpad = np.zeros((len(avec),len(mvec)+20))
    for i in range(0,tlenM):
        ia = np.where(avec==a[i])[0][0]
        im = np.where(mvec==m[i])[0][0]
        ir = np.where(rvec==r[i])[0][0]
        occrateRpad[ia,ir+10] = tempoccrateR[i]
        occrateMpad[ia,im+10] = tempoccrateM[i]
        
    rvecPad = np.zeros(len(rvec)+20)
    rvecPad[10:-10] = rvec
    rminvecPad = np.zeros(len(rminvec)+20)
    rminvecPad[10:-10] = rminvec
    rmaxvecPad = np.zeros(len(rmaxvec)+20)
    rmaxvecPad[10:-10] = rmaxvec
    
    mvecPad = np.zeros(len(mvec)+20)
    mvecPad[10:-10] = mvec
    mminvecPad = np.zeros(len(mminvec)+20)
    mminvecPad[10:-10] = mminvec
    mmaxvecPad = np.zeros(len(mmaxvec)+20)
    mmaxvecPad[10:-10] = mmaxvec
    
    for i in range(0,10):
        occrateRpad[:,i] = occrateRpad[:,10]
        occrateRpad[:,-i-1] = occrateRpad[:,-11]
        occrateMpad[:,i] = occrateMpad[:,10]
        occrateMpad[:,-i-1] = occrateMpad[:,-11]
        
        rvecPad[9-i] = rvecPad[10-i] / np.exp(dlnr)
        rvecPad[-10+i] = rvecPad[-11+i] * np.exp(dlnr)
        rminvecPad[9-i] = rminvecPad[10-i] / np.exp(dlnr)
        rminvecPad[-10+i] = rminvecPad[-11+i] * np.exp(dlnr)
        rmaxvecPad[9-i] = rmaxvecPad[10-i] / np.exp(dlnr)
        rmaxvecPad[-10+i] = rmaxvecPad[-11+i] * np.exp(dlnr)
        
        mvecPad[9-i] = mvecPad[10-i] / np.exp(dlnm)
        mvecPad[-10+i] = mvecPad[-11+i] * np.exp(dlnm)
        mminvecPad[9-i] = mminvecPad[10-i] / np.exp(dlnm)
        mminvecPad[-10+i] = mminvecPad[-11+i] * np.exp(dlnm)
        mmaxvecPad[9-i] = mmaxvecPad[10-i] / np.exp(dlnm)
        mmaxvecPad[-10+i] = mmaxvecPad[-11+i] * np.exp(dlnm)

    # Copy the "directly"-measured regions of the tables into new grids
    CVoccrateR = np.zeros((len(avec),len(rvec)))
    CVoccrateM = np.zeros((len(avec),len(mvec)))
    
    newR = mass_to_radius(mvec,temp_rng)
    newM = radius_to_mass(rvec)
    
    newRpad = np.log(mass_to_radius(mvecPad,temp_rng))
    newMpad = np.log(radius_to_mass(rvecPad))
    
    newRmin = np.log(mass_to_radius(mminvec,temp_rng))
    newMmin = np.log(radius_to_mass(rminvec))
    
    newRmax = np.log(mass_to_radius(mmaxvec,temp_rng))
    newMmax = np.log(radius_to_mass(rmaxvec))
    
    convMgrid = np.zeros((len(mvec)+20,len(mvec)+20))
    convRgrid = np.zeros((len(rvec)+20,len(rvec)+20))
    
    for i in range(0,len(rvec)):
        if rvec[i] < rsplit: CVoccrateR[:,i] = occrateRpad[:,i+10]
        if mvec[i] > msplit: CVoccrateM[:,i] = occrateMpad[:,i+10]

    # Fill in the computed regions of the tables by convolving the opposite grid with the dispersion relation
    for i in range(0,len(mvecPad)):
        if mvecPad[i]<=np.log(2.04): sigmaR = np.log(1.0403)
        elif mvecPad[i]<=np.log(131.6): sigmaR = np.log(1.1460)
        else: sigmaR = np.log(1.0737)

        if np.exp(newRpad[i]) < rsplit: continue

        for j in range(0,len(rvecPad)):
            convRgrid[i,j] = abs(norm.cdf(np.log(rmaxvecPad[j]),newRpad[i],sigmaR)-norm.cdf(np.log(rminvecPad[j]),newRpad[i],sigmaR))
            if 10<=j<len(rvec)+10: CVoccrateR[:,j-10] += occrateMpad[:,i]*convRgrid[i,j]
        
    for i in range(0,len(rvecPad)):
        if newMpad[i]<=np.log(2.04): sigmaM = np.log(1.0403)/0.279
        elif newMpad[i]<=np.log(131.6): sigmaM = np.log(1.1460)/0.589
        else: sigmaM = np.log(1.0737)/0.044
        
        if np.exp(newMpad[i]) > msplit: continue
        
        for j in range(0,len(mvecPad)):            
            convMgrid[i,j] = abs(norm.cdf(np.log(mmaxvecPad[j]),newMpad[i],sigmaM)-norm.cdf(np.log(mminvecPad[j]),newMpad[i],sigmaM))
            if 10<=j<len(mvec)+10: CVoccrateM[:,j-10] += occrateRpad[:,i]*convMgrid[i,j]
    
    # Scale by logarithmic pixel size
    CVoccrateR /= (dlnr * dlna)
    CVoccrateM /= (dlnm * dlna)
    
    # Interpolate to the output grid
    mvec = np.log(mvec)
    rvec = np.log(rvec)
    avec = np.log(avec)
    
    occ_interpR = interp2d(rvec, avec, CVoccrateR, kind='cubic')
    occ_interpM = interp2d(mvec, avec, CVoccrateM, kind='cubic')

    # Combine the scaled grids into the final combined grid
    finalsplit = np.where(rmid>=rsplit)[0][0]
    newoccrate = np.zeros((len(amid),len(mmid)))
    newoccrate[:,0:finalsplit] = occ_interpR(np.log(rmid[0:finalsplit]),np.log(amid))
    newoccrate[:,finalsplit:] = occ_interpM(np.log(mmid[finalsplit:]),np.log(amid))
    newoccrate = newoccrate * (newoccrate > 0) # Occurence rate cannot be negative

    #Rescale to the final pixel size
    newoccrate[:,0:finalsplit] *= np.outer(np.log(asize), np.log(rsize))[:,0:finalsplit]
    newoccrate[:,finalsplit:] *= np.outer(np.log(asize), np.log(msize))[:,finalsplit:]
    
    if np.max(newoccrate) > 1:
        print('Error: max occurrence rate > 1. Try increasing the grid size.')
        exit()
        # Easier to see the error message without a return.
        
    return newoccrate, medge, redge, aedge, mmid, rmid, amid


def load_occurrence_rates_combined(filepath, subdivide=1, usebins=False):
    
    # split between RV and transit rates in Dulz et al. (2020)
    
    rsplit = 10.0
    msplit = (rsplit/(1.008*2.04**0.279))**(1/0.589)*2.04

    # Read in tables from files
    
    massfile = filepath + 'Mass.csv'
    radfile = filepath + 'Radius.csv'
    
    dataM = pd.read_csv(massfile)
    dataM.columns = dataM.columns.str.replace(' ','')
    
    dataR = pd.read_csv(radfile)
    dataR.columns = dataR.columns.str.replace(' ','')
    
    tempoccrateM = dataM['Occ'].values
    tlenM = len(tempoccrateM)
    
    mmin = dataM['M_min'].values
    m    = dataM['M_mid'].values
    mmax = dataM['M_max'].values
    
    amin = dataM['a_min'].values
    a    = dataM['a_mid'].values
    amax = dataM['a_max'].values
    
    tempoccrateR = dataR['Occ'].values
    tlenR = len(tempoccrateR)
    
    rmin = dataR['R_min'].values
    r    = dataR['R_mid'].values
    rmax = dataR['R_max'].values    
    
    # Create x and y-axis vectors
    mminvec = np.sort(np.unique(mmin))
    mvec    = np.sort(np.unique(m))
    mmaxvec = np.sort(np.unique(mmax))
    
    rminvec = np.sort(np.unique(rmin))
    rvec    = np.sort(np.unique(r))
    rmaxvec = np.sort(np.unique(rmax))
    
    aminvec = np.sort(np.unique(amin))
    avec    = np.sort(np.unique(a))
    amaxvec = np.sort(np.unique(amax))

    # Check that the tables are logarithmically spaced
    # And compute logarithmic pixel size: d^2 occrate / dlna dlnr (or dlnm)
    lnm = np.log(mvec)
    difflnm = lnm[1:len(lnm)] - lnm[0:len(lnm)-1]
    if np.abs(np.max(difflnm)-np.min(difflnm)) / np.abs(np.min(difflnm)) > 0.01:
        print('Mass spacing for occurrence rates must be constant in ln space')
        exit()
    dlnm = difflnm[0]
    
    lnr = np.log(rvec)
    difflnr = lnr[1:len(lnr)] - lnr[0:len(lnr)-1]
    if np.abs(np.max(difflnr)-np.min(difflnr)) / np.abs(np.min(difflnr)) > 0.01:
        print('Radius spacing for occurrence rates must be constant in ln space')
        exit()
    dlnr = difflnr[0]
    
    lna = np.log(avec)
    difflna = lna[1:len(lna)] - lna[0:len(lna)-1]
    if np.abs(np.max(difflna)-np.min(difflna)) / np.abs(np.min(difflna)) > 0.01:
        print('SMA spacing for occurrence rates must be constant in ln space')
        exit()
    dlna = difflna[0]

    # Assemble the combined m-space plus r-space output grid with appropriate bin edges
    # Note: default input is 100x100, default output is 100x129
    # Because the m-space and r-space grids don't line up after conversions between the two
    
    npix = len(avec)*subdivide
    mbins = np.exp(np.linspace(np.log(mvec[0]),np.log(mvec[-1]),npix))
    rbins = np.exp(np.linspace(np.log(rvec[0]),np.log(rvec[-1]),npix))
    abins = np.exp(np.linspace(np.log(avec[0]),np.log(avec[-1]),npix))
    
    medge_temp = np.zeros(len(mbins)+1)
    medge_temp[0] = mbins[0] * np.sqrt(mbins[0]/mbins[1])
    medge_temp[-1] = mbins[-1] * np.sqrt(mbins[-1]/mbins[-2])
    medge_temp[1:-1] = np.sqrt(mbins[:-1]*mbins[1:])
    medge_temp = np.append(medge_temp,msplit)
    if usebins: medge_temp = np.append(medge_temp,radius_to_mass(rbound[1:-1]))
    medge_temp.sort()
    mmid_temp = np.sqrt(medge_temp[0:-1]*medge_temp[1:])

    temp_rng = np.random.default_rng()
    rmid_alt = mass_to_radius(mmid_temp,temp_rng)
    redge_alt = mass_to_radius(medge_temp,temp_rng)
    
    redge_temp = np.zeros(len(rbins)+1)
    redge_temp[0] = rbins[0] * np.sqrt(rbins[0]/rbins[1])
    redge_temp[-1] = rbins[-1] * np.sqrt(rbins[-1]/rbins[-2])
    redge_temp[1:-1] = np.sqrt(rbins[:-1]*rbins[1:])
    redge_temp = np.append(redge_temp,rsplit)
    if usebins: redge_temp = np.append(redge_temp,rbound[1:-1])
    redge_temp.sort()
    rmid_temp = np.sqrt(redge_temp[0:-1]*redge_temp[1:])

    mmid_alt = radius_to_mass(rmid_temp)
    medge_alt = radius_to_mass(redge_temp)
    
    aedge = np.zeros(len(abins)+1)
    aedge[0] = abins[0] * np.sqrt(abins[0]/abins[1])
    aedge[-1] = abins[-1] * np.sqrt(abins[-1]/abins[-2])
    aedge[1:-1] = np.sqrt(abins[:-1]*abins[1:])
    if usebins: aedge = np.append(aedge,np.unique(abound[:,1:-1].flatten()))
    aedge.sort()
    amid = np.sqrt(aedge[0:-1]*aedge[1:])
    asize = aedge[1:]/aedge[0:-1]
    
    risplit = np.where(rmid_temp>=rsplit)[0][0]
    misplit = np.where(mmid_temp>=msplit)[0][0]
    
    mmid  = np.concatenate((mmid_alt[0:risplit],  mmid_temp[misplit:]))
    medge = np.concatenate((medge_alt[0:risplit], medge_temp[misplit:]))
    rmid  = np.concatenate((rmid_temp[0:risplit], rmid_alt[misplit:]))
    redge = np.concatenate((redge_temp[0:risplit],redge_alt[misplit:]))
    
    rsize = redge[1:]/redge[0:-1]
    msize = medge[1:]/medge[0:-1]
    
    # Turn the input occurrence rates into a 2D array
    occrateR = np.zeros((len(avec),len(rvec)))
    occrateM = np.zeros((len(avec),len(mvec)))
    for i in range(0,tlenM):
        ia = np.where(avec==a[i])[0][0]
        im = np.where(mvec==m[i])[0][0]
        ir = np.where(rvec==r[i])[0][0]
        occrateR[ia,ir] = tempoccrateR[i]
        occrateM[ia,im] = tempoccrateM[i]

    # Scale by logarithmic pixel size
    rjsplit = np.where(rvec>=rsplit)[0][0]
    occrateR /= (dlnr * dlna)
    occrateM /= (dlnm * dlna)
    
    # Interpolate to the output grid
    mvec = np.log(mvec)
    rvec = np.log(rvec)
    avec = np.log(avec)
    
    occ_interpR = interp2d(rvec, avec, occrateR, kind='cubic')
    occ_interpM = interp2d(mvec, avec, occrateM, kind='cubic')
    
    risplit = np.where(rmid>=rsplit)[0][0]
    newoccrate = np.zeros((len(amid),len(mmid)))
    newoccrate[:,0:risplit] = occ_interpR(np.log(rmid[0:risplit]),np.log(amid))
    newoccrate[:,risplit:] = occ_interpM(np.log(mmid[risplit:]),np.log(amid))
    newoccrate = newoccrate * (newoccrate > 0) # Occurence rate cannot be negative
    
    # Multiply table-resolution occurrence rates by pixel size. AM I ACTUALLY USING THESE?!
    dlnm_in = (np.log(np.max(mmaxvec)) - np.log(np.min(mminvec))) / len(mvec)
    dlnm_out = np.log(msize) / dlnm_in
    
    dlnr_in = (np.log(np.max(rmaxvec)) - np.log(np.min(rminvec))) / len(rvec)
    dlnr_out = np.log(rsize) / dlnr_in
    
    dlna_in = (np.log(np.max(amaxvec)) - np.log(np.min(aminvec))) / len(avec)
    dlna_out = np.log(asize) / dlna_in

    newoccrate[:,0:risplit] *= np.outer(np.log(asize), np.log(rsize))[:,0:risplit]
    newoccrate[:,risplit:] *= np.outer(np.log(asize), np.log(msize))[:,risplit:]
    
    if np.max(newoccrate) > 1:
        print('Error: max occurrence rate > 1. Try increasing the grid size.')
        exit()
        # Easier to see the error message without a return.
        
    return newoccrate, medge, redge, aedge, mmid, rmid, amid


def add_planets(stars, plorb, expected, orM_array, ora_array, hillsphere_flag, randrad, rng):
    plorb = plorb
    nstars = len(stars)
    hillsphere_flag = np.full(len(hillsphere_flag),False) # reset this flag to no recalculation for all stars
    
    # First make a 2D histogram of a and M
    temp_a = []
    temp_M = []
    for i in range(0,nstars):
        temp_alist = plorb[i,:,pllabel.index('a')] / np.sqrt(stars['Lstar'].loc[i]) # undo the sqrt(L) factor
        temp_a.append(temp_alist)
        temp_M.append(plorb[i,:,pllabel.index('M')])
        
    temp_a = np.array(temp_a).flatten()
    temp_M = np.array(temp_M).flatten()

    realbinsx = np.log(ora_array)
    realbinsy = np.log(orM_array)

    j = np.where(temp_M > 0)[0]
    h, xedges, yedges = np.histogram2d(np.log(temp_a[j]),np.log(temp_M[j]),[realbinsx,realbinsy])
    
    # Now create the planets that we need
    nplanets_histo = expected - h  # 2D histogram of number of needed planets

    ntempsum = 0
    
    # Create a 1D structure to build the needed planets    
    nplanets = int(np.sum(nplanets_histo))
    nplanets = 0
    for i in range(0,len(ora_array)-1):
        for j in range(0,len(orM_array)-1):
            if nplanets_histo[i,j] > 0:
                ntemp = int(nplanets_histo[i,j])
                nplanets += ntemp
    nplanets = int(nplanets)
    temp_planets = np.zeros((nplanets,8))
    temp_planets[:,pllabel.index('a')] = 1.e20
    
    temp_planets[:,pllabel.index('longnode')] = rng.random(nplanets)*360.
    temp_planets[:,pllabel.index('argperi')] = rng.random(nplanets)*360.
    temp_planets[:,pllabel.index('meananom')] = rng.random(nplanets)*360.
    temp_planets[:,pllabel.index('e')]= settings.emin + rng.random(nplanets)*(settings.emax-settings.emin)
    temp_planets[:,pllabel.index('i')] = settings.imin + rng.random(nplanets)*(settings.imax-settings.imin)

    ntest = 0

    lora_array = np.log(ora_array)
    lorM_array = np.log(orM_array)
    
    # Generate random orbital parameters and sizes (some may be unstable)
    iref = 0
    for i in range(0,len(orM_array)-1):
        for j in range(0,len(ora_array)-1):
            if nplanets_histo[j,i] > 0:
                ntemp = int(nplanets_histo[j,i])
                for k in range(0,ntemp):
                    tempM = np.exp(rng.random()*(lorM_array[i+1]-lorM_array[i]) + lorM_array[i])
                    temp_planets[iref+k,pllabel.index('M')] = tempM
                iref += ntemp
    
    iref = 0
    for i in range(0,len(orM_array)-1):
        for j in range(0,len(ora_array)-1):
            if nplanets_histo[j,i] > 0:
                ntemp = int(nplanets_histo[j,i])
                for k in range(0,ntemp):
                    tempa = np.exp(rng.random()*(lora_array[j+1]-lora_array[j]) + lora_array[j])
                    temp_planets[iref+k,pllabel.index('a')] = tempa
                iref += ntemp
    
    # Calculate radius in Earth radii based on Chen & Kipping (2017)
    temp_planets[:,pllabel.index('R')] = mass_to_radius(temp_planets[:,pllabel.index('M')], rng, temp_planets[:,pllabel.index('a')], randrad)
    
    # Randomly assign to stars and sort by semi-major axis
    starindices = np.vectorize(int)(rng.random(nplanets) * nstars)
    temp_planets[:,pllabel.index('a')] *= np.sqrt(np.array(stars['Lstar'].loc[starindices].values).astype(float)) # scale by sqrt(Lstar)
    
    isum = 0

    for i in range(0,nstars):
        ilist = np.where(starindices==i)[0]
        isum += len(ilist)
        if len(ilist)==0:
            pass
        else:
            pold = [plorb[i,j] for j in range(0,maxnplanets) if plorb[i,j,pllabel.index('R')]>0]
            if len(pold)==0: pold = [plorb[i,0]]
            temp_planets2 = np.concatenate((np.array(temp_planets[ilist]),np.array(pold)))
            
            alist = temp_planets2[:,pllabel.index('a')]
            alist = np.argsort(alist)
            temp_planets2 = temp_planets2[alist]
            
            if len(alist)<=maxnplanets:
                plorb[i,0:len(alist)] = temp_planets2
            else:
                plorb[i] = temp_planets2[0:maxnplanets]
            hillsphere_flag[i] = True # any star w/ a planet added needs recalculation of hill spheres
            
    #print('{0:d} planets created.'.format(isum))
    
    return plorb, hillsphere_flag


def remove_unstable_planets(stars, plorb, hillsphere_flag):

    nstars = len(stars)
    mhs_factor = 6 # Dulz et al use 9, but that takes a really
                   # long time to cram them in. So we speed things
                   # up by relaxing this.

    # Delete Hill-unstable planets for each star
    newplanets = []                   
    for i in range(0,nstars):
        if hillsphere_flag[i]:
            threestarmass_EM = 3*stars['mass'].loc[i] * 333000. # star mass in Earth masses
            # They have been sorted by semi-major axis, so now look at them pairwise
            j = 0
            ii = 0
            while plorb[i,j+1,pllabel.index('R')] > 0:
                aout = plorb[i,j+1,pllabel.index('a')]
                ain  = plorb[i,j,pllabel.index('a')]
                Mout = plorb[i,j+1,pllabel.index('M')] # Earth masses
                Min  = plorb[i,j,pllabel.index('M')]   # Earth masses
                mutual_hill_sphere = 2 * (aout-ain)/(aout+ain) * (threestarmass_EM / (Mout + Min))**(1./3.)

                if mutual_hill_sphere < mhs_factor:  # erase the less massive planet
                    if Mout > Min: j0 = j
                    else: j0 = j+1
                    plorb[i,j0] = np.zeros(8)
                    plorb[i,j0,pllabel.index('a')] = 1.e20
                    
                    l = np.argsort(plorb[i,:,pllabel.index('a')])       # re-sort to get rid of deleted planet
                    plorb[i] = plorb[i,l]
                    if j>0: j-=1 # there's now one fewer planet; re-check against the next planet in if there is one
                    
                else: j+=1  # if the planet pair was okay, increase j
                ii+=1
                if j==maxnplanets-1: break # Don't go past the end of the planets list
    return plorb


def assign_albedo_file(stars, plorb, rng):
    '''
    Here we assign random albedo files to the grid of planets
    Note that this is currently designed to only do this in 
    stellar-flux space, i.e. models are selected based on
    stellar radiation received from planet, not absolute 
    semi-major axis
    '''    
    nstars = len(stars)    
    plalbedo = []
    
    nstars = len(stars)
    directory = './geometric_albedo_files/'
    albedos = pd.read_csv('albedo_list.csv')
    
    files = albedos['Files'].values
    probs = albedos['prob'].values
    rmin = albedos['rmin'].values
    rmax = albedos['rmax'].values
    amin = albedos['amin'].values
    amax = albedos['amax'].values
    nfiles = len(files)
    
    for i in range(0,nstars):
        plalbedo.append([])
        nplanets = len(np.where(plorb[i,:,pllabel.index('R')]>0.)[0])
        for j in range(0,nplanets):
            a_pl = plorb[i,j,pllabel.index('a')]/np.sqrt(stars['Lstar'].loc[i])
            R_pl = plorb[i,j,pllabel.index('R')]
            ilist = []
            for k in range(0,len(files)):
                if albedos['EEC'].loc[k]:
                    if rmin[k]/np.sqrt(a_pl) < R_pl and R_pl < rmax[k] and amin[k] < a_pl and a_pl < amax[k]:
                        ilist.append(k)
                else:
                    if rmin[k] < R_pl and R_pl < rmax[k] and amin[k] < a_pl and a_pl < amax[k]:
                        ilist.append(k)
                        
            je = np.where(albedos['EEC'].loc[ilist].values)[0]
            jn = np.where(~albedos['EEC'].loc[ilist].values)[0]
                        
            if len(ilist)==0:
                print('Error: No albedo file specified for R={0:f} and a={0:f}.'.format(R_pl,a_pl))

            flist = files[ilist]
            if len(je)==0:
                plist = np.array(probs[ilist])
            else:
                plist = np.concatenate(( np.array(probs[je])*settings.eecprob, np.array(probs[jn])*(1.-settings.eecprob) ))
            plist = plist/np.sum(plist)
            plist = np.cumsum(plist)    
            
            choose = rng.random()
            find = np.where(plist>=choose)[0]            
            findex = int(find[0])
        
            plalbedo[i].append(flist[findex])
    return plalbedo
