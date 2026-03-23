import numpy as np
import pandas as pd
import time

from exovista import Settings
from exovista.constants import MSH, R_sun

settings = Settings.Settings()

def generate_spots(stars, settings):
    # spot_warmup_time: length of burn-in phase for spot creation, can be assigned for all stars or star-by-star
    # fac_warmup_time: length of burn-in phase for faculae creation, can be assigned for all stars or star-by-star
    
    settings = settings
    rng = np.random.default_rng(settings.seed)
    indices = []
    spot_table = []

    for i in range(0,len(stars)):
        s = stars.iloc[i]
        indices.append(s['ID'])

        s['MeanSpotArea'] = 500*MSH
            
        if s['SpotDistribution']=='big':
            size = s['SpotCoverage']*2.e6*MSH*s['rstar']**2
            spots = pd.DataFrame({'Star':        [s['ID']],
                                  'Latitude':    [-40.],
                                  'Longitude':   [100.],
                                  'MaxArea':     [size],
                                  'InitialArea': [size],
                                  'CurrentArea': [size],
                                  'AreaRatio':   [5.0],        # ratio of spot area/umbra area
                                  'IsGrowing':   [False]})
        
        else:
            spots = pd.DataFrame({'Star':        [],
                                  'Latitude':    [],
                                  'Longitude':   [],
                                  'MaxArea':     [],
                                  'InitialArea': [],
                                  'CurrentArea': [],
                                  'AreaRatio':   [],
                                  'IsGrowing':   []})  # ratio of spot area/umbra area
            spots = warm_up_star(s, spots, rng)
            
        spot_table.append(spots)
        
        faculae = pd.DataFrame({'Star':               [],
                                'Latitude':           [],
                                'Longitude':          [],
                                'MaxRad':             [],
                                'InitialRad':         [],
                                'Depth':              [],
                                'Lifetime':           [],
                                'FloorTeffSlope':     [],
                                'FloorTeffMinRad':    [],
                                'FloorTeffBaseDTeff': [],
                                'WallTeffSlope':      [],
                                'WallTeffIntercept':  [],
                                'IsGrowing':          []})
                                
        '''
        spot_generator = SpotGenerator.from_params(spotparams,nlat,nlon,gridmaker,rng)
        fac_geneator = FaculaGenerator.from_params(facparams,nlat,nlon,gridmaker,rng)
        flare_generator = FlareGenerator.from_params(flareparams,rng)
        granulation = Granulation.from_params(granparams,rng)
        '''
        
    spot_table = pd.DataFrame({'ID':indices,'Spots':spot_table})
    return spot_table

def warm_up_star(star, spots, rng):
    '''
    "Warm up" the star by adding and removing features to reach equilibrium
    as opposed to starting with an arbitrary spot distribution.
    
    star: star DataFrame entry
    spot_warmup_time: time to run to reach spot equilibrium, in DAYS
    fac_warmup_time: time to run to reach facula equilibrium, in HOURS
    (Yes, the defaults are supposed to be zero.)

    initial_coverage: fraction of the star's surface covered by stars, should be assigned star-by-star
    '''

    #r = star['rstar']
    #cover = star['SpotCoverage']
    if star['SpotCoverage'] > 0: spots = generate_mature_spots(star, spots, rng)
    
    spot_warmup_step = 1.0
    fac_warmup_step = 1.0
    N_steps_spot = int(np.ceil(star['SpotWarmupTime']/spot_warmup_step))
    #N_steps_fac = int(np.ceil(star['FacWarmupTime']/fac_warmup/step))

    if N_steps_spot > 0:
        for i in range(0,N_steps_spot):
            spots = birth_spots(spot_warmup_step, star, spots, rng)
        #for i in range(0,N_steps_fac):
        #    birth_faculae(fac_warmup_step)

    #get_flares(star)
    
    return spots

def generate_mature_spots(star, spots, rng):
    if star['SpotCoverage'] > 1 or star['SpotCoverage'] < 0:
        print('Error: spot coverage must be between 0 and 1.')
        return
    
    current_omega = 0.0 # initial coverage in square degrees
    target_omega = 4*np.pi*star['SpotCoverage'] * (180/np.pi)**2 # target coverage in square degrees
    mean_lifetime = star['MeanSpotArea']/star['DecayRate'] - np.log(star['InitialArea']/star['MeanSpotArea']) / star['GrowthRate']
    growth_rate = star['GrowthRate']
    decay_rate = star['DecayRate']

    spot_list = []
    
    while current_omega < target_omega:
        '''
        # Seems to be an option that is never used in VSPEC
        if is_static:         # boolean attribute of the spot generator
            area0 = init_area # input parameter of the spot generator
            area_range = new_spot.area_max - area0
            area = rng.random()*area_range + area0
            new_spot.area_current = area
        else:
        '''
        
        max_areas = rng.lognormal(mean=np.log(star['MeanSpotArea']/MSH),
                                  sigma=star['LogSigmaArea']) * MSH
        new_r_A = rng.normal(loc=5, scale=1)
        while np.any(new_r_A <= 0):
            new_r_A = rng.normal(loc=5, scale=1)
        lat, lon = get_coordinates(1, star['SpotDistribution'], rng)
        init_area = star['InitialArea']  # may want a distribution here
        
        new_spot = {'Star':        star['ID'],
                    'Latitude':    lat[0],
                    'Longitude':   lon[0],
                    'MaxArea':     max_areas,
                    'InitialArea': init_area,
                    'CurrentArea': init_area,
                    'AreaRatio':   new_r_A,  # ratio of spot area/umbra area
                    'IsGrowing':   True}
        
        age = rng.random()*mean_lifetime
        new_spot = age_spot(new_spot, growth_rate, decay_rate, age)
        spot_solid_angle = new_spot['CurrentArea'] / (R_sun*star['rstar'])**2 * (180./np.pi)**2
        current_omega += spot_solid_angle
        
        spot_list.append(new_spot)

    spots = pd.DataFrame(spot_list)
    return spots

def birth_spots(dt, star, spots, rng):
    N_exp = coverage * 4*np.pi*(star['rstar']*R_Sun)**2 / star['MeanSpotArea'] * dt / mean_lifetime
    nspots = rng.poisson(lam=nspots)
    spots = generate_new_spots(nspots, star, spots, rng)
    return spots

def generate_new_spots(nspots, star, rng):
    new_max_areas = rng.lognormal(mean=np.log(star['MeanSpotArea']/MSH),
                                  sigma=star['LogSigmaArea'],
                                  size=nspots) * MSH
    new_r_A = rng.normal(loc=5, scale=1, size=nspots)
    while np.any(new_r_A <= 0):
        new_r_A = rng.normal(loc=5, scale=1, size=nspots)
    lat, lon = get_coordinates(nspots, star['SpotDistribution'], rng)
    init_area = star['InitialArea']  # may want a distribution here

    spot_list = []
    
    for i in range(0,nspots):
        new_spot = {'Star':        star['ID'],
                    'Latitude':    lat[i],
                    'Longitude':   lon[i],
                    'MaxArea':     new_max_areas[i],
                    'InitialArea': init_area,
                    'CurrentArea': init_area,
                    'AreaRatio':   new_r_A[i],  # ratio of spot area/umbra area
                    'IsGrowing':   True}        
        spot_list.append(new_spot)

    spots = pd.DataFrame(spot_list)
    return spots

def age_spot(spot, growth_rate, decay_rate, dt):
    if spot['IsGrowing']:
        tau = np.log(1+growth_rate) # inverse time to grow from A to A(1+g) (exponential growth)
        if tau==0: time_to_max = np.inf
        else: time_to_max = np.log(spot['MaxArea']/spot['CurrentArea']) / tau # time to grow to Max Area
        
        if dt < time_to_max: # if spot does NOT reach Max Area in time dt
            spot['CurrentArea'] *= np.exp(tau * dt)
        else:
            spot['IsGrowing'] = False
            decay_time = dt - time_to_max   # Amount that dt is after Max Area
            area_decay = decay_time * decay_rate # Linear decrease in area
            if area_decay > spot['MaxArea']: spot['CurrentArea'] = 0.
            else: spot['CurrentArea'] = spot['MaxArea'] - area_decay
    else:
        area_decay = dt * decay_rate
        if area_decay > spot['MaxArea']: spot['CurrentArea'] = 0.
        else: spot['CurrentArea'] = spot['CurrentArea'] - area_decay
        
    # again for faculae, etc.
                
    return spot

def get_coordinates(nspots, distribution, rng):
    
    if distribution == 'solar':
        # (dist approx from 2017ApJ...851...70M)
        hemi = rng.choice([-1,1], size=nspots)
        lat = rng.normal(15, 5, size=nspots)*hemi
        lon = rng.random(size=nspots)*360
    elif distribution == 'iso':
        lon = rng.random(size=nspots)*360
        lat = np.arcsin(2*rng.random(size=nspots) - 1) * 180/np.pi
    else: raise ValueError('Unknown spot distribution function.')
    
    return lat, lon


def age_star(star, spots, dt):
    return

def birth_faculae(dt):
    # pending getting spots working
    return

def get_flares(star):
    # pending getting spots working
    return
