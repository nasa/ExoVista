import numpy as np
import pandas as pd
import sys
from os import path
from src import read_solarsystem
from src import generate_starspots
from src import generate_scene
from src import Settings

# ExoVista v2.4

# Generates a single, user-defined planetary system.

filename = ''
filein = ''
ndisk = 2

if len(sys.argv)>1:
    filein = sys.argv[1]
    if not path.exists(filein):
        print('Error: input file not found.')
        filein = ''
        
while not path.exists(filename):
    if filein != '':
        filename = filein
    else:
        print('Enter input file name.')
        filename = input()
        
    if not path.exists(filename):
        print('Error: file not found.')
        filename = ''
        filein = ''
        continue

    line = ''
    
    fin = open(filename,'r')
    line = fin.readline()
    
    if line != 'Star\n':
        print('Error: wrong file format.')
        filename = ''
        filein = ''
        continue
    while line != 'Planets\n':
        try:
            line = fin.readline()
        except:
            print('Error: file contains no planets.')
            filename = ''
            filein = ''
            break
    while line != 'Disks\n':
        try:
            line = fin.readline()
        except:
            print('Error: file contains no disk components.')
            filename = ''
            filein = ''
            break
    ndisk = -1
    while line != 'Settings\n':
        try:
            line = fin.readline()
            if line != 'Settings\n': ndisk += 1
        except: break
        if len(line.split())==0:
            ndisk -= 1
            break

# "standard" settings configuration, uses a hard-coded random seed for comparability
settings = Settings.Settings(output_dir='./output', ncomponents=ndisk, timemax=10.0, dt=10/365.25, starspots=True, diskoff=False, seed=0)

s,p,a,d,c,new_settings = read_solarsystem.read_solarsystem(settings,system_file=filename)
sp = generate_starspots.generate_spots(s,new_settings)
print('Generating scene...')
generate_scene.generate_scene(s,sp,p,d,a,c,new_settings)
