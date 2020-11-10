#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:36:09 2020

@author: smith
"""

import os
import ACWS_DLC_helperFunctions as hf

#Aggregate plot-poses by plot type
parentDirectory = '/d1/studies/Raspicam/MOR-Flox_Cohort3_PhysicalWithdrawal/data/plot-poses/'
hf.extractPoses(parentDirectory)

#Interpolate low-likelihood values
file = '/d1/studies/DLC_data/Acute_Oxy_Videos/5mgkg_092820/cropped/C9_LT_5mgkg_092820DLC_resnet50_NPWSep26shuffle1_230000_filtered_annotated.csv'
df = hf.interpolateValues(file, 0.4)

directory = '/d1/studies/DLC_data/Acute_Oxy_Videos/5mgkg_092820/cropped/'
csvs=[]
files = os.listdir(directory)
for f in files:
    if f.endswith('.csv'):
        fullpath = os.path.join(directory, f)
        csvs.append(fullpath)
        
for file in csvs:
    hf.interpolateValues(file)


#Trim CSVs
directory = '/d1/studies/DLC_data/OperantDLC/analyzedVids/csvs/train/'
hf.trimCSVs(27000, directory)



