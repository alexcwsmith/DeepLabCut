#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:38:48 2020

@author: smith
"""

import deeplabcut as dlc
import pandas as pd
import os
from pickle_protocol import pickle_protocol

df = pd.read_csv('/d1/studies/DLC_data/Combined_NPW/Cohort1_mp4s/egocentric/C3-RB_Phase1DLC_resnet50_NPWSep26shuffle1_800000_egocentric.csv', index_col=0, skiprows=[0,1])

df2 = pd.read_hdf('/d1/studies/DLC_data/Combined_NPW/Cohort1_mp4s/egocentric/C3-RB_Phase1DLC_resnet50_NPWSep26shuffle1_800000.h5')

cols = df2.columns

df.columns=cols

df = df + 100

with pickle_protocol(4):
    df.to_hdf('/d1/studies/DLC_data/Combined_NPW/Cohort1_mp4s/egocentric/C3-RB_Phase1DLC_resnet50_NPWSep26shuffle1_800000_pp4.h5', key='df_with_missing')


videos = os.listdir(videoDirectory)
vids = []
for vid in videos:
    if vid.endswith('.mp4'):
        fullpath = os.path.join(videoDirectory, vid)
        vids.append(fullpath)

dlc.create_labeled_video(path_config_file, vids, videotype='.mp4', displayedbodyparts='all', draw_skeleton=True, keypoints_only=True, outputframerate=30)


