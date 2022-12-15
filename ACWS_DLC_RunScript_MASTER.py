#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:54:31 2020

@author: smith
"""
#%% Imports:
import os
os.chdir('/d1/software/DeepLabCut/')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions, visualization, frameselectiontools
print("Imported DLC version " + str(dlc.__version__))
import custom.helperFunctionsDLC as hf
import pathlib
import pandas as pd
import glob
%matplotlib inline
os.chdir('/d1/studies/DLC_Data/')

#%%Make list of videos for training or analysis
videoDirectory = '/d1/studies/DLC_Data/MeganRotation-smith-2022-12-13/videos' #to the path where your videos are saved
videoFormat = '.mp4' #the format of the videos in videoDirectory
vids = hf.loadDir(videoDirectory, videoFormat, shuffle=True)

#%%Set up experiment:
new = False #set to True if you are creating a new project, or 'vids' to just add new videos to current project.
if not new:
    path_config_file='/d1/studies/DLC_Data/MeganRotation-smith-2022-12-13/config.yaml' #if new, modify this after you've created the new project

    cfg=auxiliaryfunctions.read_config(path_config_file) #this just makes it easy to check any parameters in config. e.g. inputting cfg['pcutoff'] will return the current pcutoff
    projectPath = cfg['project_path']

if new:
    task = 'MeganRotation'
    experimenter='smith'
    if new !='vids':
        dlc.create_new_project(task, experimenter, vids, videotype=videoFormat, copy_videos=True)
    elif new =='vids':
        dlc.add_new_videos(path_config_file, vids, copy_videos=True)
#%%EXTRACT & LABEL FRAMES:
dlc.extract_frames(path_config_file, mode='manual') #mode can be 'automatic' or 'manual'
dlc.label_frames(path_config_file)
dlc.check_labels(path_config_file, draw_skeleton=True)
dlc.dropimagesduetolackofannotation(path_config_file)

#%%CREATE TRAINING DATASET:
"""a new feature is that you can now easily use a parameters from previous iteration or model's pose_cfg.yaml file 
when generating a new training dataset. This is helpful if you commonly use a non-default learning rate schedule, 
batch size, or image augmentation pipeline (or any other parameters). You can set the path to a previous training
iteration pose_cfg.yaml file in the poseCfgTemplate variable below. If you want to use the default pose_cfg.yaml
just set posecfg_template=None (or False) in the create_training_dataset function, or just remove pose_cfg_template=
entirely and it will use the default.
"""
poseCfgTemplate='/d1/studies/DLC_Data/MeganRotation-smith-2022-12-13/dlc-models/iteration-1/MeganRotationDec13-trainset95shuffle1/train/pose_cfg.yaml'
dlc.create_training_dataset(path_config_file, num_shuffles=2, net_type='resnet_50', augmenter_type='imgaug', posecfg_template=None)

#%%TRAIN NETWORK
dlc.train_network(path_config_file, shuffle=1, max_snapshots_to_keep=None, allow_growth=True, keepdeconvweights=True)
#%% EVALUATE & PLOT LOSS:
dlc.evaluate_network(path_config_file, plotting=False, term_gpu=False)
hf.plotLoss(path_config_file)

#%%ANALYZE VIDEOS:
videoDirectory = '/d1/studies/DLC_Data/MouseIVSA-smith-2021-06-11/testCUDA_2' #change to the path where your videos are saved
vids = glob.glob(os.path.join(videoDirectory, '*'+videoFormat))

import time
print(time.ctime())
dlc.analyze_videos(path_config_file, vids, videotype=videoFormat, term_gpu=True, save_as_csv=True, batchsize=64) #dynamic=(True, 0.3, 10),
print(time.ctime())

#%%ANALYZE A SINGLE VIDEOS:
dlc.analyze_videos(path_config_file, ['/d1/studies/DLC_data/Soham/NPW-Soham-2020-08-17_singleanimal-Soham-2020-09-01/videos/C3-LT_Phase3.mp4'], batchsize=64, save_as_csv=True, videotype=videoFormat)

#%%CREATE LABELED VIDEOS
dlc.create_labeled_video(path_config_file, vids,  videotype=videoFormat, displayedbodyparts='all', outputframerate=30)

#%% Do you want to refine labels from the just-analized videos? If so, set refine to True.
refine=False

#%%OPTIONAL, REFINE NETWORK:
bodyParts=cfg['bodyparts']
# if you have any bodyParts that you do not want to refine the labels of, include them in this list:
bodyPartToExclude='cueLight' # if you want to exclude any key points from refinement
bodyParts.remove(bodyPartsToExclude) #If you are tracking stationary objects that are not always present, you may want to exclude them from outlier frame deection
dlc.extract_outlier_frames(path_config_file, vids, comparisonbodyparts=bodyParts, videotype=videoFormat, outlieralgorithm='jump', epsilon=175)

#%%OPTIONALLY REFINE LABELS AND CREATE NEW TRAINING DATASET. ADDS +1 to ITERATON IN CONFIG.YAML
dlc.refine_labels(path_config_file)
dlc.merge_datasets(path_config_file)

#see note above about poseCfg_template
poseCfgTemplate='/d1/studies/DLC_Data/MeganRotation-smith-2022-12-13/dlc-models/iteration-1/MeganRotationDec13-trainset95shuffle1/train/pose_cfg.yaml'
dlc.create_training_dataset(path_config_file, augmenter_type='imgaug')
#%% Retrain the network. You can go back to line 65 and restart the training and refinement process.
dlc.train_network(path_config_file, max_snapshots_to_keep=None, allow_growth=True, keepdeconvweights=True)
#AFTER RE-TRAINING NETWORK, GO BACK TO LINE 51 (EVALUATE NETWORK) AND CONTINUE TO REFINE IF NECESSARY

#%%CREATE LABELED VIDEOS:
destfolder = videoDirectory
#FOR ALL VIDEOS IN A LIST (VIDS):
#vids = glob.glob(os.path.join(videoDirectory, '*HD1*'+videoFormat))
dlc.create_labeled_video(path_config_file, vids,  videotype=videoFormat, displayedbodyparts='all', outputframerate=30)
#VIDEO WITH ONLY KEY POINTS LABELED (I.E. JOHANNSON STYLE)
dlc.create_labeled_video(path_config_file, vids,  videotype=videoFormat, displayedbodyparts='all', keypoints_only=True, outputframerate=30)

#%%FOR ONE VIDEO:
dlc.create_labeled_video(path_config_file, ['/d1/studies/DLC_data/OperantWithCues-smith-2020-10-22/analyzedVideos/h5s/StopMotion/C16_NP_2020-10-25.mp4'],  videotype=videoFormat, displayedbodyparts='all', outputframerate=30, keypoints_only=True, trainingsetindex=0, destfolder=destfolder)

###PLOT TRAJECTORIES:
bodyParts=cfg['bodyparts']
destfolder=videoDirectory
dlc.plot_trajectories(path_config_file, vids, videotype=videoFormat, displayedbodyparts=bodyParts, filtered=False, destfolder=destfolder, track_method='', imagetype='.png', linewidth=0.5, resolution=300)
#Reorganize plots into folders with each plot type:
parentDirectory = '/d1/studies/DLC_Data/MeganRotation-smith-2022-12-13/videos/plot-poses'
hf.extractPoses(parentDirectory)


###OPTIONALLY FILTER TRAJECTORIES AND RE-PLOT:
dlc.filterpredictions(path_config_file, vids, filtertype='median', windowlength=5, save_as_csv=True, destfolder=destfolder)
dlc.plot_trajectories(path_config_file, vids, videotype=videoFormat, displayedbodyparts='all', destfolder=destfolder, filtered=True)


###CUSTOM DOWNSTREAM ANALYSIS (FOR CPP HERE):
directory = ('/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/analyzedVideos/csvs/')
csvPath = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/videos/VG1_LB_2020-12-21DLC_resnet50_VGluT2_RTAJan3shuffle1_300000.h5'
modelPrefix = 'DLC_resnet50_VGluT2_RTAJan3shuffle1_200000'
bodyPart = 'spine2'
axis='x'
flipped=True
index=True

leftTime, rightTime, leftIndex, rightIndex = hf.calcZoneTimes(csvPath, modelPrefix, bodyPart, axis='x', flippedX=flipped, index=True)
hf.extractZones(directory, modelPrefix, bodyPart, axis='x', flipped=True, save=True)

vidPath = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/VG1_RT_2021-01-06.mp4'
saveDir = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/'
hf.extractFrames(vidPath, saveDir)


csvPath = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/VG1_RT_2021-01-06DLC_resnet50_VGluT2_RTAJan3shuffle1_300000.csv'
frameDir = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/VG1_RT_2021-01-06'
fps=30
size=(800,400)
hf.makeZoneVideo(csvPath, modelPrefix, bodyPart, axis, frameDir, fps, size, flippedX=True)

directory = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/analyzedVideos/csvs/'
files = os.listdir(directory)
for f in files:
    if f.endswith('200000.csv'):
        n, e = os.path.splitext(f)
        n = n.split('DLC')[0]
        csvPath = os.path.join(directory, f)
        hf.countBouts(csvPath, modelPrefix, bodyPart, axis='x', saveDir=saveDir, flippedX=True)

mice = []
leftTimes = []
rightTimes = []

directory = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/analyzedVideos/csvs/'
files = os.listdir(directory)
for f in files:
    if f.endswith('200000.csv'):
        n, e = os.path.splitext(f)
        n = n.split('DLC')[0]
        fullpath = os.path.join(directory, f)
        leftTime, rightTime = hf.calcZoneTimes(fullpath, modelPrefix, 'spine2', 'x', flippedX=True, index=False)
        mice.append(n)
        leftTimes.append(leftTime)
        rightTimes.append(rightTime)
df = pd.DataFrame([mice, leftTimes, rightTimes]).T
df.columns=['Mouse', 'Left Time', 'Right Time']
df.to_csv(os.path.join(directory, 'Combined_Zone_Data.csv'))

#%% QUALITY CONTROL:
    
hf.checkLabelSD(path_config_file, drop=None)


