#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:54:31 2020

@author: smith
"""

import os
os.chdir('/d1/studies/DeepLabCut_Forked/DeepLabCut/')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import deeplabcut as dlc
print("Imported DLC version " + str(dlc.__version__))
import custom.helperFunctionsDLC as hf
import pathlib
import pandas as pd
os.chdir('/home/howelab/')
%matplotlib inline


###Set up experiment:
new = False #set to True if you are creating a new project
path_config_file='/d1/studies/Richard/CSISF_lean_obese-ROC-2021-03-09/config.yaml' #if new, modify this after you've created the new project

videoDirectory = '/d1/studies/Richard/CSISF_videos_sorted/CSFISF_obese_training/' #change to the path where your videos are saved
videos = os.listdir(videoDirectory)
vids=[]
for vid in videos:
    if vid.endswith('.mp4'): #you may need to change .mp4 to your video extension, here and throughout the script
        fullpath = os.path.join(videoDirectory, vid)
        vids.append(fullpath)   

if new: #change task and experimenter below if new=True
    task = 'CSISF_lean_obese'
    experimenter='ROC'
    dlc.create_new_project(task, experimenter, vids, videotype='.mp4', copy_videos=True) 
  #  dlc.add_new_videos(path_config_file, vids, copy_videos=True)



###EXTRACT FRAMES & LABEL THEM:
dlc.extract_frames(path_config_file, mode='automatic') #Add argument mode='manual' to manually select frames, make sure to grab diverse frames to capture broad range of behaviors, also consider if there are lighting differences within the frame
dlc.label_frames(path_config_file)
dlc.check_labels(path_config_file, draw_skeleton=True)
###CREATE TRAINING DATASET:
dlc.create_training_dataset(path_config_file, augmenter_type='imgaug')

###TRAIN & EVALUATE NETWORK:
dlc.train_network(path_config_file, saveiters=25000, max_snapshots_to_keep=15, allow_growth=True, keepdeconvweights=True)
dlc.evaluate_network(path_config_file, plotting=True)


###ANALYZE VIDEOS:
videoDirectory = '/d1/studies/Richard/CSISF_videos_sorted/CSFISF_obese_crop/'
videos = os.listdir(videoDirectory)
vids = []
for vid in videos:
    if vid.endswith('.mp4'):
        fullpath = os.path.join(videoDirectory, vid)
        vids.append(fullpath)

import time
print(time.localtime())
dlc.analyze_videos(path_config_file, vids, videotype='.mp4', save_as_csv=True, batchsize=64) #dynamic=(True, 0.3, 10),
print(time.localtime())
###ANALYZE A SINGLE VIDEO:
dlc.analyze_videos(path_config_file, ['/d1/studies/DLC_data/Soham/NPW-Soham-2020-08-17_singleanimal-Soham-2020-09-01/videos/C3-LT_Phase3.mp4'], batchsize=96, save_as_csv=True, videotype='.mp4')


###OPTIONAL, REFINE NETWORK:
dlc.extract_outlier_frames(path_config_file, vids, videotype='.mp4', outlieralgorithm='jump', epsilon=400)

###OPTIONALLY REFINE LABELS AND CREATE NEW TRAINING DATASET. ADDS +1 to ITERATON IN CONFIG.YAML
dlc.refine_labels(path_config_file)
dlc.merge_datasets(path_config_file)
dlc.create_training_dataset(path_config_file, augmenter_type='imgaug')
dlc.train_network(path_config_file, saveiters=25000, max_snapshots_to_keep=10, allow_growth=True, keepdeconvweights=True)
#AFTER RE-TRAINING NETWORK, GO BACK TO LINE 48 (EVALUATE NETWORK) AND CONTINUE TO REFINE IF NECESSARY

###CREATE LABELED VIDEOS:
destfolder = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/'
#FOR ALL VIDEOS IN A LIST (VIDS):
dlc.create_labeled_video(path_config_file, vids,  videotype='.mp4', displayedbodyparts='all', outputframerate=30)
#VIDEO WITH ONLY KEY POINTS LABELED (I.E. JOHANNSON STYLE)
dlc.create_labeled_video(path_config_file, vids,  videotype='.mp4', displayedbodyparts='all', keypoints_only=True, outputframerate=30)

###FOR ONE VIDEO:
dlc.create_labeled_video(path_config_file, ['/d1/studies/DLC_data/OperantWithCues-smith-2020-10-22/analyzedVideos/h5s/StopMotion/C16_NP_2020-10-25.mp4'],  videotype='.mp4', displayedbodyparts='all', outputframerate=30, keypoints_only=True, trainingsetindex=0, destfolder=destfolder)

###PLOT TRAJECTORIES:
dlc.plot_trajectories(path_config_file, vids, videotype='.mp4', displayedbodyparts='all', filtered=False, destfolder=destfolder, track_method='', imagetype='.png', linewidth=0.5, resolution=300)


###OPTIONALLY FILTER TRAJECTORIES AND RE-PLOT:
dlc.filterpredictions(path_config_file, vids, filtertype='median', windowlength=9, save_as_csv=True, destfolder=destfolder)
dlc.plot_trajectories(path_config_file, vids, videotype='.mp4', displayedbodyparts='all', destfolder=destfolder, filtered=True)



###CUSTOM DOWNSTREAM ANALYSIS:
parentDirectory = '/d1/studies/DLC_data/VGluT2_RTA-smith-2021-01-03/vidsToAnalyze/plot-poses/'
hf.extractPoses(parentDirectory)

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




