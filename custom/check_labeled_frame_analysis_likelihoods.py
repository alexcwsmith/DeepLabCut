#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 09:29:35 2021

@author: smith
"""

import os
os.chdir('/d1/studies/DeepLabCut_Forked/DeepLabCut/')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import deeplabcut as dlc
print("Imported DLC version " + str(dlc.__version__))
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions, visualization, frameselectiontools


config='/d1/studies/DLC_data/Matt_Dlight/Matt_Dlight_Photometry-smith-2021-04-20/config.yaml'

videoDirectory = '/d1/studies/DLC_data/Matt_Dlight/Matt_Dlight_Photometry-smith-2021-04-20/videos/'
videoFormat = '.mp4'
task = 'Matt_Dlight_Photometry'
experimenter='smith'
videos=glob.glob(videoDirectory+'*'+videoFormat)
csvs=glob.glob(videoDirectory+'*'+'.csv')
shuffle=1
trainingsetindex=0
comparisonbodyparts='all'
modelprefix=''
track_method=''
verbose=True
save=True
return_all=False
videotype=videoFormat
p_bound=None
destfolder=None

def check_labeled_frames_analysis_likelihoods(
    config,
    videos,
    videotype=".avi",
    shuffle=1,
    trainingsetindex=0,
    comparisonbodyparts="all",
    p_bound=None,
    destfolder=None,
    modelprefix="",
    track_method="",
    return_all=False,
    verbose=True,
    save=False,
):
    """Check analyzed videos against labeled frames to identify potentially mislabeled frames where the scorer likelihood is low.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle : int, optional
        The shufle index of training dataset. The extracted frames will be stored in the labeled-dataset for
        the corresponding shuffle of training dataset. Default is set to 1

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    comparisonbodyparts: list of strings, optional
        This selects the body parts for which the comparisons with the outliers are carried out. Either ``all``, then all body parts
        from config.yaml are used, or a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    p_bound: float between 0 and 1, optional
        p-cutoff for likelihood to use. If not specified pcutoff from config.yaml is used.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    modelprefix : string, optional
        Prefix for name of model, if any. Default ''.

    track_method : string, optional
        Tracking method for multi-animal projects. 'skeleton' or 'box'.
        
    return_all : bool, optional
        Return analysis likelihoods for all labeled frames, bypassing p_bound.

    verbose : bool, optional
        Print info as frames are detected. Default True.
    
    save : bool, optional
        Save csv file with results into project path. Default False only returns frame indices without saving.
    """

    cfg = auxiliaryfunctions.read_config(config)
    if p_bound == None:
        pcutoff = cfg["pcutoff"]
    else:
        pcutoff = float(p_bound)
    projectPath = cfg["project_path"]
    iteration = cfg['iteration']
    
    bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, comparisonbodyparts
    )
    if not len(bodyparts):
        raise ValueError("No valid bodyparts were selected.")

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    Videos = auxiliaryfunctions.Getlistofvideos(videos, videotype)
    if len(Videos) == 0:
        print("No suitable videos found in", videos)

    cat = pd.DataFrame()

    for video in Videos:
        if destfolder is None:
            videofolder = str(Path(video).parents[0])
        else:
            videofolder = destfolder
        vname = os.path.splitext(os.path.basename(video))[0]

        try:
            df, dataname, _, _ = auxiliaryfunctions.load_analyzed_data(
                videofolder, vname, DLCscorer, track_method=track_method
            )
            labeledFrames = os.listdir(
                os.path.join(projectPath, "labeled-data/", vname)
            )
            labeledFrameIndices = []
            for frame in labeledFrames:
                if not frame.endswith(".png"):
                    labeledFrames.remove(frame)
                elif frame.endswith(".png"):
                    frameNumber = frame.strip(".png").strip("img")
                    labeledFrameIndices.append(frameNumber)
            labeledFrameIndices = sorted(labeledFrameIndices)
            labeledFrameData = df.loc[df.index.isin(labeledFrameIndices)].copy()
            for bp in bodyparts:
                if return_all:
                    labeledFrameData.drop([(DLCscorer, bp, 'x'), (DLCscorer, bp, 'y')], inplace=True, axis=1)
                    lhdata = pd.concat([labeledFrameData], keys=[vname], names=['Video'], axis=1)
                    cat = pd.concat([cat, lhdata], axis=1)
                elif not return_all:
                    lowConfIndices = labeledFrameData.loc[
                        labeledFrameData[(DLCscorer, bp, "likelihood")] < pcutoff
                    ].index.tolist()
                                
                    framesToCheck = []
                    for frame in lowConfIndices:
                        if str(frame) in labeledFrameIndices:
                            framesToCheck.append(int(frame))
                            if verbose:
                                print(
                                    vname
                                    + " labeled Frame "
                                    + str(frame)
                                    + " confidence under "
                                    + str(pcutoff)
                                )
                    dfcheck = pd.DataFrame(framesToCheck)
                    try:
                        dfcheck.columns = [vname+'_'+bp]
                        cat = pd.concat([cat, dfcheck], axis=1)
                    except ValueError:
                        if verbose:
                            print("No low-confidence labeled frames detected for " + str(vname))
                        continue
        except FileNotFoundError as e:
            print(e)
            print(
                "It seems the video has not been analyzed yet, or the video is not found! "
                "You can only use this function after videos are analyzed. Please run 'analyze_video' first. "
                "Or, please double check your video file path."
            )
    if return_all:
        cat.drop(labels=['x','y'], axis=1, level='coords', inplace=True)
        cat = cat.T
        cat.drop_duplicates(inplace=True)
        cat = cat.T
    if save:
        if return_all:
            if not os.path.exists(os.path.join(videofolder, 'LabeledFrameAnalyzedData/')):
                os.mkdir(os.path.join(videofolder, 'LabeledFrameAnalyzedData/'))
            cat.to_csv(os.path.join(videofolder, 'LabeledFrameAnalyzedData/'+'Iteration'+str(iteration)+'_LabeledFrameAnalysisLikelihoods.csv'))
        elif not return_all:
            pcutoff_sn = str(pcutoff).split('.')[-1]
            pcutoff_sn = pcutoff_sn+"e-"+str(len(pcutoff_sn))
            cat.to_csv(os.path.join(videofolder, 'LabeledFrameAnalyzedData/', 'Iteration'+str(iteration)+'_FramesToDoubleCheckLabels_pcutoff'+str(pcutoff_sn)+'.csv'))
    return cat

