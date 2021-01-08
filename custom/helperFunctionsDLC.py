#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:29:26 2020

@author: smith
"""

import numpy as np
import pandas as pd
import os
import shutil
import cv2

def extractPoses(parentDirectory):
    """Extract and re-organizes files from plot_poses (result of dlc.plot_trajectories) into folders for each plot type.
    

    Parameters
    ----------
    parentDirectory : string
        Path to plot_poses directory that is result of deeplabcut.plot_trajectories.

    Returns
    -------
    None. Creates new folders for each plot type (e.g. trajectory, plot, hist) containing data from all animals.

    """
    
    paths = os.listdir(parentDirectory)
    folders=[]
    for path in paths:
        folder = os.path.join(parentDirectory, path)
        folders.append(folder)
        
    for fold in folders:
        if fold.split('/')[-1].startswith('VG'):
            files = os.listdir(fold)
            sample = fold.split('/')[-1].split('.')[0]
            basepath=fold.split('/')[:-1]
            basepath = '/'.join(basepath)
            if basepath.endswith('_filtered'):
                dtypes = ['trajectory_filtered', 'plot_filtered', 'hist_filtered', 'plot-likelihood_filtered']
            else:       
                dtypes = ['trajectory', 'plot', 'hist', 'plot-likelihood']
            for d in dtypes:
                if not os.path.exists(os.path.join(basepath, d)):
                    os.mkdir(os.path.join(basepath, d))
            for file in files:
                fullpath = os.path.join(basepath, sample + '/' + file)
                f, e = os.path.splitext(fullpath.split('/')[-1])
                targpath = os.path.join(basepath, f + '/' + sample + '_' + f + e)
                shutil.copyfile(fullpath, targpath)


def calcZoneTimes(csvPath, modelPrefix, bodyPart, axis='x', flippedX=True, index=False):
    """Calculate time spent in each half of arena, split along a given axis.
    

    Parameters
    ----------
    csvPath : string
        Path to data file. Can be either .csv or .h5.
    modelPrefix : string
        Name of DLC model that is level 0 of multiindex.
    bodyPart : string
        Labeled body part to use for calculations. Must be in level 1 of multiindex.
    axis : string, optional
        Axis to split into zones ('x' or 'y'). The default is 'x'.
    flippedX : bool, optional
        If the video was collected with horizontal mirroring, set True. The default is True.
    index : bool, optional
        Whether to return frame indices for left/right zones, or only time spent in each. The default is False.

    Returns
    -------
    leftTime : int
        # of seconds spent in left zone.
    rightTime : int
        # of seconds spent in right zone.
    leftIndex: list (returned if index=True)
        List of frame indices in left zone.
    rightIndex: list (returned if index=True)
        List of frame indices in right zone.

    """
    if csvPath.endswith('.csv'):
        df = pd.read_csv(csvPath, header=[0,1,2], index_col=0)
    elif csvPath.endswith('.h5'):
        df = pd.read_hdf(csvPath)
    else:
        raise NameError("Must be either .csv or .h5 file")
    sx = df[(modelPrefix, bodyPart, axis)]
    middle = (sx.max()+sx.min())/2
    if flippedX:
        left = df.loc[df[modelPrefix, bodyPart, axis]>middle]
        leftTime = left.shape[0]/30
        right = df.loc[df[modelPrefix, bodyPart, axis]<middle]
        rightTime = right.shape[0]/30
    elif not flippedX:
        left = df.loc[df[modelPrefix, bodyPart, axis]<middle]
        leftTime = left.shape[0]/30
        right = df.loc[df[modelPrefix, bodyPart, axis]>middle]
        rightTime = right.shape[0]/30
        
    if index:
        leftIndex = list(left.index)
        rightIndex = list(right.index)
        return(leftTime, rightTime, leftIndex, rightIndex)
    elif not index:
        return leftTime, rightTime

def extractZones(directory, modelPrefix, bodyPart, axis='x', flipped=False, save=True):
    """Create new data files for each zone created in calcZoneTimes.
    

    Parameters
    ----------
    directory : string
        Path to directory containing data.
    modelPrefix : string
        Name of DLC model that is level 0 of multiindex.
    bodyPart : string
        Labeled body part to use for calculations. Must be in level 1 of multiindex.
    axis : string, optional
        Axis to split into zones ('x' or 'y'). The default is 'x'.
    flipped : bool, optional
        If the video was collected with mirroring along specified axis, set True. The default is True.
    save : bool, optional
        Whether to save or only return result files. The default is True.

    Returns
    -------
    leftTime : int
        # of seconds spent in left zone.
    rightTime : int
        # of seconds spent in right zone.
    leftIndex: list (returned if index=True)
        List of frame indices in left zone.
    rightIndex: list (returned if index=True)
        List of frame indices in right zone.

    """
    
    files = os.listdir(directory)
    if save:
        if not os.path.exists(os.path.join(directory, 'SplitZoneData/')):
            os.mkdir(os.path.join(directory, 'SplitZoneData/'))
    for f in files:
        if f.endswith('.csv') or f.endswith('.h5'):
            n, e = os.path.splitext(f)
            fullpath = os.path.join(directory, f)
            if f.endswith('.csv'):
                df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
            elif f.endswith('.h5'):
                df = pd.read_hdf(fullpath)
            leftTime, rightTme, leftIndex, rightIndex = calcZoneTimes(fullpath, modelPrefix, bodyPart, axis, flippedX=flipped, index=True)
            leftData = df.loc[df.index.isin(leftIndex)]
            rightData = df.loc[df.index.isin(rightIndex)]
            if save:
                leftData.to_csv(os.path.join(directory, 'SplitZoneData/' + n + '_LeftZone.csv'))
                rightData.to_csv(os.path.join(directory, 'SplitZoneData/' + n + '_RightZone.csv'))
    return leftData, rightData


def extractFrames(vidPath, saveDir):
    """Extract all frames from video as .jpg files.
    

    Parameters
    ----------
    vidPath : string
        Path to video to extract frames from.
    saveDir : string
        Path to folder to save frames in.

    Returns
    -------
    None.

    """
    sampleName, ext = os.path.splitext(vidPath.split('/')[-1])
    
    if not os.path.exists(os.path.join(saveDir, sampleName + '/')):
        os.mkdir(os.path.join(saveDir, sampleName))
        
    vc = cv2.VideoCapture(vidPath)
    c=0
    
    while(True):
        rval, frame = vc.read()
        
        if rval:
            cv2.imwrite(os.path.join(saveDir, sampleName + '/' + str(c) + '.jpg'), frame)
            c+=1
            cv2.waitKey(1)
        else:
            break
    vc.release()

def makeZoneVideo(csvPath, modelPrefix, bodyPart, axis, frameDir, fps, size, flippedX=True):
    """Create new videos containing only frames in each zone.
    

    Parameters
    ----------
    csvPath : string
        Path to data file. Can be either .csv or .h5.
    modelPrefix : string
        Name of DLC model that is level 0 of multiindex.
    bodyPart : string
        Labeled body part to use for calculations. Must be in level 1 of multiindex.
    axis : string, optional
        Axis to split into zones ('x' or 'y'). The default is 'x'.
    frameDir: string
        Path to saved frames.
    fps: int
        FPS for result video.
    size: tuple
        (width, height) size of video frames (must be same dimensions as original)
    flippedX : bool, optional
        If the video was collected with horizontal mirroring, set True. The default is True.

    Returns
    -------
    None.
    """
    
    sampleName = frameDir.split('/')[-1]
    if sampleName == '':
        sampleName = frameDir.split('/')[-2]
    leftDir = os.path.join(frameDir, 'Left/')
    rightDir = os.path.join(frameDir, 'Right/')
    if not os.path.exists(leftDir):
        os.mkdir(leftDir)
    if not os.path.exists(rightDir):
        os.mkdir(rightDir)

    left, right, leftIndex, rightIndex = calcZoneTimes(csvPath, modelPrefix, bodyPart, axis, flippedX=flippedX, index=True)
    paths = os.listdir(frameDir)
    for path in paths:
        if path.endswith('.jpg'):
            fullpath = os.path.join(frameDir, path)
            frame = int(path.strip('.jpg'))
            if frame in leftIndex:
                if not os.path.exists(os.path.join(leftDir, path)):
                    shutil.move(fullpath, leftDir)
            elif frame in rightIndex:
                if not os.path.exists(os.path.join(rightDir, path)):
                    shutil.move(fullpath, rightDir)
            else:
                pass
    left_img_array = []
    right_img_array = []
    if not os.path.exists(os.path.join(frameDir, sampleName + '_LeftZone.mp4')):
        for frame in leftIndex:
            try:
                img = cv2.imread(os.path.join(leftDir, str(frame) + '.jpg'))
                height, width, layers = img.shape
                size=(width, height)
                left_img_array.append(img)
            except:
                print("Error in leftIndex frame " + str(frame))
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(frameDir, sampleName + '_LeftZone.mp4'), fourcc, fps, size)
        for i in range(len(left_img_array)):
            print("writing frame " + str(i))
            out.write(left_img_array[i])
        out.release()

    if not os.path.exists(os.path.join(frameDir, sampleName + '_RightZone.mp4')):   
        for frame in rightIndex:
            try:
                img = cv2.imread(os.path.join(rightDir, str(frame) + '.jpg'))
                height, width, layers = img.shape
                size=(width, height)
                right_img_array.append(img)
            except:
                print("Error in rightIndex frame " + str(frame))
                
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(frameDir, sampleName + '_RightZone.mp4'), fourcc, fps, size)
        for i in range(len(right_img_array)):
            out.write(right_img_array[i])
        out.release()

def consecutive(data, stepsize=1):
    data = data[:]
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def countBouts(csvPath, modelPrefix, bodyPart, axis='x', saveDir=None, flippedX=True):
    """Calculate time spent in each half of arena, split along a given axis.
    

    Parameters
    ----------
    csvPath : string
        Path to data file. Can be either .csv or .h5.
    modelPrefix : string
        Name of DLC model that is level 0 of multiindex.
    bodyPart : string
        Labeled body part to use for calculations. Must be in level 1 of multiindex.
    axis : string, optional
        Axis to split into zones ('x' or 'y'). The default is 'x'.
    saveDir: string, optional
        Path to directory to save results. If None do not save, only return.
    flippedX : bool, optional
        If the video was collected with horizontal mirroring, set True. The default is True.

    Returns
    -------
    leftTime : int
        # of seconds spent in left zone.
    rightTime : int
        # of seconds spent in right zone.
    leftIndex: list (returned if index=True)
        List of frame indices in left zone.
    rightIndex: list (returned if index=True)
        List of frame indices in right zone.

    """

    sampleName = csvPath.split('/')[-1].split('DLC')[0]
    left, right, leftIndex, rightIndex = calcZoneTimes(csvPath, modelPrefix, bodyPart, axis, flippedX=flippedX, index=True)
    leftCons = consecutive(leftIndex, stepsize=1)
    rightCons = consecutive(rightIndex, stepsize=1)
    df = pd.DataFrame()
    l = []
    lstarts = []
    r = []
    rstarts = []
    for i in leftCons:
        l.append(len(i))
        lstarts.append(i[0])
    for i in rightCons:
        r.append(len(i))
        rstarts.append(i[0])
    df = pd.DataFrame([lstarts, l, rstarts, r]).T
    df.columns=['Left Bout Start Frame', 'Left Bout Length', 'Right Bout Start Frame', 'Right Bout Length']
    if saveDir:
        df.to_csv(os.path.join(saveDir, sampleName + '_ZoneBouts.csv'))
    return df    


