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
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from custom import customAuxiliaryFunctions as caux


def h5toCSV(directory):
    """Convert all .h5 files in directory to .csvs.
    
    Parameters
    ----------
    directory : string
        Directory containing .h5 files.

    Returns
    -------
    None.
    """
    files = os.listdir(directory)
    for f in files:
        if f.endswith('.h5'):
            n, e = os.path.splitext(f)
            fullpath=os.path.join(directory, f)
            df = pd.read_hdf(fullpath)
            df.to_csv(os.path.join(directory, n + '.csv'))

def interpolateValues(file, pcutoff):
    """Interpolate values with low prediction confidence (i.e. low likelihood).
    
    Parameters
    ----------
    file : string
        Path to CSV or h5 file to interpolate.
    
    Returns
    -------
    Interpolated dataframe.
    """
    if file.endswith('.csv'):
        df = pd.read_csv(file, index_col=0, header=[0,1,2])  
    elif file.endswith('.h5'):
        df = pd.read_hdf(file)
    scorerName = df.columns[0][0]
    bodyparts = set(df.columns[x][1] for x in range(len(df.columns)))
    for bp in bodyparts:
        df.loc[df[(scorerName, bp, 'likelihood')] < pcutoff, (scorerName, bp, 'x')]=None
        df.loc[df[(scorerName, bp, 'likelihood')] < pcutoff, (scorerName, bp, 'y')]=None
    filename, ext = os.path.splitext(file)
    df = df.interpolate(method='linear')
    df = df.dropna(how='any', axis=0)
    df.to_csv(filename + '_interpolated.csv')
    df.to_hdf(filename + '_interpolated.h5', key='df_with_missing')
    return(df)

def trimFrames(directory, startFrame=None, stopFrame=None):
    """Crop csv or data files within specific frames. Good for removing unuseful frames from beginning, end, or both.
    
    Parameters
    ----------
    directory : string
        Path to directory containing data files to process.
    startFrame : int (optional, default None)
        Starting frame for trimmed data.
    stopFrame : int (optional, default None)
        Ending frame for trimmed data.
    """
    files = os.listdir(directory)
    saveDir = os.path.join(directory, 'trimmedData/')
    if not os.path.exist(saveDir):
        os.mkdir(saveDir)
    for f in files:
        if f.endswith('.csv'):
            sampleName = f.split('DLC')[0]
            fullpath = os.path.join(directory, f)
            df = pd.read_csv(fullpath, header = [0,1,2], index_col=0)
            df = df[startFrame:stopFrame]
            df.to_csv(os.path.join(saveDir, sampleName + '_trimmed.csv'))
        elif f.endswith('.h5'):
            sampleName = f.split('DLC')[0]
            fullpath = os.path.join(directory, f)
            df = pd.read_hdf(fullpath)
            df = df[startFrame:stopFrame]
            df.to_hdf(os.path.join(saveDir, sampleName + '_trimmed.csv'), key='df_with_missing')


def extractPoses(parentDirectory, prefix=''):
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
        if fold.split('/')[-1].startswith(prefix):
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


def calcZoneTimes(csvPath, modelPrefix, bodyPart, axis='x', fps=30, flippedX=True, index=False):
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
    fps : int/float, optional
        FPS of the video, for time calculations. Default 30.0.
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
        leftTime = left.shape[0]/fps
        right = df.loc[df[modelPrefix, bodyPart, axis]<middle]
        rightTime = right.shape[0]/fps
    elif not flippedX:
        left = df.loc[df[modelPrefix, bodyPart, axis]<middle]
        leftTime = left.shape[0]/fps
        right = df.loc[df[modelPrefix, bodyPart, axis]>middle]
        rightTime = right.shape[0]/fps
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

def calcDistanceMA(h5File, indiv1, indiv2, bodyPart1, bodyPart2, directory=os.getcwd(), distThreshold=5):
    """Calculate the distance between two bodyparts on two individuals in multi-animal DLC.

    Parameters
    ----------
    h5File : string
        Path to h5 file containing tracking data.
    indiv1 : string
        Label of first individual (e.g. mouse1)
    indiv2 : string
        Label of second individual (e.g. mouse2)
    bodyPart1 : string
        Body part of first individual (e.g. nose)
    bodyPart2 : string
        Body part of second individual (e.g. nose)
    directory : string (optional)
        Directory to save to. Default current working directory.
    distThreshold : int/float (optional)
        Pixel distance threshold to be considered interacting. The default is 5.

    Returns
    -------
    Pandas dataframe of interactions.

    """
    df = pd.read_hdf(h5File)
    sampleName = h5File.split('/')[-1].split('DLC')[0]
    scorerName = df.columns[0][0]
    bp1x = df[(scorerName, indiv1, bodyPart1, 'x')]
    bp1x.interpolate(method='linear', inplace=True)
    bp1y = df[(scorerName, indiv1, bodyPart1, 'y')]
    bp1y.interpolate(method='linear', inplace=True)
    bp2x = df[(scorerName, indiv2, bodyPart2, 'x')]
    bp2x.interpolate(method='linear', inplace=True)
    bp2y = df[(scorerName, indiv2, bodyPart2, 'y')]
    bp2y.interpolate(method='linear', inplace=True)
    bp1coords = list(zip(bp1x.tolist(), bp1y.tolist()))
    bp2coords = list(zip(bp2x.tolist(), bp2y.tolist()))
    dists = []
    for i in range(len(bp1coords)):
        d = np.linalg.norm(np.array(bp1coords[i])-np.array(bp2coords[i]))
        dists.append(d)
    distDf = pd.DataFrame([bp1coords,bp2coords,dists]).T
    distDf.columns=[indiv1+'_'+bodyPart1, indiv2+'_'+bodyPart2, 'Distance']
    distDf.to_csv(os.path.join(directory, sampleName + '_EuclideanDistances.csv'))
    interactions = distDf.loc[distDf['Distance']<distThreshold]
    interactions.to_csv(os.path.join(directory, sampleName + '_DistThreshold' + str(distThreshold) + '_Interactions.csv'))
    return(interactions)


def compareAndPlotInteractions(directory, group1, group2):
    """Compare results from calcMAinteractions with a t-test, and create bar graph of two groups.

    Parameters
    ----------
    directory : string (optional)
        Directory containing interaction data.
    group1 : list of strings
        Trial IDs for group 1.
    group22 : string
        Trial IDs for group 2.

    Returns
    -------
    Pandas dataframe of results. Also saves bar graph.

    """
    
    files = os.listdir(directory)
    samplesG1 = []
    countsG1 = []
    samplesG2 = []
    countsG2 = []
    for f in files:
        if f.endswith('_Interactions.csv'):
            sampleName = f.split('/')[0].split('_Dist')[0]
            df = pd.read_csv(os.path.join(directory, f), index_col=0)
            count = df.shape[0]
            if sampleName in group1:
                samplesG1.append(sampleName)
                countsG1.append(count)
            elif sampleName in group2:
                samplesG2.append(sampleName)
                countsG2.append(count)
                
    g1_arr = np.array(countsG1)
    g2_arr = np.array(countsG2)
    g1mean = np.mean(countsG1)
    g2mean = np.mean(countsG2)
    g1sem = np.std(countsG1)/np.sqrt(len(countsG1))
    g2sem = np.std(countsG2)/np.sqrt(len(countsG2))
    err = [g1sem, g2sem]
    pval = np.array([ttest_ind(g1_arr, g2_arr, nan_policy='omit')])[0][1]
    fdr_pass,qvals,_,_ = multipletests(pval, method='fdr_bh',alpha=0.05)
    
    res = [g1mean, g2mean, g1sem, g2sem, pval]
    resNames = ['Group1_Mean', 'Group2_Mean', 'Group1_SEM', 'Group2_SEM', 't-test p-value']
    result = pd.DataFrame([samplesG1, countsG1, samplesG2, countsG2, resNames, res]).T
    result.columns=['Trial_Group1', 'InteractionCount_Group1', 'Trial_Group2', 'InteractionCount_Group2', 'Statistic', 'Value']
    result.to_csv(os.path.join(directory, 'Combined_InteractionCounts.csv'))

    plt.figure()
    plt.bar(['Group1','Group2'],[g1mean, g2mean])
    plt.errorbar(['Group1', 'Group2'], [g1mean, g2mean], yerr=err, fmt='o', color='r')
    plt.ylabel('# Frames Interacting', fontsize=16)
    plt.savefig(os.path.join(directory, 'InteractionFrames.png'))
    plt.show()
    return(result)

def calcDistanceToObject(file, bodyPart, obj, directory=os.getcwd(), pcutoff=None, distThreshold=None, stationary=False):
    """Calculate the distance between a labaled bodyPart and an object in single-animal DLC,
    and optionally calculate # of frames bodyPart is interacting with object.

    Parameters
    ----------
    file : string
        Path to h5 or csv file containing tracking data.
    bodyPart : string
        Body part to calculate distance from.
    obj : string
        Name of labeled object.
    directory : string (optional)
        Directory to save to. Default current working directory.
    pcutoff : float
        Likelihood threshold, frames below this will be interpolated.
    distThreshold : int/float (optional)
        Pixel distance threshold to be considered interacting. The default is None (no interaction will be calculated).
    stationary : bool
        If the object is stationary, pass True and the average detected coordinate will be used rather than per-frame coordinate.
    Returns
    -------
    Pandas dataframe of distances (optionally also interactions).
    """
    if file.endswith('.h5'):
        df = pd.read_hdf(file)
    elif file.endswith('.csv'):
        df = pd.read_csv(file, header=[0,1,2], index_col=0)
    sampleName = file.split('/')[-1].split('DLC')[0]
    scorerName = df.columns[0][0]
    if pcutoff:
        df.loc[df[(scorerName, bodyPart, 'likelihood')] < pcutoff, (scorerName, bodyPart, 'x')]=None
        df.loc[df[(scorerName, bodyPart, 'likelihood')] < pcutoff, (scorerName, bodyPart, 'y')]=None        
    bp1x = df[(scorerName, bodyPart, 'x')]
    bp1x.interpolate(method='linear', inplace=True)
    bp1y = df[(scorerName, bodyPart, 'y')]
    bp1y.interpolate(method='linear', inplace=True)
    objx = df[(scorerName, obj, 'x')]
    objx.interpolate(method='linear', inplace=True)
    objy = df[(scorerName, obj, 'y')]
    objy.interpolate(method='linear', inplace=True)
    objcoords = list(zip(objx.tolist(), objy.tolist()))
    bp1coords = list(zip(bp1x.tolist(), bp1y.tolist()))
    if stationary:
        objx=np.mean(objx)
        objy=np.mean(objy)
        objcoords=(objx, objy)
    dists = []
    for i in range(len(bp1coords)):
        if not stationary:
            d = np.linalg.norm(np.array(bp1coords[i])-np.array(objcoords[i]))
            dists.append(d)
        elif stationary:
            d = np.linalg.norm(np.array(bp1coords[i])-np.array(objcoords))
            dists.append(d)
    distDf = pd.DataFrame([bp1coords,objcoords,dists]).T
    distDf.columns=[bodyPart, obj, 'Distance']
    distDf.to_csv(os.path.join(directory, sampleName + '_EuclideanDistances.csv'))
    distDf.to_hdf(os.path.join(directory, sampleName + '_EuclideanDistances.h5'), key='df_with_missing')
    if distThreshold:
        interactions = distDf.loc[distDf['Distance']<distThreshold]
        interactions.to_csv(os.path.join(directory, sampleName + '_DistThreshold' + str(distThreshold) + '_Interactions.csv'))
        interactions.to_hdf(os.path.join(directory, sampleName + '_DistThreshold' + str(distThreshold) + '_Interactions.h5'))
        return(interactions)
    elif not distThreshold:
        return(distDf)

def calcVelocity(file, bodyParts, directory, window=5):
    """Calculate velocity of bodyParts between given number of frames (window).
    
    Parameters
    ----------
    file : string
        Path to .h5 or .csv file with data.
    bodyParts : list or array
        Labeled parts to calculate velocity of.
    directory : string
        Directory to save result.
    window : int (optional, default 5)
        Size of window to calculate velocity across.
    """
    if file.endswith('.h5'):
        df = pd.read_hdf(file)
    elif file.endswith('.csv'):
        df = pd.read_csv(file, header=[0,1,2], index_col=0)
    sampleName = file.split('/')[-1].split('DLC')[0]
    scorerName = df.columns[0][0]
    cat = pd.DataFrame()
    for bp in bodyParts:
        velocities = []
        frames = []
        bpx = df[(scorerName, bp, 'x')].tolist()
        bpy = df[(scorerName, bp, 'y')].tolist()
        lz = list(zip(bpx, bpy))
        for i in range(window, len(lz), window):
            velo = np.linalg.norm(np.array(lz[i])-np.array(lz[i-window]))
            velocities.append(velo)
            frames.append(i)
        cat[bp]=velocities
    cat['average']=cat.mean(axis=1)
    cat.index=frames
    cat.to_csv(os.path.join(directory, sampleName + '_Velocity_Window' + str(window) + '.csv'))
    return(cat)


def makeObjectStationary(directory, objectName, dtype='.h5', pcutoff=0.5, file=None):
    """If a stationary object is in the video, alter result data files to reflect it's mean position.
    
    Parameters
    ----------
    directory : str
        Path to directory where data files are kept.
    objectName : str
        Label of stationary object.
    dtype : str (optional, default .h5)
        Type of data file to load, can be '.h5' or '.csv'
    pcutoff : float, (optional, default 0.5)
        Cutoff for likelihood, frames below this will be dropped to nan.
    file : str (optional, default None)
        Path to single file to process, otherwise all files matching dtype in directory will be processed.
    """
    files = []
    if not file:
        allFiles = os.listdir(directory)
        for f in allFiles:
            if f.endswith(dtype):
                files.append(f)
    elif file:
        files = file
        if len(files.split('/'))>1:
            files = [os.path.basename(files)]
    for file in files:
        n, e = os.path.splitext(file)
        fullpath = os.path.join(directory, file)
        if fullpath.endswith('.h5'):
            df = pd.read_hdf(fullpath)
        elif fullpath.endswith('.csv'):
            df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
        scorerName = df.columns[0][0]
        df.loc[df[(scorerName, objectName, 'likelihood')]<pcutoff]=np.nan
        meanX = df[(scorerName, objectName, 'x')].mean()
        meanY = df[(scorerName, objectName, 'y')].mean()
        df[(scorerName, objectName, 'x')]=meanX
        df[(scorerName, objectName, 'y')]=meanY
        df[(scorerName, objectName, 'likelihood')]=1.0
        if dtype=='.h5':
            df.to_hdf(os.path.join(directory, n + '_stationary'+objectName+e), key='df_with_missing')
        elif dtype=='.csv':
            df.to_csv(os.path.join(directory, n + '_stationary'+objectName+e))

def alignToObject(h5Path, obj):
    """Set object position to 0,0 for all frames, and create data file with all other bodyparts coordinate distance from object.
    
    Parameters
    ----------
    h5Path : string
        Path to .h5 file.
    obj : string
        Object label to align to.
    """
    directory = '/'.join(h5Path.split('/')[:-1])
    fileName = h5Path.split('/')[-1]#.split('DLC')[0]
    f, e = os.path.splitext(fileName)
    df = pd.read_hdf(h5Path)
    cols = df.columns
    scorerName=cols[0][0]
    bodyParts = []
    for col in cols:
        bp = col[1]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts)) #remove duplicates
    df_ego = df
    bodyParts_norm = bodyParts
    bodyParts_norm.remove(obj)
    for bp in bodyParts_norm: #normalize bodyparts by subtracting one from all 
        df_ego[(scorerName, bp, 'x')] = df_ego[(scorerName, bp, 'x')] - df_ego[(scorerName, obj, 'x')] #for x position
        df_ego[(scorerName, bp, 'y')] = df_ego[(scorerName, bp, 'y')] - df_ego[(scorerName, obj, 'y')] #for y position
    df_ego[(scorerName, obj, 'x')]=0
    df_ego[(scorerName, obj, 'y')]=0
    if not os.path.exists(os.path.join(directory, 'alignedTo'+obj+'/')):
        os.mkdir(os.path.join(directory, 'alignedTo'+obj+'/'))
    df_ego.to_hdf(os.path.join(directory, 'alignedTo'+obj+'/' + f + '_alignedTo'+obj+'.h5'), key='df_with_missing')

def combineDistancesPerGroup(dataFiles, bodyPart, obj):
    f0 = pd.read_hdf(dataFiles[0])
    cols = f0.columns
    bodyParts = []
    cat = pd.DataFrame()
    for col in cols:
        bp = col[1]
        bodyParts.append(bp)
    bodyParts = list(set(bodyParts))
    bodyParts.remove(bodyPart)
    for i in range(len(dataFiles)):
        df = pd.read_hdf(dataFiles[i])
        df.drop(obj, axis=1, level=1, inplace=True)
        for bp in bodyParts:
            df.drop(bp, axis=1, level=1, inplace=True)
        cat = pd.concat([cat, df],axis=0,ignore_index=True)
    return(cat)

def compareAndPlotInteractions(directory, group1, group2):
    """Compare results from calcMAinteractions with a t-test, and create bar graph of two groups.

    Parameters
    ----------
    directory : string (optional)
        Directory containing interaction data.
    group1 : list of strings
        Trial IDs for group 1.
    group22 : string
        Trial IDs for group 2.

    Returns
    -------
    Pandas dataframe of results. Also saves bar graph.

    """
    
    files = os.listdir(directory)
    samplesG1 = []
    countsG1 = []
    samplesG2 = []
    countsG2 = []
    for f in files:
        if f.endswith('_Interactions.csv'):
            sampleName = f.split('/')[0].split('_Dist')[0]
            df = pd.read_csv(os.path.join(directory, f), index_col=0)
            count = df.shape[0]
            if sampleName in group1:
                samplesG1.append(sampleName)
                countsG1.append(count)
            elif sampleName in group2:
                samplesG2.append(sampleName)
                countsG2.append(count)
                
    g1_arr = np.array(countsG1)
    g2_arr = np.array(countsG2)
    g1mean = np.mean(countsG1)
    g2mean = np.mean(countsG2)
    g1sem = np.std(countsG1)/np.sqrt(len(countsG1))
    g2sem = np.std(countsG2)/np.sqrt(len(countsG2))
    err = [g1sem, g2sem]
    pval = np.array([ttest_ind(g1_arr, g2_arr, nan_policy='omit')])[0][1]
    fdr_pass,qvals,_,_ = multipletests(pval, method='fdr_bh',alpha=0.05)
    
    res = [g1mean, g2mean, g1sem, g2sem, pval]
    resNames = ['Group1_Mean', 'Group2_Mean', 'Group1_SEM', 'Group2_SEM', 't-test p-value']
    result = pd.DataFrame([samplesG1, countsG1, samplesG2, countsG2, resNames, res]).T
    result.columns=['Trial_Group1', 'InteractionCount_Group1', 'Trial_Group2', 'InteractionCount_Group2', 'Statistic', 'Value']
    result.to_csv(os.path.join(directory, 'Combined_InteractionCounts.csv'))

def calcVelocity(file, bodyParts, directory, window=5):
    """Calculate velocity of bodyParts between given number of frames (window).
    
    Parameters
    ----------
    file : string
        Path to .h5 or .csv file with data.
    bodyParts : list or array
        Labeled parts to calculate velocity of.
    directory : string
        Directory to save result.
    window : int (optional, default 5)
        Size of window to calculate velocity across.
    """
    if file.endswith('.h5'):
        df = pd.read_hdf(file)
    elif file.endswith('.csv'):
        df = pd.read_csv(file, header=[0,1,2], index_col=0)
    sampleName = file.split('/')[-1].split('DLC')[0]
    scorerName = df.columns[0][0]
    cat = pd.DataFrame()
    for bp in bodyParts:
        velocities = []
        frames = []
        bpx = df[(scorerName, bp, 'x')].tolist()
        bpy = df[(scorerName, bp, 'y')].tolist()
        lz = list(zip(bpx, bpy))
        for i in range(window, len(lz), window):
            velo = np.linalg.norm(np.array(lz[i])-np.array(lz[i-window]))
            velocities.append(velo)
            frames.append(i)
        cat[bp]=velocities
    cat['average']=cat.mean(axis=1)
    cat.index=frames
    cat.to_csv(os.path.join(directory, sampleName + '_Velocity_Window' + str(window) + '.csv'))
    return(cat)
