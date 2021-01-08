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
    df = pd.read_csv(csvPath, header=[0,1,2], index_col=0)
    sx = df[(modelPrefix, 'spine2', 'x')]
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

def extractZones(directory, modelPrefix, bodyPart, axis='x', flipped=False):
    files = os.listdir(directory)
    if not os.path.exists(os.path.join(directory, 'SplitZoneData/')):
        os.mkdir(os.path.join(directory, 'SplitZoneData/'))
    for f in files:
        if f.endswith('.csv'):
            n, e = os.path.splitext(f)
            fullpath = os.path.join(directory, f)
            df = pd.read_csv(fullpath, header=[0,1,2], index_col=0)
            leftTime, rightTme, leftIndex, rightIndex = calcZoneTimes(fullpath, modelPrefix, bodyPart, axis, flippedX=flipped, index=True)
            leftData = df.loc[df.index.isin(leftIndex)]
            rightData = df.loc[df.index.isin(rightIndex)]
            leftData.to_csv(os.path.join(directory, 'SplitZoneData/' + n + '_LeftZone.csv'))
            rightData.to_csv(os.path.join(directory, 'SplitZoneData/' + n + '_RightZone.csv'))


def extractFrames(vidPath, saveDir):
    sampleName = vidPath.split('/')[-1].strip('.mp4')

    if not os.path.exists(os.path.join(saveDir, sampleName + '/')):
        os.mkdir(os.path.join(saveDir, sampleName))
        
    vc = cv2.VideoCapture(vidPath)
    c=0
    
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval=False
    
    while rval:
        rval, frame = vc.read()
        cv2.imwrite(os.path.join(saveDir, sampleName + '/' + str(c) + '.jpg'), frame)
        c = c+1
        cv2.waitKey(1)
    vc.release()

def makeZoneVideo(csvPath, modelPrefix, frameDir, fps, size, flippedX=True):
    sampleName = frameDir.split('/')[-1]
    if sampleName == '':
        sampleName = frameDir.split('/')[-2]
    leftDir = os.path.join(frameDir, 'Left/')
    rightDir = os.path.join(frameDir, 'Right/')
    if not os.path.exists(leftDir):
        os.mkdir(leftDir)
    if not os.path.exists(rightDir):
        os.mkdir(rightDir)

    left, right, leftIndex, rightIndex = calcZoneTimes(csvPath, modelPrefix, flippedX=flippedX, index=True)
    paths = os.listdir(os.path.join(frameDir, sampleName + '/'))
    for path in paths:
        fullpath = os.path.join(frameDir, sampleName + '/' + path)
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


def countBouts(csvPath, modelPrefix, saveDir, flippedX=True):
    sampleName = csvPath.split('/')[-1].split('DLC')[0]
    left, right, leftIndex, rightIndex = calcZoneTimes(csvPath, modelPrefix, flippedX=flippedX, index=True)
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
    df.to_csv(os.path.join(saveDir, sampleName + '_ZoneBouts.csv'))
    return df    


