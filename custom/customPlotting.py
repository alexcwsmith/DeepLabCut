"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import os

####################################################
# Dependencies
####################################################
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import ttest_ind

from custom import customVisualization as cvis
from custom import customAuxiliaryFunctions as caux
from custom import customAuxfuncs_multianimal as cxma

#from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization


def Histogram(vector, color, bins, ax=None, linewidth=1.0):
    dvector = np.diff(vector)
    dvector = dvector[np.isfinite(dvector)]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.hist(dvector, color=color, histtype="step", bins=bins, linewidth=linewidth)


def PlottingResults(
    tmpfolder,
    Dataframe,
    cfg,
    bodyparts2plot,
    individuals2plot,
    object2plot=None,
    objectsize=None,
    showfigures=False,
    suffix=".png",
    resolution=100,
    linewidth=1.0,
):
    """ Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods."""
    pcutoff = cfg["pcutoff"]
    colors = cvis.get_cmap(len(bodyparts2plot), name=cfg["colormap"])
    alphavalue = cfg["alphavalue"]
    if individuals2plot:
        Dataframe = Dataframe.loc(axis=1)[:, individuals2plot]
    animal_bpts = Dataframe.columns.get_level_values("bodyparts")
    # Pose X vs pose Y
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("X position in pixels")
    ax1.set_ylabel("Y position in pixels")
    ax1.invert_yaxis()

    # Poses vs time
    fig2 = plt.figure(figsize=(10, 3))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("X-(dashed) and Y- (solid) position in pixels")

    # Likelihoods
    fig3 = plt.figure(figsize=(10, 3))
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Likelihood (use to set pcutoff)")

    # Histograms
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_ylabel("Count")
    ax4.set_xlabel("DeltaX and DeltaY")
    bins = np.linspace(0, np.amax(Dataframe.max()), 100)

    with np.errstate(invalid="ignore"):
        for bpindex, bp in enumerate(bodyparts2plot):
            if (
                bp in animal_bpts
            ):  # Avoid 'unique' bodyparts only present in the 'single' animal
                prob = Dataframe.xs(
                    (bp, "likelihood"), level=(-2, -1), axis=1
                ).values.squeeze()
                mask = prob < pcutoff
                temp_x = np.ma.array(
                    Dataframe.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
                    mask=mask,
                )
                temp_y = np.ma.array(
                    Dataframe.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
                    mask=mask,
                )
                if object2plot:
                    obj_x = np.ma.array(
                        Dataframe.xs((object2plot, "x"), level=(-2, -1), axis=1).values.squeeze(), mask=mask
                        )
                    obj_y = np.ma.array(
                        Dataframe.xs((object2plot, "y"), level=(-2, -1), axis=1).values.squeeze(), mask=mask
                        )   
                    ax1.plot(obj_x, obj_y, marker='o', markerfacecolor='None', markeredgecolor='red', markeredgewidth=2, ms=objectsize)
                ax1.plot(temp_x, temp_y, ".", color=colors(bpindex), alpha=alphavalue)

                ax2.plot(
                    temp_x,
                    "--",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )
                ax2.plot(
                    temp_y,
                    "-",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )
                ax3.plot(
                    prob,
                    "-",
                    color=colors(bpindex),
                    linewidth=linewidth,
                    alpha=alphavalue,
                )

                Histogram(temp_x, colors(bpindex), bins, ax4, linewidth=linewidth)
                Histogram(temp_y, colors(bpindex), bins, ax4, linewidth=linewidth)

    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cfg["colormap"]),
        norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot) - 1),
    )
    sm._A = []
    for ax in ax1, ax2, ax3, ax4:
        cbar = plt.colorbar(sm, ax=ax, ticks=range(len(bodyparts2plot)))
        cbar.set_ticklabels(bodyparts2plot)

    fig1.savefig(
        os.path.join(tmpfolder, "trajectory" + suffix),
        bbox_inches="tight",
        dpi=resolution,
    )
    fig2.savefig(
        os.path.join(tmpfolder, "plot" + suffix), bbox_inches="tight", dpi=resolution
    )
    fig3.savefig(
        os.path.join(tmpfolder, "plot-likelihood" + suffix),
        bbox_inches="tight",
        dpi=resolution,
    )
    fig4.savefig(
        os.path.join(tmpfolder, "hist" + suffix), bbox_inches="tight", dpi=resolution
    )

    if not showfigures:
        plt.close("all")
    else:
        plt.show()


##################################################
# Looping analysis over video
##################################################


def plot_trajectories(
    config,
    videos,
    videotype=".avi",
    shuffle=1,
    trainingsetindex=0,
    filtered=False,
    displayedbodyparts="all",
    displayedobjects=None,
    objectsize=None,
    displayedindividuals="all",
    showfigures=False,
    destfolder=None,
    modelprefix="",
    track_method="",
    imagetype=".png",
    resolution=100,
    linewidth=1.0,
):
    """
    Plots the trajectories of various bodyparts across the video.

    Parameters
    ----------
     config : string
    Full path of the config.yaml file as a string.

    videos : list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed. The default is ``.avi``

    shuffle: list, optional
    List of integers specifying the shuffle indices of the training dataset. The default is [1]

    trainingsetindex: int, optional
    Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    filtered: bool, default false
    Boolean variable indicating if filtered output should be plotted rather than frame-by-frame predictions. Filtered version can be calculated with deeplabcut.filterpredictions

    displayedbodyparts: list of strings, optional
        This select the body parts that are plotted in the video.
        Either ``all``, then all body parts from config.yaml are used,
        or a list of strings that are a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml to select only these two body parts.

    displayedobjects : string
        Label of object to plot.
        
    objectsize : int
        Size of object to plot, in pixels.

    showfigures: bool, default false
    If true then plots are also displayed.

    destfolder: string, optional
        Specifies the destination folder that was used for storing analysis data (default is the path of the video).

    imagetype: string, default ".png"
        Specifies the output image format, tested '.tif', '.jpg', '.svg' and ".png". 

    resolution: int, default 100
        Specifies the resolution (in dpi) of saved figures. Note higher resolution figures take longer to generate.

    linewidth: float, default 1.0
        Specifies width of line for line and histogram plots.

    imagetype: string, default ".png"
        Specifies the output image format, tested '.tif', '.jpg', '.svg' and ".png". 

    resolution: int, default 100
        Specifies the resolution (in dpi) of saved figures. Note higher resolution figures take longer to generate.

    linewidth: float, default 1.0
        Specifies width of line for line and histogram plots.

    Example
    --------
    for labeling the frames
    >>> deeplabcut.plot_trajectories('home/alex/analysis/project/reaching-task/config.yaml',['/home/alex/analysis/project/videos/reachingvideo1.avi'])
    --------

    """
    cfg = caux.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    DLCscorer, DLCscorerlegacy = caux.GetScorerName(
        cfg, shuffle, trainFraction, modelprefix=modelprefix
    )  # automatically loads corresponding model (even training iteration based on snapshot index)
    bodyparts = caux.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, displayedbodyparts
    )
    individuals = cxma.IntersectionofIndividualsandOnesGivenbyUser(
        cfg, displayedindividuals
    )
    Videos = caux.Getlistofvideos(videos, videotype)
    if not len(Videos):
        print(
            "No videos found. Make sure you passed a list of videos and that *videotype* is right."
        )
        return

    failed = []
    for video in Videos:
        if destfolder is None:
            videofolder = str(Path(video).parents[0])
        else:
            videofolder = destfolder

        vname = str(Path(video).stem)
        print("Loading ", video, "and data.")
        try:
            df, _, _, suffix = caux.load_analyzed_data(
                videofolder, vname, DLCscorer, filtered, track_method
            )
            failed.append(False)
            tmpfolder = os.path.join(videofolder, "plot-poses", vname)
            caux.attempttomakefolder(tmpfolder, recursive=True)
            # Keep only the individuals and bodyparts that were labeled
            labeled_bpts = [
                bp
                for bp in df.columns.get_level_values("bodyparts").unique()
                if bp in bodyparts
            ]
            for animal in individuals:
                PlottingResults(
                    tmpfolder,
                    df,
                    cfg,
                    labeled_bpts,
                    animal,
                    displayedobjects,
                    objectsize,
                    showfigures,
                    suffix + animal + imagetype,
                    resolution=resolution,
                    linewidth=linewidth,
                )
        except FileNotFoundError as e:
            failed.append(True)
            print(e)
            try:
                _ = caux.load_detection_data(
                    video, DLCscorer, track_method
                )
                print(
                    'Call "deeplabcut.refine_training_dataset.convert_raw_tracks_to_h5()"'
                    " prior to plotting the trajectories."
                )
            except FileNotFoundError as e:
                print(e)
                print(
                    f"Make sure {video} was previously analyzed, and that "
                    f'detections were successively converted to tracklets using "deeplabcut.convert_detections2tracklets()" '
                    f'and "deeplabcut.convert_raw_tracks_to_h5()".'
                )

    if not all(failed):
        print(
            'Plots created! Please check the directory "plot-poses" within the video directory'
        )
    else:
        print(
            f"Plots could not be created! "
            f"Videos were not evaluated with the current scorer {DLCscorer}."
        )

def plotAverageTrajectories(DataFrame, directory, groupName, bodyparts2plot, object2plot, objectsize=25, pcutoff=.5, imagetype='.png', resolution=100, linewidth=1.0):
    alphavalue = .005
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("X position in pixels")
    ax1.set_ylabel("Y position in pixels")
    ax1.invert_yaxis()
    
    with np.errstate(invalid="ignore"):
        for bpindex, bp in enumerate(bodyparts2plot):
            prob = DataFrame.xs(
                (bp, "likelihood"), level=(-2, -1), axis=1
            ).values.squeeze()
            mask = prob < pcutoff
            temp_x = np.ma.array(
                DataFrame.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
                mask=mask,
            )
            temp_y = np.ma.array(
                DataFrame.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
                mask=mask,
            )
            if object2plot:
                obj_x = 0
                obj_y = 0
                ax1.plot(obj_x, obj_y, marker='o', markerfacecolor='None', markeredgecolor='red', markeredgewidth=2, ms=objectsize)
            ax1.plot(temp_x, temp_y, ".", color='blue', alpha=alphavalue)
    
    fig1.savefig(
        os.path.join(directory, "trajectory_" + groupName + imagetype),
        bbox_inches="tight",
        dpi=resolution,
    )
    
def plotAverageDistHistogram(group1df, group2df, group1name, group2name, directory, bodyparts2plot, object2plot, pcutoff=None, downsample=2, imagetype='.png', resolution=100):
    scorerName = group1df.columns[0][0]
    dists_g1 = []
    dists_g2 = []
    group1df.dropna(how='any', axis=0, inplace=True)
    group2df.dropna(how='any', axis=0, inplace=True)
    dataFrames = [group1df, group2df]
    for i in range(len(dataFrames)):
        DataFrame = dataFrames[i]
        if len(DataFrame.columns)==3:
            if pcutoff:
                if isinstance(bodyparts2plot, list):
                    DataFrame.loc[DataFrame[(scorerName, bodyparts2plot[0], 'likelihood')]<pcutoff, (scorerName, bodyparts2plot[0], 'likelihood')]=np.nan
                else:
                    DataFrame.loc[DataFrame[(scorerName, bodyparts2plot, 'likelihood')]<pcutoff, (scorerName, bodyparts2plot, 'likelihood')]=np.nan  
            DataFrame.dropna(how='any',axis=0, inplace=True)
            DataFrame.drop('likelihood', axis=1, level=2, inplace=True)
        for f in range(0, DataFrame.shape[0], downsample):
            d = euclidean(np.array(DataFrame)[f], np.array([0,0]))
            if i == 0:
                dists_g1.append(d)
            elif i == 1:
                dists_g2.append(d)
    distDf_g1 = pd.DataFrame(dists_g1)
    distDf_g2 = pd.DataFrame(dists_g2)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_xlabel("Distance to " + object2plot + ' (pixels)', fontsize=14)
    ax1.set_ylabel("Frequency", fontsize=16)
    ax2.set_xlabel("Distance to " + object2plot + ' (pixels)', fontsize=14)
    ax2.set_ylabel("Frequency", fontsize=16)
    
    distDf_g1.hist(grid=False, ax=ax1, bins=50, xlabelsize=12, ylabelsize=12)
    distDf_g2.hist(grid=False, ax=ax2, bins=50, xlabelsize=12, ylabelsize=12)
    ax1.set_title(group1name, fontsize=18)
    ax2.set_title(group2name, fontsize=18)
    
    fig.savefig(
        os.path.join(directory, "histogram_" + group1name + "_vs_" + group2name + imagetype),
        bbox_inches="tight",
        dpi=resolution,
    )
    
    stat, pvals = ttest_ind(distDf_g1, distDf_g2, nan_policy='omit')
    output = str('t-statistic ' + str(stat[0]) + '\n' + 'p-value ' + str(pvals[0]) + '\n' + group1name + '_N=' + 
                 str(distDf_g1.shape[0]) + '\n' + group2name + '_N='+str(distDf_g2.shape[0]) + '\n' + 'Sampled every ' + str(downsample) + ' frames')
    with open(os.path.join(directory, 't-test_'+group1name+'_vs_'+group2name+'.txt'), 'a') as f:
        f.write(output)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("video")
    cli_args = parser.parse_args()
