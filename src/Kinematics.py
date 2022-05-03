import SimpleITK as sitk
import numpy as np
import torch
import os
import glob
import skimage.measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import pandas as pd

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
          '#a65628', '#984ea3', '#e41a1c', '#dede00',"#9aa0ab"]

comps = ['Scapohid', 'Lunate', 'Triquetral', 'Pisiform',
         'Hamate', 'Capitate', 'Trapezoid', 'Trapezium','Ground Truth']


def _getSortedEigens(vals,vecs):
    '''
    Assumes the values and vectors are returned as per np.linalg.eig(), ie. as columns. Assumes 3D.
    :param vals: N eigenvalues resulting from eigendecomposition
    :param vecs: N eigenvectors stored in a single array
    :return: sorted values and vectors so that values and vectors still match
    '''
    vec0 = vecs[:,0]
    vec1 = vecs[:,1]
    vec2 = vecs[:,2]
    vectors = [vec0,vec1,vec2]
    return (list(t) for t in zip(*sorted(zip(vals,vectors),reverse=True)))


def _getPatches(hasGroundTruth:bool=False):
    out = []
    if not hasGroundTruth:
        for i in range(0,8):
            out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    else:
        for i in range(0,9):
            out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    return out


def _getWarpedComponentPlots(label: str):
    vAlpha = 0.05
    vElevation = 90
    vAzim = 0

    fpResults = "../images/results"
    fpIn = os.path.join(fpResults,label)
    fpOut = os.path.join(fpIn,label,"plots","estimations")
    if not os.path.exists(fpOut):
        os.mkdir(fpOut)
    vDesiredComponents = range(1,9)
    vComponentPrefix = "warped_seg_"
    patches = _getPatches()

    # Create a list of coordinates
    Pd = torch.arange(0,60,dtype=torch.float32)
    Ph = torch.arange(0,100,dtype=torch.float32)
    Pw = torch.arange(0,100,dtype=torch.float32)
    P = torch.cartesian_prod(Pd,Ph,Pw)
    P = P.reshape(60,100,100,3).detach().numpy()

    for file in glob.glob(os.path.join(fpIn,"frame_*_to_frame_*")):
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d',proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 60)
        ax.set_zticks([])
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .102, 0.64, .102), loc='lower left',
                      ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Estimated Carpal Bone Positions", fontsize=24)
        ax.set_title("Frame "+vSource+" to Frame "+vTarget, fontsize=14)
        fig.tight_layout()

        for comp in vDesiredComponents:
            compColor = comp-1

            # Read in the component warped label image
            img = sitk.ReadImage(os.path.join(fpIn,os.path.basename(file),vComponentPrefix+str(comp)+".nii"))
            npImg = sitk.GetArrayFromImage(img)

            # Plot the component point cloud
            ptCloud = np.where(np.stack((npImg,)*3,axis=-1) >= 0.7, P, 0)
            ptCloud = np.ma.masked_equal(ptCloud,0)
            ax.scatter(ptCloud[:,:,:,1].flatten(),
                       ptCloud[:,:,:,2].flatten(),
                       ptCloud[:,:,:,0].flatten(),
                       color=colors[compColor],
                       alpha=vAlpha)

            centroid = skimage.measure.centroid(npImg)
            moments = skimage.measure.inertia_tensor(npImg)
            lam, Q = np.linalg.eig(moments)
            lam, Q = _getSortedEigens(vals=lam,vecs=Q)
            frame1 = Q[0]/np.linalg.norm(Q[0])
            frame2 = Q[1]/np.linalg.norm(Q[1])
            frame3 = Q[2]/np.linalg.norm(Q[2])
            ax.scatter(centroid[1],centroid[2],centroid[0],
                       s=100,color=colors[compColor],edgecolors='black')

            ax.quiver(centroid[1],centroid[2],centroid[0],*frame1,color='r',length=lam[0]*0.2)
            ax.quiver(centroid[1],centroid[2],centroid[0],*frame2,color='g',length=lam[1]*0.2)
            ax.quiver(centroid[1],centroid[2],centroid[0],*frame3,color='b',length=lam[2]*0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame1), color='r', length=lam[0] * 0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame2), color='g', length=lam[1] * 0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame3), color='b', length=lam[2] * 0.2)

        fig.savefig(os.path.join(fpOut,os.path.basename(file)),bbox_inches='tight')
        fig.clear()
        plt.close()


def _getModelAccuracy_Centroids(label):
    colorGray = "#9aa0ab"
    vElevation = 90
    vAzim = 0

    fpGroundTruths = "../images/input/rave"
    vTruthPrefix = "em_frame_"

    fpResults = "../images/results"
    fpIn_Preds = os.path.join(fpResults,label)
    vComponentPrefix = "warped_seg_"

    fpOut = os.path.join(fpIn_Preds, label, "plots","modelAccuracy_Centroids")

    if not os.path.exists(fpOut):
        os.mkdir(fpOut)

    vDesiredComponents = range(1, 9)
    patches = _getPatches(True)

    figErr, axErr = plt.subplots(figsize=(10,10))
    figErr.suptitle("Translation Error Between Prediction and Ground Truth")
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("Euclidean Distance between Centroids")
    axErr.yaxis.grid(True)
    figErr.tight_layout()
    aAllErrors = {}

    for file in glob.glob(os.path.join(fpIn_Preds,"frame_*_to_frame_*")):
        aFrameErrors = np.zeros(8)
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d',proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 60)
        ax.set_zticks([])
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .102, 0.64, .102), loc='lower left',
                      ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Centroids for Ground Truth vs Predictions", fontsize=24)
        ax.set_title("Frame "+vSource+" to Frame "+vTarget, fontsize=14)
        fig.tight_layout()

        for comp in vDesiredComponents:
            compColor = comp - 1

            # Read in the component warped label image
            imgPred = sitk.ReadImage(os.path.join(fpIn_Preds, os.path.basename(file),
                                              vComponentPrefix + str(comp) + ".nii"))
            npImgPred = sitk.GetArrayFromImage(imgPred)

            imgTruth = sitk.ReadImage(os.path.join(fpGroundTruths,vTruthPrefix + str(vTarget)
                                                   + '_seg_' + str(comp) + '.nii.gz'))
            npImgTruth = sitk.GetArrayFromImage(imgTruth)[10:70, 60:160, 50:150] # crop taken from main
                                                                                 # Should probably be a utility func.

            predCentroid = skimage.measure.centroid(npImgPred)
            truthCentroid = skimage.measure.centroid(npImgTruth)
            errorVector_Centroid = truthCentroid - predCentroid
            aFrameErrors[comp-1] = np.linalg.norm(errorVector_Centroid,ord=2)

            ax.scatter(truthCentroid[1],truthCentroid[2],truthCentroid[0],
                       s=200,color=colorGray,alpha=0.5,edgecolors='black')
            ax.scatter(predCentroid[1],predCentroid[2],predCentroid[0],
                       s=200,color=colors[compColor],alpha=0.5,edgecolors='black')
            ax.quiver(predCentroid[1],predCentroid[2],predCentroid[0],
                      errorVector_Centroid[1],errorVector_Centroid[2],errorVector_Centroid[0],
                      color=colorGray,length=np.linalg.norm(errorVector_Centroid)*0.2)
        fig.savefig(os.path.join(fpOut, os.path.basename(file)), bbox_inches='tight')
        fig.clear()
        plt.close()
        aAllErrors[vTarget] = aFrameErrors

    frames,errors = [*zip(*aAllErrors.items())]
    axErr.boxplot(errors,patch_artist=True,boxprops=dict(facecolor='lightgray'),medianprops=dict(color='k'))
    figErr.savefig(os.path.join(fpOut,"summaryErrorTranslation"),bbox_inches='tight')
    plt.show()
    figErr.clear()
    plt.close()


def _getModelAccuracy_Rotations(label):
    patches = _getPatches(True)
    vElevation = 90
    vAzim = 0

    fpGroundTruths = "../images/input/rave"
    vTruthPrefix = "em_frame_"

    fpResults = "../images/results"
    fpIn_Preds = os.path.join(fpResults,label)
    vComponentPrefix = "warped_seg_"

    fpOut = os.path.join(fpIn_Preds, label, "plots","modelAccuracy_Rotations")

    if not os.path.exists(fpOut):
        os.mkdir(fpOut)

    vDesiredComponents = range(1, 9)

    figErr, axErr = plt.subplots(figsize=(10, 10))
    figErr.suptitle("Rotation Error Between Predictions and Ground Truth")
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("1 - cos(Theta) of Angle Between Primary Axes")
    axErr.yaxis.grid(True)
    figErr.tight_layout()

    aAllErrors = {}

    # Create a list of coordinates
    Pd = torch.arange(0, 60, dtype=torch.float32)
    Ph = torch.arange(0, 100, dtype=torch.float32)
    Pw = torch.arange(0, 100, dtype=torch.float32)
    P = torch.cartesian_prod(Pd, Ph, Pw)
    P = P.reshape(60, 100, 100, 3).detach().numpy()

    for file in glob.glob(os.path.join(fpIn_Preds,"frame_*_to_frame_*")):
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]
        aFrameErrors = np.zeros(len(vDesiredComponents))

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 60)
        ax.set_zticks([])
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .102, 0.64, .102), loc='lower left',
                  ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Centroids for Ground Truth vs Predictions", fontsize=24)
        ax.set_title("Frame " + vSource + " to Frame " + vTarget, fontsize=14)
        fig.tight_layout()

        for comp in vDesiredComponents:
            # Read in the component warped label image
            imgPred = sitk.ReadImage(os.path.join(fpIn_Preds, os.path.basename(file),
                                              vComponentPrefix + str(comp) + ".nii"))
            npImgPred = sitk.GetArrayFromImage(imgPred)

            imgTruth = sitk.ReadImage(os.path.join(fpGroundTruths,vTruthPrefix + str(vTarget)
                                                   + '_seg_' + str(comp) + '.nii.gz'))
            npImgTruth = sitk.GetArrayFromImage(imgTruth)[10:70, 60:160, 50:150] # crop taken from main
                                                                                 # Should probably be a utility func.

            ptCloud = np.where(np.stack((npImgPred,) * 3, axis=-1) >= 0.7, P, 0)
            ptCloud = np.ma.masked_equal(ptCloud, 0)
            ax.scatter(ptCloud[:, :, :, 1].flatten(),
                       ptCloud[:, :, :, 2].flatten(),
                       ptCloud[:, :, :, 0].flatten(),
                       color='blue',
                       alpha=0.01)
            '''
            ptCloud = np.where(np.stack((npImgTruth,) * 3, axis=-1) >= 0.7, P, 0)
            ptCloud = np.ma.masked_equal(ptCloud, 0)
            ax.scatter(ptCloud[:, :, :, 1].flatten(),
                       ptCloud[:, :, :, 2].flatten(),
                       ptCloud[:, :, :, 0].flatten(),
                       color='gray',
                       alpha=0.01)
            '''
            predFrame = skimage.measure.inertia_tensor(npImgPred)
            predEigenvalues, predEigenvectors = np.linalg.eig(predFrame)
            predEigenvalues, predEigenvectors = _getSortedEigens(predEigenvalues,predEigenvectors)
            truthFrame = skimage.measure.inertia_tensor(npImgTruth)
            truthEigenvalues, truthEigenvectors = np.linalg.eig(truthFrame)
            truthEigenvalues, truthEigenvectors = _getSortedEigens(truthEigenvalues,truthEigenvectors)

            angle = np.divide(np.dot(predEigenvectors[0],truthEigenvectors[0]),
                           np.linalg.norm(predEigenvectors[0])*np.linalg.norm(truthEigenvectors[0]))
            aFrameErrors[comp-1] = 1 - abs(angle)

            predCentroid = skimage.measure.centroid(npImgPred)
            truthCentroid = skimage.measure.centroid(npImgTruth)

            ax.quiver(predCentroid[1], predCentroid[2], predCentroid[0], *predEigenvectors[0], color='r', length=predEigenvalues[0] * 0.2)
            ax.quiver(predCentroid[1], predCentroid[2], predCentroid[0], *predEigenvectors[1], color='g', length=predEigenvalues[1] * 0.2)
            ax.quiver(predCentroid[1], predCentroid[2], predCentroid[0], *predEigenvectors[2], color='b', length=predEigenvalues[2] * 0.2)
            ax.quiver(truthCentroid[1], truthCentroid[2], truthCentroid[0], *truthEigenvectors[0], color='k', length=truthEigenvalues[0] * 0.2)
            ax.quiver(truthCentroid[1], truthCentroid[2], truthCentroid[0], *truthEigenvectors[1], color='k', length=truthEigenvalues[1] * 0.2)
            ax.quiver(truthCentroid[1], truthCentroid[2], truthCentroid[0], *truthEigenvectors[2], color='k', length=truthEigenvalues[2] * 0.2)
        aAllErrors[vTarget] = aFrameErrors
        fig.savefig(os.path.join(fpOut, os.path.basename(file)), bbox_inches='tight')
        fig.clear()
        plt.close()
    frames, errors = [*zip(*aAllErrors.items())]
    axErr.boxplot(errors, patch_artist=True, boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'))
    figErr.savefig(os.path.join(fpOut, "summaryErrorRotation"), bbox_inches='tight')
    figErr.clear()
    plt.close()

def _getComponentTransformAccuracy(label):
    pass


def _getWarpField2ComponentAccuracy(label):
    pass


def getPlots(label):
    '''
    Driver function for Kinematic plotting functions.
    :param label: model label as stored in results folder.
    :return: None.
    '''
    if not os.path.exists(os.path.join("../images/results",label,label,"plots")):
        os.mkdir(os.path.join("../images/results",label,label,"plots"))
    #_getWarpedComponentPlots(label)
    #_getModelAccuracy_Centroids(label)
    _getModelAccuracy_Rotations(label)


getPlots("DICE_00_2000_MSE_00_2000_smooth_00_2000_nJD_00_0000_rigid_00_2000_trans_00_2000_weight_00_5000")