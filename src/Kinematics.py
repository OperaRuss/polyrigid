import SimpleITK as sitk
import numpy as np
import torch
import os
import glob
import skimage.measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

def _getPatches():
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
              '#a65628', '#984ea3', '#e41a1c', '#dede00']

    comps = ['Scapohid', 'Lunate', 'Triquetral', 'Pisiform',
             'Hamate', 'Capitate', 'Trapezoid', 'Trapezium']

    out = []
    for i in range(0,8):
        out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    return out

def getWarpedComponentPlots(label: str):
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
              '#a65628', '#984ea3', '#e41a1c', '#dede00']

    comps = ['Scapohid', 'Lunate', 'Triquetral', 'Pisiform',
             'Hamate', 'Capitate', 'Trapezoid', 'Trapezium']
    
    vAlpha = 0.05
    vElevation = 90
    vAzim = 0

    fpResults = "../images/results"
    fpIn = os.path.join(fpResults,label)
    fpOut = os.path.join(fpIn,label,"estimations")
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
        ax = fig.add_subplot(projection='3d')
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
            frame1 = Q[:,0]/np.linalg.norm(Q[:,0])
            frame2 = Q[:,1]/np.linalg.norm(Q[:,1])
            frame3 = Q[:,2]/np.linalg.norm(Q[:,2])
            ax.scatter(centroid[1],centroid[2],centroid[0],
                       s=100,color=colors[compColor])

            ax.quiver(centroid[1],centroid[2],centroid[0],*frame1,color='r',length=lam[0]*0.2)
            ax.quiver(centroid[1],centroid[2],centroid[0],*frame2,color='g',length=lam[1]*0.2)
            ax.quiver(centroid[1],centroid[2],centroid[0],*frame3,color='b',length=lam[2]*0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame1), color='r', length=lam[0] * 0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame2), color='g', length=lam[1] * 0.2)
            ax.quiver(centroid[1], centroid[2], centroid[0], *(-1*frame3), color='b', length=lam[2] * 0.2)

        fig.savefig(os.path.join(fpOut,os.path.basename(file)),bbox_inches='tight')
        fig.clear()
        plt.close()


def getCentroidCheck(label):
    pass


def getPlots(label):
    '''
    Driver function for Kinematic plotting functions.
    :param label: model label as stored in results folder.
    :return: None.
    '''
    getWarpedComponentPlots(label)


getPlots("DICE_00_2000_MSE_00_2000_smooth_00_2000_nJD_00_0000_rigid_00_2000_trans_00_2000_weight_00_5000")