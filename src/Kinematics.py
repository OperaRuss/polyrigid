import SimpleITK as sitk
import numpy as np
import torch
import os
import glob
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
          '#a65628', '#984ea3', '#e41a1c', '#dede00',"#9aa0ab"]

comps = ['Scapohid', 'Lunate', 'Triquetral', 'Pisiform',
         'Hamate', 'Capitate', 'Trapezoid', 'Trapezium','Ground Truth']

compCoords = {1:(30,50,10),2:(30,35,10),3:(30,20,10),4:(30,5,10),
              5:(10,5,10),6:(10,20,10),7:(10,35,10),8:(10,50,10)}

axisColors = ['#f50000','#00f500','#0000f5','#000000']
axisLabels = ['Primary Axis','Secondary Axis', 'Tertiary Axis','Ground Truth']


def _getSortedEigens(vals,vecs):
    '''
    Assumes the values and vectors are returned as per np.linalg.eig(), ie. as columns. Assumes 3D.
    :param vals: N eigenvalues resulting from eigendecomposition
    :param vecs: N eigenvectors stored in a single array
    :return: sorted values and vectors (as ROWS!) where vectors and values match in ordering.
    '''
    vec0 = vecs[:,0]
    vec1 = vecs[:,1]
    vec2 = vecs[:,2]
    vectors = [vec0,vec1,vec2]
    return (list(t) for t in zip(*sorted(zip(vals,vectors),reverse=True)))


def _getPatches_Components(hasGroundTruth:bool=False):
    out = []
    if not hasGroundTruth:
        for i in range(0,8):
            out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    else:
        for i in range(0,9):
            out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    return out

def _getPatches_Axes():
    out = []
    for i in range(len(axisColors)):
        out.append(mpatch.Patch(color=axisColors[i],label=axisLabels[i]))
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
    patches = _getPatches_Components()

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
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .075, 0.64, .075), loc='lower left',
                      ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Coordinate Frames Fit to Pose Estimations", y=0.85,fontsize=24)
        ax.set_title("Frame "+vSource+" to Frame "+vTarget, fontsize=14,y=0.9)
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
    patches = _getPatches_Components(True)

    figErr, axErr = plt.subplots(figsize=(10,10))
    figErr.suptitle("Translation Error Between Prediction and Ground Truth",y=0.85)
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("Voxels by Euclidean Distance")
    axErr.yaxis.grid(True)
    figErr.tight_layout()
    aAllErrors = {}

    Pd = torch.arange(0, 60, dtype=torch.float32)
    Ph = torch.arange(0, 100, dtype=torch.float32)
    Pw = torch.arange(0, 100, dtype=torch.float32)
    P = torch.cartesian_prod(Pd, Ph, Pw)
    P = P.reshape(60, 100, 100, 3).detach().numpy()

    for file in glob.glob(os.path.join(fpIn_Preds,"frame_*_to_frame_*")):
        aFrameErrors = np.zeros(8)
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d',proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_ylim(45, 85)
        ax.set_xlim(10, 75)
        ax.set_zlim(0, 60)
        ax.set_zticks([])
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .07, 0.64, .07), loc='lower left',
                      ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Centroids for Ground Truth vs Predictions", y=0.85,fontsize=24)
        ax.set_title("Frame "+vSource+" to Frame "+vTarget, fontsize=14,y=0.9)
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

            ptCloud = np.where(np.stack((npImgPred,) * 3, axis=-1) >= 0.7, P, 0)
            ptCloud = np.ma.masked_equal(ptCloud, 0)
            ax.scatter(ptCloud[:, :, :, 1].flatten(),
                       ptCloud[:, :, :, 2].flatten(),
                       ptCloud[:, :, :, 0].flatten(),
                       color=colors[compColor],
                       alpha=0.01)

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
    figErr.clear()
    plt.close()


def _getModelAccuracy_Rotations(label):
    patches = _getPatches_Components(True)
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
    figErr.suptitle("Rotation Error Between Predictions and Ground Truth",y=0.85)
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("1 - cos(Theta) of Angle Between Primary Axes")
    axErr.yaxis.grid(True)
    figErr.tight_layout()

    aAllErrors = {}

    patches = _getPatches_Axes()

    for file in glob.glob(os.path.join(fpIn_Preds,"frame_*_to_frame_*")):
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]
        aFrameErrors = np.zeros(len(vDesiredComponents))

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-5, 45)
        ax.set_ylim(-5, 55)
        ax.set_zlim(0, 10)
        ax.legend(handles=patches, bbox_to_anchor=(0.2, .15, 0.64, .15), loc='lower left',
                  ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Rotation Accuracy for Warp Field", y=0.85,fontsize=24)
        ax.set_title("Frame " + vSource + " to Frame " + vTarget, fontsize=14,y=0.9)
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

            coord = compCoords[comp]
            ax.quiver(*coord, *predEigenvectors[0], color='r', length=predEigenvalues[0] * 0.2)
            ax.quiver(*coord, *predEigenvectors[1], color='g', length=predEigenvalues[1] * 0.2)
            ax.quiver(*coord, *predEigenvectors[2], color='b', length=predEigenvalues[2] * 0.2)
            ax.quiver(*coord, *truthEigenvectors[0], color='k', length=truthEigenvalues[0] * 0.2)
            ax.quiver(*coord, *truthEigenvectors[1], color='k', length=truthEigenvalues[1] * 0.2)
            ax.quiver(*coord, *truthEigenvectors[2], color='k', length=truthEigenvalues[2] * 0.2)
        ax.text(0, 3, 10, "Hamate", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(0, 17, 10, "Capitate", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(0, 33, 10, "Trapezoid", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(0, 46, 10, "Trapezium", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(43, 3, 10, "Pisiform", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(43, 17, 10, "Triquetral", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(43, 33, 10, "Lunate", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        ax.text(43, 46, 10, "Scaphoid", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})
        aAllErrors[vTarget] = aFrameErrors
        fig.savefig(os.path.join(fpOut, os.path.basename(file)), bbox_inches='tight')
        fig.clear()
        plt.close()
    frames, errors = [*zip(*aAllErrors.items())]
    axErr.boxplot(errors, patch_artist=True, boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'))
    figErr.savefig(os.path.join(fpOut, "summaryErrorRotation"), bbox_inches='tight')
    figErr.clear()
    plt.close()


def _getNeutralSegmentationDescriptors():
    fpGroundTruths = "../images/input/rave"
    componentLabelImages = {}
    numComponents = 8

    for comp in range(1,numComponents+1):
        imgComp = sitk.ReadImage(os.path.join(fpGroundTruths,"em_frame_10_seg_"+str(comp)+".nii.gz"))
        npComp = sitk.GetArrayFromImage(imgComp)[10:70, 60:160, 50:150]
        componentLabelImages[str(comp)] = npComp

    componentCentroids = np.zeros((numComponents,3),dtype=np.float32)
    componentAxes = np.zeros((numComponents,3,3),dtype=np.float32)
    componentAxisLengths = np.zeros((numComponents,3))
    for k,v in sorted(componentLabelImages.items()):
        componentCentroids[int(k)-1] = skimage.measure.centroid(v)
        compInertiaTensor = skimage.measure.inertia_tensor(v)
        eigvals, eigvecs = np.linalg.eig(compInertiaTensor)
        eigvecs[:, 0] = eigvecs[:, 0]/np.linalg.norm(eigvecs[:, 0])
        eigvecs[:, 1] = eigvecs[:, 1]/np.linalg.norm(eigvecs[:, 1])
        eigvecs[:, 2] = eigvecs[:, 2]/np.linalg.norm(eigvecs[:, 2])
        eigvals, eigvecs = _getSortedEigens(eigvals,eigvecs)
        componentAxes[int(k)-1] = eigvecs
        componentAxisLengths[int(k)-1] = eigvals

    return {'centroids': componentCentroids,
            'axes': componentAxes,
            'axislengths': componentAxisLengths,
            'segmentations': componentLabelImages}


def _getComposedComponentTransforms(label):
    fpResults = os.path.join("../images/results/",label)

    stepwiseTransformations = {}
    composedTransformations = {}

    for file in glob.glob(os.path.join(fpResults,"frame_*_to_frame_*")):
        vSource = os.path.basename(file).split('_')[1]
        vTarget = file.split('_')[-1]
        npTransformsEuclid = np.load(os.path.join(file,
                                                  "component_transforms_final_euclidean.npz"))

        frameTransforms = {}
        for k,v in sorted(npTransformsEuclid.items()):
            frameTransforms[k] = v

        stepwiseTransformations[vTarget] = frameTransforms

    #Initialize at identity for frame 10
    frameTransforms = {}
    for i in range(1,9):
        frameTransforms[str(i)] = np.eye(4)
    stepwiseTransformations['10'] = frameTransforms
    composedTransformations['10'] = frameTransforms

    # Compose Transformations by Left-side Dot Product
    for frame in range(11,20):
        frameTransforms = {}
        for comp in range(1,9):
            frameTransforms[str(comp)] = np.dot(stepwiseTransformations[str(frame)][str(comp)],
                                                stepwiseTransformations[str(frame-1)][str(comp)])
        composedTransformations[str(frame)] = frameTransforms
    for frame in range(9,-1,-1):
        frameTransforms = {}
        for comp in range(1,9):
            frameTransforms[str(comp)] = np.dot(stepwiseTransformations[str(frame)][str(comp)],
                                                stepwiseTransformations[str(frame+1)][str(comp)])
        composedTransformations[str(frame)] = frameTransforms
    return composedTransformations


def _getNearestNeighbor(point):
    out = []
    for p in point:
        out = np.round(p).astype(float)
    return out


def _getComponentTransformAccuracy_Translation(label):
    colorGray = "#9aa0ab"
    patches = _getPatches_Components(True)
    vElevation = 90
    vAzim = 0

    fpGroundTruths = "../images/input/rave"
    vTruthPrefix = "em_frame_"

    fpOut = os.path.join("../images/results", label, label, "plots", "modelAccuracy_EstimatedTranslations")

    if not os.path.exists(fpOut):
        os.mkdir(fpOut)

    vDesiredComponents = range(1, 9)

    baseSegDescriptors = _getNeutralSegmentationDescriptors()
    composedTransforms = _getComposedComponentTransforms(label)

    figErr, axErr = plt.subplots(figsize=(10, 10))
    figErr.suptitle("Translation Error Between Estimated Transforms and Ground Truth",y=0.85)
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("Voxels by Euclidean Distance")
    axErr.yaxis.grid(True)
    figErr.tight_layout()

    aAllErrors = {}

    # Create a list of coordinates
    Pd = torch.arange(0, 60, dtype=torch.float32)
    Ph = torch.arange(0, 100, dtype=torch.float32)
    Pw = torch.arange(0, 100, dtype=torch.float32)
    P = torch.cartesian_prod(Pd, Ph, Pw)
    P1 = torch.ones(60*100*100)
    P = torch.cat((P,P1.unsqueeze(-1)),dim=1)
    P = P.reshape(60, 100, 100, 4).detach().numpy()

    for targetFrame in range(0,20):
        if targetFrame < 10:
            vSource = str(targetFrame+1)
        elif targetFrame > 10:
            vSource = str(targetFrame-1)
        else:
            vSource = '10'
        vTarget = str(targetFrame)

        aFrameErrors = np.zeros(len(vDesiredComponents))

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_ylim(45, 85)
        ax.set_xlim(10, 75)
        ax.set_zlim(0, 60)
        ax.set_zticks([])
        ax.legend(handles=patches, bbox_to_anchor=(0.2, 0.07, 0.64, 0.07), loc='lower left',
                  ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Centroids for Ground Truth vs Estimated Transforms", y=0.85, fontsize=24)
        ax.set_title("Frame " + vSource + " to Frame " + vTarget, fontsize=14,y=0.9)
        fig.tight_layout()

        for comp in vDesiredComponents:
            compColor = comp-1

            # Read in the component warped label image
            imgTruth = sitk.ReadImage(os.path.join(fpGroundTruths,vTruthPrefix + str(vTarget)
                                                   + '_seg_' + str(comp) + '.nii.gz'))
            npImgTruth = sitk.GetArrayFromImage(imgTruth)[10:70, 60:160, 50:150] # crop taken from main
                                                                                 # Should probably be a utility func.
            trans = composedTransforms[str(targetFrame)][str(comp)].T
            ptCloud = np.einsum('ij,...i',trans,P)
            ptCloud = (ptCloud/(ptCloud[:,:,:,3,None] + 1e-7))[:,:,:,:3]
            ptCloud = np.where(np.stack((baseSegDescriptors['segmentations'][str(comp)],) * 3, axis=-1) >= 0.7, ptCloud, 0)
            ptCloud = np.ma.masked_equal(ptCloud, 0)
            ax.scatter(ptCloud[:, :, :, 1].flatten(),
                       ptCloud[:, :, :, 2].flatten(),
                       ptCloud[:, :, :, 0].flatten(),
                       color=colors[compColor],
                       alpha=0.01)

            predCentroid = baseSegDescriptors['centroids'][comp-1]
            predCentroid_homo = np.concatenate((predCentroid,[1.]))
            predCentroid_trans = np.dot(composedTransforms[str(targetFrame)][str(comp)],predCentroid_homo)
            predCentroid = np.divide(predCentroid_trans,predCentroid_trans[-1])[:-1]
            truthCentroid = skimage.measure.centroid(npImgTruth)
            errorVector_Centroid = truthCentroid - predCentroid
            aFrameErrors[comp - 1] = np.linalg.norm(errorVector_Centroid, ord=2)

            ax.scatter(truthCentroid[1], truthCentroid[2], truthCentroid[0],
                       s=200, color=colorGray, alpha=0.5, edgecolors='black')
            ax.scatter(predCentroid[1], predCentroid[2], predCentroid[0],
                       s=200, color=colors[compColor], alpha=0.5, edgecolors='black')
            ax.quiver(predCentroid[1], predCentroid[2], predCentroid[0],
                      errorVector_Centroid[1], errorVector_Centroid[2], errorVector_Centroid[0],
                      color=colorGray, length=np.linalg.norm(errorVector_Centroid) * 0.2)
        aAllErrors[vTarget] = aFrameErrors
        fig.savefig(os.path.join(fpOut, "frame_"+vSource+"_to_frame_"+vTarget), bbox_inches='tight')
        fig.clear()
        plt.close()
    frames, errors = [*zip(*aAllErrors.items())]
    axErr.boxplot(errors, patch_artist=True, boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'))
    figErr.savefig(os.path.join(fpOut, "summaryErrorEstimatedTranslation"), bbox_inches='tight')
    figErr.clear()
    plt.close()

def _getComponentTransformAccuracy_Rotation(label):
    colorGray = "#9aa0ab"
    patches = _getPatches_Components(True)
    vElevation = 90
    vAzim = 0

    fpGroundTruths = "../images/input/rave"
    vTruthPrefix = "em_frame_"

    fpOut = os.path.join("../images/results", label, label, "plots", "modelAccuracy_EstimatedRotations")

    if not os.path.exists(fpOut):
        os.mkdir(fpOut)

    vDesiredComponents = range(1, 9)

    baseSegDescriptors = _getNeutralSegmentationDescriptors()
    composedTransforms = _getComposedComponentTransforms(label)

    figErr, axErr = plt.subplots(figsize=(10, 10))
    figErr.suptitle("Rotation Error Between Predictions and Ground Truth",y=0.85)
    axErr.set_xlabel("Target Frame")
    axErr.set_ylabel("1 - cos(Theta) of Angle Between Primary Axes")
    axErr.yaxis.grid(True)
    figErr.tight_layout()

    patches = _getPatches_Axes()

    aAllErrors = {}

    for targetFrame in range(0, 20):
        if targetFrame < 10:
            vSource = str(targetFrame + 1)
        elif targetFrame > 10:
            vSource = str(targetFrame - 1)
        else:
            vSource = '10'
        vTarget = str(targetFrame)

        aFrameErrors = np.zeros(len(vDesiredComponents))

        # Set up plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', proj_type='ortho')
        ax.view_init(elev=vElevation, azim=vAzim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-5, 45)
        ax.set_ylim(-5,55)
        ax.set_zlim(0, 10)
        ax.legend(handles=patches, bbox_to_anchor=(0.2, 0.15, 0.64, 0.15), loc='lower left',
                  ncol=4, mode="expand", borderaxespad=0.)
        fig.suptitle("Rotation Accuracy of Estimated Transforms", y=0.85, fontsize=24)
        ax.set_title("Frame " + vSource + " to Frame " + vTarget, fontsize=14,y=0.9)
        fig.tight_layout()

        for comp in vDesiredComponents:
            compColor = comp - 1

            # Read in the component warped label image
            imgTruth = sitk.ReadImage(os.path.join(fpGroundTruths, vTruthPrefix + str(vTarget)
                                                   + '_seg_' + str(comp) + '.nii.gz'))
            npImgTruth = sitk.GetArrayFromImage(imgTruth)[10:70, 60:160, 50:150]  # crop taken from main

            predFrame = baseSegDescriptors['axes'][comp-1]
            predFrame0_homo = np.concatenate((predFrame[0],[1.]))
            predFrame1_homo = np.concatenate((predFrame[1],[1.]))
            predFrame2_homo = np.concatenate((predFrame[2],[1.]))
            predFrame0_trans = np.dot(composedTransforms[str(targetFrame)][str(comp)],predFrame0_homo)
            predFrame1_trans = np.dot(composedTransforms[str(targetFrame)][str(comp)],predFrame1_homo)
            predFrame2_trans = np.dot(composedTransforms[str(targetFrame)][str(comp)],predFrame2_homo)
            predFrame0 = np.divide(predFrame0_trans,predFrame0_trans[-1])[:-1]
            predFrame1 = np.divide(predFrame1_trans, predFrame1_trans[-1])[:-1]
            predFrame2 = np.divide(predFrame2_trans, predFrame2_trans[-1])[:-1]

            predLengths = baseSegDescriptors['axislengths'][comp-1]

            truthFrame = skimage.measure.inertia_tensor(npImgTruth)
            truthEigenvalues, truthEigenvectors = np.linalg.eig(truthFrame)
            truthEigenvalues, truthEigenvectors = _getSortedEigens(truthEigenvalues, truthEigenvectors)

            angle = np.divide(np.dot(predFrame0, truthEigenvectors[0]),
                              np.linalg.norm(predFrame0) * np.linalg.norm(truthEigenvectors[0]))

            aFrameErrors[comp - 1] = 1 - abs(angle)
            coord = compCoords[comp]

            ax.quiver(*coord, *predFrame0, color='r',length=10)
            ax.quiver(*coord, *predFrame1, color='g',length=10)
            ax.quiver(*coord, *predFrame2, color='b',length=10)
            ax.quiver(*coord, *truthEigenvectors[0], color='k',length=10)
            ax.quiver(*coord, *truthEigenvectors[1], color='k',length=10)
            ax.quiver(*coord, *truthEigenvectors[2], color='k',length=10)
        ax.text(0,3,10,"Hamate",fontfamily='sans-serif',fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(0, 17, 10, "Capitate", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(0, 33, 10, "Trapezoid", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(0, 46, 10, "Trapezium", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(43, 3, 10, "Pisiform", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(43, 17, 10, "Triquetral", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(43, 33, 10, "Lunate", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        ax.text(43, 46, 10, "Scaphoid", fontfamily='sans-serif', fontstyle='italic',
                bbox={'facecolor': 'gray','alpha':0.5,'pad':5})
        aAllErrors[vTarget] = aFrameErrors
        fig.savefig(os.path.join(fpOut, "frame_" + vSource + "_to_frame_" + vTarget), bbox_inches='tight')
        fig.clear()
        plt.close()
    frames, errors = [*zip(*aAllErrors.items())]
    axErr.boxplot(errors, patch_artist=True, boxprops=dict(facecolor='lightgray'), medianprops=dict(color='k'))
    figErr.savefig(os.path.join(fpOut, "summaryErrorEstimatedRotation"), bbox_inches='tight')
    figErr.clear()
    plt.close()


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
    _getWarpedComponentPlots(label)
    _getModelAccuracy_Centroids(label)
    _getModelAccuracy_Rotations(label)
    _getComponentTransformAccuracy_Translation(label)
    _getComponentTransformAccuracy_Rotation(label)

#getPlots("DICE_00_2000_MSE_00_2000_smooth_00_2000_nJD_00_0000_rigid_00_2000_trans_00_2000_weight_00_5000")