import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import SimpleITK as sitk
import vtk

# Colorblind friendly color palette
# Taken from https://gist.github.com/thriveth/8560036
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#e41a1c', '#dede00']

comps = ['Scapohid','Lunate','Triquetral','Pisiform',
         'Hamate','Capitate','Trapezoid','Trapezium']

def _getPatches():
    out = []
    for i in range(0,8):
        out.append(mpatch.Patch(color=colors[i],label=comps[i]))
    return out

def getEvaluationPlots(params:str=None):
    plt.close('all')

    fpResults = os.path.join("../images/results",params)
    fpOutPath = os.path.join(fpResults,params)

    if not os.path.exists(fpOutPath):
        os.mkdir(fpOutPath)

    fig, ax = plt.subplots(1,1)
    for file in sorted(glob.glob(fpResults+"/frame_*_to_frame_*")):
        if os.path.isdir(file):
            loss = np.load(os.path.join(file, 'model_loss.npz'))
            temp = {}
            for k,v in loss.items():
                temp[int(k)] = v

            x,y = zip(*sorted(temp.items()))
            ax.plot(x,y,color='k')
    ax.set_title("MSE Loss for All Frames in Dynamic Sequence")
    ax.set_ylabel("MSE Loss")
    ax.set_xlabel("Iteration")
    plt.xticks(np.arange(0,len(y)+1,20))
    if params is not None:
        plt.savefig(os.path.join(fpOutPath,"summary_MSE.png"),bbox_inches='tight')
    else:
        plt.savefig("../images/results/summary_MSE.png",bbox_inches='tight')

    finLoss = {}
    netDICE = {}
    percNegJDets = {}
    meanRigid = {}

    figNJD,axNJD = plt.subplots(1)
    axNJD.set(title="Percentage of Negative Jacobian Determinants")
    axNJD.set_ylabel('% Negative Determinants')
    axNJD.set_xlabel('Target Frame')
    axNJD.axvline(10, linestyle='--', color='b', alpha=0.5)
    axNJD.set_xticks(np.arange(0, 21, 2))

    figDICE,axDICE = plt.subplots(1)
    axDICE.set(title="Net Gain in DICE Score")
    axDICE.set_ylabel('Change in DICE Score')
    axDICE.set_xlabel('Target Frame')
    axDICE.axvline(10, linestyle='--', color='b', alpha=0.5)
    axDICE.set_xticks(np.arange(0, 21, 2))

    figLoss, axLoss = plt.subplots(1)
    axLoss.set(title="Final Loss Score Achieved")
    axLoss.set_ylabel('Regularized Loss Score')
    axLoss.set_xlabel('Target Frame')
    axLoss.axvline(10, linestyle='--', color='b', alpha=0.5)
    axLoss.set_xticks(np.arange(0, 21, 2))

    figRigid, axRigid = plt.subplots(1)
    axRigid.set(title="Mean Rigidity of Estimated Transformations")
    axRigid.set_ylabel('Rigidity Score')
    axRigid.set_xlabel('Target Frame')
    axRigid.axvline(10, linestyle='--', color='b', alpha=0.5)
    axRigid.set_xticks(np.arange(0, 21, 2))

    fig, axs = plt.subplots(2,2,figsize=(10,10)) # in clockwise from upper left: numNeg, netDICE, percentageNegJDets,meanRigidityScore
    for file in sorted(glob.glob(fpResults+"/frame_*_to_frame_*")):
        if os.path.isdir(file):
            results = np.load(os.path.join(file, 'model_results.npz'))
            label = int(results['target'])
            finLoss[label] = results['finalLoss']
            netDICE[label] = results['netDICE']
            percNegJDets[label] = results['percentageNegJDets']
            meanRigid[label] = results['meanRigidityScore']

    x,y = zip(*sorted(finLoss.items()))
    axLoss.plot(x, y, '-ok')
    axs[0,0].plot(x,y,'-ok')
    axs[0,0].set(title="Final Loss Score Achieved")
    axs[0,0].axhline(np.mean(y),linestyle='--',color='r')
    axs[0,0].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(netDICE.items()))
    axDICE.plot(x,y, '-ok')
    axs[0,1].plot(x,y,'-ok')
    axs[0,1].set(title="Net DICE Gain")
    axs[0,1].axhline(np.mean(y),linestyle='--',color='r')
    axs[0,1].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(percNegJDets.items()))
    axNJD.plot(x,y,'-ok')
    axs[1,0].plot(x,y,'-ok')
    axs[1,0].set(title="% Neg JDets")
    axs[1,0].axhline(np.mean(y),linestyle='--',color='r')
    axs[1,0].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(meanRigid.items()))
    axRigid.plot(x,y,'-ok')
    axs[1,1].plot(x,y,'-ok')
    axs[1,1].set(title="Mean Rigidity Score")
    axs[1,1].axhline(np.mean(y),linestyle='--',color='r')
    axs[1,1].set_xticks(np.arange(0,21,2))

    fig.suptitle("Final Evaluation Scores After Registration on All Frames")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if params is not None:

        fig.savefig(os.path.join(fpOutPath,"summary_metrics.png"), bbox_inches='tight')
        figNJD.savefig(os.path.join(fpOutPath,"summaryNJD.png"), bbox_inches='tight')
        figDICE.savefig(os.path.join(fpOutPath,"summaryDICE.png"), bbox_inches='tight')
        figLoss.savefig(os.path.join(fpOutPath,"summaryLoss.png"), bbox_inches='tight')
        figRigid.savefig(os.path.join(fpOutPath,"summaryRigid.png"), bbox_inches='tight')
    else:
        fig.savefig("../images/results/summary_metrics.png", bbox_inches='tight')
        figNJD.savefig("../images/results/summaryNJD.png", bbox_inches='tight')
        figDICE.savefig("../images/results/summaryDICE.png", bbox_inches='tight')
        figLoss.savefig("../images/results/summaryLoss.png", bbox_inches='tight')
        figRigid.savefig("../images/results/summaryRigid.png", bbox_inches='tight')

    sequenceConfusion(fpOutPath,params)
    getBoneAccuracyPlots(params)
    displayFrames(params)

def confusionMatrix(targetSeg, predictionSeg, outPath,label,prefix:str=""):
    imgTP = np.where((targetSeg==predictionSeg),targetSeg,0)
    imgFP = np.where((predictionSeg!=targetSeg),predictionSeg,0)
    imgFN = np.where((predictionSeg!=targetSeg),targetSeg,0)
    imgTN = np.multiply(np.subtract(np.max(targetSeg),targetSeg),
                                    np.subtract(np.max(predictionSeg),predictionSeg))
    imgIoU = targetSeg + predictionSeg - imgTP

    sitk.WriteImage(sitk.GetImageFromArray(imgTP,False),outPath+'/img'+prefix+'TP.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgFP,False),outPath+'/img'+prefix+'FP.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgFN,False),outPath+'/img'+prefix+'FN.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgTN,False),outPath+'/img'+prefix+'TN.nii')

    TP = np.count_nonzero(imgTP)
    FP = np.count_nonzero(imgFP)
    FN = np.count_nonzero(imgFN)
    TN = np.count_nonzero(imgTN)
    IoU = np.count_nonzero(imgIoU)

    out = {'target':label,'TP':TP,'FP':FP,'FN':FN,'TN':TN,'DICE':2*(TP/IoU+1e-7),
           'accuracy':(TP+TN)/(TP+FP+FN+TN+1e-7)*100,'precision':TP/(TP+FP+1e-7)*100,
           'recall':TP/(TP+FN)*100}

    return out

def sequenceConfusion(outPath,params:str=""):
    fpResults = os.path.join("../images/results",params) # inPath

    aDice = {}
    aPrecision = {}
    aRecall = {}
    aAccuracy = {}

    for file in sorted(glob.glob(fpResults + "/frame_*_to_frame_*")):
        if os.path.isdir(file):
            cfsn = np.load(os.path.join(file, 'confusionMatrix.npz'))
            label = int(cfsn['target'])
            aDice[label] = cfsn['DICE']
            aPrecision[label] = cfsn['precision']
            aRecall[label] = cfsn['recall']
            aAccuracy[label] = cfsn['accuracy']

    dice, diceAx = plt.subplots(1)
    x, y = zip(*sorted(aDice.items()))
    diceAx.plot(x, y, '-ok')
    diceAx.set(title="DICE Scores Across Sequence")
    diceAx.set_ylabel('DICE Score')
    diceAx.set_xlabel('Target Frame')
    diceAx.axvline(10,linestyle='--',color='b',alpha=0.5)
    diceAx.set_xticks(np.arange(0, 21, 2))
    dice.savefig(outPath+'/summaryDICE.png',bbox_inches='tight')

    precision, precAx = plt.subplots(1)
    x, y = zip(*sorted(aPrecision.items()))
    precAx.plot(x, y, '-ok')
    precAx.set(title="Precision Across Sequence")
    precAx.set_ylabel('Precision (%)')
    precAx.set_xlabel('Target Frame')
    precAx.axvline(10,linestyle='--',color='b',alpha=0.5)
    precAx.set_xticks(np.arange(0, 21, 2))
    precision.savefig(outPath+'/summaryPrecision.png',bbox_inches='tight')

    recall, recallAx = plt.subplots(1)
    x, y = zip(*sorted(aRecall.items()))
    recallAx.plot(x, y, '-ok')
    recallAx.set(title="Recall Across Sequence")
    recallAx.set_ylabel('Recall (%)')
    recallAx.set_xlabel('Target Frame')
    recallAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    recallAx.set_xticks(np.arange(0, 21, 2))
    recall.savefig(outPath + '/summaryRecall.png', bbox_inches='tight')

    accuracy, accAx = plt.subplots(1)
    x, y = zip(*sorted(aAccuracy.items()))
    accAx.plot(x, y, '-ok')
    accAx.set(title="Accuracy Across Sequence")
    accAx.set_ylabel('Accuracy (%)')
    accAx.set_xlabel('Target Frame')
    accAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    accAx.set_xticks(np.arange(0, 21, 2))
    accuracy.savefig(outPath + '/summaryAccuracy.png', bbox_inches='tight')

def getBoneAccuracyPlots(params:str=""):
    plt.close('all')

    fpResults = os.path.join("../images/results",params) #inPath
    fpOut = os.path.join(fpResults,params)
    patches = _getPatches()

    dice, diceAx = plt.subplots(1)
    diceAx.set(title="DICE by Component Across Sequence")
    diceAx.set_ylabel('DICE Score')
    diceAx.set_xlabel('Target Frame')
    diceAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    diceAx.legend(handles=patches,bbox_to_anchor=(1.05, 1),
                  loc='upper left', borderaxespad=0.)
    diceAx.set_xticks(np.arange(0, 21, 2))

    prec, precAx = plt.subplots(1)
    precAx.set(title="Precision by Component Across Sequence")
    precAx.set_ylabel('Precision (%)')
    precAx.set_xlabel('Target Frame')
    precAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    precAx.legend(handles=patches)
    precAx.set_xticks(np.arange(0, 21, 2))

    rec, recAx = plt.subplots(1)
    recAx.set(title="Recall by Component Across Sequence")
    recAx.set_ylabel('Recall (%)')
    recAx.set_xlabel('Target Frame')
    recAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    recAx.legend(handles=patches)
    recAx.set_xticks(np.arange(0, 21, 2))

    acc, accAx = plt.subplots(1)
    accAx.set(title="Accuracy by Component Across Sequence")
    accAx.set_ylabel('Accuracy (%)')
    accAx.set_xlabel('Target Frame')
    accAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    accAx.legend(handles=patches)
    accAx.set_xticks(np.arange(0, 21, 2))

    for comp in range(1,9):
        aDice = {}
        aPrecision = {}
        aRecall = {}
        aAccuracy = {}

        for file in sorted(glob.glob(fpResults + "/frame_*_to_frame_*")):
            if os.path.isdir(file):
                cfsn = np.load(os.path.join(file, "component_"+str(comp)+"_cfsn.npz"))
                label = int(cfsn['target'])
                aDice[label] = cfsn['DICE']
                aPrecision[label] = cfsn['precision']
                aRecall[label] = cfsn['recall']
                aAccuracy[label] = cfsn['accuracy']

        x, y = zip(*sorted(aDice.items()))
        diceAx.plot(x, y, ls='-',c=colors[comp-1])

        x, y = zip(*sorted(aPrecision.items()))
        precAx.plot(x, y, ls='-',c=colors[comp-1])

        x, y = zip(*sorted(aRecall.items()))
        recAx.plot(x, y, ls='-',c=colors[comp-1])

        x, y = zip(*sorted(aAccuracy.items()))
        accAx.plot(x, y, ls='-',c=colors[comp-1])

    acc.savefig(os.path.join(fpOut,'summaryComponents_Accuracy.png'), bbox_inches='tight')
    dice.savefig(os.path.join(fpOut,'summaryComponents_DICE.png'), bbox_inches='tight')
    prec.savefig(os.path.join(fpOut,'summaryComponents_Precision.png'), bbox_inches='tight')
    rec.savefig(os.path.join(fpOut,'summaryComponents_Recall.png'), bbox_inches='tight')

def _getStaticMeshes(fpInPath, targetFrame: int,
                     filePrefix:str='warped_seg_', fpOutPath:str='../images/results/meshes'):
    '''
    This script was modeled after an example provided by Dr. James Fishbaugh. It
    takes in an input file path, reads all .nii files, converts them to static meshes
    with the marching cubes algorithm, and outputs them to the out path.
    :param fpInPath: Filepath to a folder containing .nii binary labels
    :param filePrefix: Prefix for all label images of interest
    :param fpOutPath: Filepath to output directory
    :return:
    '''

    for file in sorted(glob.glob(fpInPath+"/"+filePrefix+"*.nii")):
        basename, niiExt = os.path.splitext(os.path.basename(file))

        print("Processing "+basename+niiExt+'.')
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(file)
        reader.Update()
        img = reader.GetOutput()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img)
        mc.ComputeNormalsOff()
        mc.ComputeGradientsOff()
        mc.SetComputeScalars(False)
        mc.SetValue(0,1)
        mc.Update()

        confilter = vtk.vtkPolyDataConnectivityFilter()
        confilter.SetInputData(mc.GetOutput())
        confilter.SetExtractionModeToLargestRegion()
        confilter.Update()

        poly = confilter.GetOutput()

        outFile = f"frame_{targetFrame:02}.vtk"
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(poly)
        writer.SetFileName(os.path.join(fpOutPath,basename,outFile))
        writer.Update()

def getStaticMeshes(fpResults:str='../images/results',params:str="",filePrefix:str='warped_seg_'):
    fpIn = os.path.join(fpResults,params)
    fpOut = os.path.join(fpResults,params,params,'meshes')
    if not os.path.exists(fpOut):
        os.mkdir(fpOut)

    for i in range(1,15):
        if not os.path.exists(os.path.join(fpOut,'warped_seg_'+str(i))):
            os.mkdir(os.path.join(fpOut,'warped_seg_'+str(i)))

    if not os.path.exists(os.path.join(fpOut,'warped_seg_binary')):
        os.mkdir(os.path.join(fpOut,'warped_seg_binary'))

    for file in sorted(glob.glob(fpIn + "/frame_*_to_frame_*")):
        target = int(file.split('_')[-1])
        if os.path.isdir(file):
            _getStaticMeshes(fpInPath=file,targetFrame=target,
                             filePrefix=filePrefix,fpOutPath=fpOut)

def displayFrames(params:str=""):
    plt.close('all')

    fpResults = os.path.join("../images/results",params) # inDir
    fpOutDir = os.path.join(fpResults,params)
    fpSubDir_Ref = os.path.join(fpOutDir,'imgRef')
    fpSubDir_Warp = os.path.join(fpOutDir,'imgWarped')
    fpSubDir_Float = os.path.join(fpOutDir,'imgFloat')
    fpSubDir_WarpOverRef = os.path.join(fpOutDir,'imgWarpOverRef')
    fpSubDir_RefOverRef = os.path.join(fpOutDir,'imgRefOverRef')
    fpSubDir_FloatOverRef = os.path.join(fpOutDir,'imgFloatOverRef')
    fpSubDir_Errors = os.path.join(fpOutDir,'imgErrors')

    aSubDirs = [fpSubDir_Errors,fpSubDir_FloatOverRef,fpSubDir_RefOverRef,
                fpSubDir_WarpOverRef,fpSubDir_Warp,fpSubDir_Ref,fpSubDir_Float]

    slice = 30

    if not os.path.exists(fpOutDir):
        os.mkdir(fpOutDir)

    for subdir in aSubDirs:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    for file in sorted(glob.glob(fpResults + "/frame_*_to_frame_*")):
        if os.path.isdir(file):
            fTarget = int(os.path.basename(file).split('_')[-1])

            imgRef = sitk.ReadImage(os.path.join(file,'imgReference.nii'))
            imgRef = sitk.GetArrayFromImage(imgRef)
            imgRef = np.stack((imgRef,)*3,axis=-1)

            imgWarp = sitk.ReadImage(os.path.join(file,'imgWarped.nii'))
            imgWarp = sitk.GetArrayFromImage(imgWarp)

            imgFloat = sitk.ReadImage(os.path.join(file,'imgFloat.nii'))
            imgFloat = sitk.GetArrayFromImage(imgFloat)

            labWarped = sitk.ReadImage(os.path.join(file,'warped_seg_binary.nii'))
            labWarped = sitk.GetArrayFromImage(labWarped)

            labRef = sitk.ReadImage(os.path.join(file,'target_seg_binary.nii'))
            labRef = sitk.GetArrayFromImage(labRef)

            labFloat = sitk.ReadImage(os.path.join(file,'float_seg_binary.nii'))
            labFloat = sitk.GetArrayFromImage(labFloat)

            labFN = sitk.ReadImage(os.path.join(file,'imgFN.nii'))
            labFN = sitk.GetArrayFromImage(labFN)

            labFP = sitk.ReadImage(os.path.join(file,'imgFP.nii'))
            labFP = sitk.GetArrayFromImage(labFP)

            labTP = sitk.ReadImage(os.path.join(file, 'imgTP.nii'))
            labTP = sitk.GetArrayFromImage(labTP)

            def saveImageBW(imSlice, fpOut, saveName):
                plt.imshow(imSlice,cmap='gray',interpolation='bilinear')
                fig = plt.gcf()
                ax = plt.gca()
                fig.set_size_inches(6,6)
                ax.set_axis_off()
                fig.savefig(os.path.join(fpOut, saveName + f'_{fTarget:02}.png'),
                            bbox_inches='tight',dpi=500)
                fig.clear()
                plt.close()

            saveImageBW(imgWarp[slice,:,:],fpSubDir_Warp,'imgWarped')
            saveImageBW(imgRef[slice,:,:],fpSubDir_Ref,'imgRef')
            saveImageBW(imgFloat[slice,:,:],fpSubDir_Float,'imgFloat')

            def saveImageColor(imSlice, fpOut, saveName):
                plt.imshow(imSlice, interpolation='bilinear')
                fig = plt.gcf()
                ax = plt.gca()
                ax.set_axis_off()
                fig.set_size_inches(6, 6)
                fig.savefig(os.path.join(fpOut, saveName + f'_{fTarget:02}.png'),
                            bbox_inches='tight',dpi=500)
                fig.clear()
                plt.close()

            imgRefWithWarpedSegs = np.copy(imgRef)
            imgRefWithWarpedSegs[:,:,:,0] = np.where(labWarped >= 0.7,labWarped,imgRef[:,:,:,0])
            saveImageColor(imgRefWithWarpedSegs[slice,:,:],fpSubDir_WarpOverRef,'imgWarpedOverRef')

            imgRefWithRefSegs = np.copy(imgRef)
            imgRefWithRefSegs[:, :, :, 0] = np.where(labRef >= 0.7, labRef, imgRef[:,:,:,0])
            saveImageColor(imgRefWithRefSegs[slice,:,:],fpSubDir_RefOverRef,'imgRefOverRef')

            imgRefWithFloatSegs = np.copy(imgRef)
            imgRefWithFloatSegs[:, :, :, 0] = np.where(labFloat >= 0.7, labFloat, imgRef[:,:,:,0])
            saveImageColor(imgRefWithFloatSegs[slice,:,:],fpSubDir_FloatOverRef,'imgFloatOverRef')

            imgRefWithErrorSegs = np.copy(imgRef)
            imgRefWithErrorSegs[:,:,:,0] = np.where(labFN >= 0.7, labFN, imgRef[:,:,:,0])
            imgRefWithErrorSegs[:,:,:,2] = np.where(labFP >= 0.7, labFP, imgRef[:,:,:,2])
            imgRefWithErrorSegs[:,:,:,1] = np.where(labTP >= 0.7, labTP*0.5, imgRef[:,:,:,1])
            saveImageColor(imgRefWithErrorSegs[slice,:,:],fpSubDir_Errors,'imgErrors')
