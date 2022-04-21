import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def getEvaluationPlots(params:str=None):
    fpResults = "../images/results"

    fig, ax = plt.subplots(1,1)
    for file in sorted(glob.glob(fpResults+"/*")):
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
        plt.savefig("../images/results/summary_MSE"+params+".png",bbox_inches='tight')
    else:
        plt.savefig("../images/results/summary_MSE.png",bbox_inches='tight')
    plt.show()
    plt.close()

    finLoss = {}
    netDICE = {}
    percNegJDets = {}
    meanRigid = {}

    fig, axs = plt.subplots(2,2,figsize=(10,10)) # in clockwise from upper left: numNeg, netDICE, percentageNegJDets,meanRigidityScore
    for file in sorted(glob.glob(fpResults+"/*")):
        if os.path.isdir(file):
            results = np.load(os.path.join(file, 'model_results.npz'))
            label = int(results['target'])
            finLoss[label] = results['finalLoss']
            netDICE[label] = results['netDICE']
            percNegJDets[label] = results['percentageNegJDets']
            meanRigid[label] = results['meanRigidityScore']

    x,y = zip(*sorted(finLoss.items()))
    axs[0,0].plot(x,y,'-ok')
    axs[0,0].set(title="Final Loss Score Achieved")
    axs[0,0].set_ylim(0.000,0.1)
    axs[0,0].axhline(np.mean(y),linestyle='--',color='r')
    axs[0,0].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(netDICE.items()))
    axs[0,1].plot(x,y,'-ok')
    axs[0,1].set(title="Net DICE Gain")
    axs[0,1].set_ylim(-0.05,0.03)
    axs[0,1].axhline(np.mean(y),linestyle='--',color='r')
    axs[0,1].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(percNegJDets.items()))
    axs[1,0].plot(x,y,'-ok')
    axs[1,0].set(title="% Neg JDets")
    axs[1,0].set_ylim(0.00,0.2)
    axs[1,0].axhline(np.mean(y),linestyle='--',color='r')
    axs[1,0].set_xticks(np.arange(0,21,2))

    x,y = zip(*sorted(meanRigid.items()))
    axs[1,1].plot(x,y,'-ok')
    axs[1,1].set(title="Mean Rigidity Score")
    axs[1,1].set_ylim(0.,0.01)
    axs[1,1].axhline(np.mean(y),linestyle='--',color='r')
    axs[1,1].set_xticks(np.arange(0,21,2))

    fig.suptitle("Final Evaluation Scores After Registration on All Frames")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    if params is not None:
        plt.savefig("../images/results/summary_metrics"+params+".png",bbox_inches='tight')
    else:
        plt.savefig("../images/results/summary_metrics.png", bbox_inches='tight')
    plt.show()
    plt.close()

    sequenceConfusion(fpResults)


def confusionMatrix(targetSeg, predictionSeg,outPath):
    imgTP = np.where((targetSeg==predictionSeg),targetSeg,0)
    imgFP = np.where((predictionSeg!=targetSeg),predictionSeg,0)
    imgFN = np.where((predictionSeg!=targetSeg),targetSeg,0)
    imgTN = np.multiply(np.subtract(np.max(targetSeg),targetSeg),
                                    np.subtract(np.max(predictionSeg),predictionSeg))
    imgIoU = targetSeg + predictionSeg - imgTP

    sitk.WriteImage(sitk.GetImageFromArray(imgTP,False),outPath+'/imgTruePositives.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgFP,False),outPath+'/imgFalsePositives.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgFN,False),outPath+'/imgFalseNegatives.nii')
    sitk.WriteImage(sitk.GetImageFromArray(imgTN,False),outPath+'/imgTrueNegatives.nii')

    TP = np.count_nonzero(imgTP)
    FP = np.count_nonzero(imgFP)
    FN = np.count_nonzero(imgFN)
    TN = np.count_nonzero(imgTN)
    IoU = np.count_nonzero(imgIoU)

    out = {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'DICE':2*(TP/IoU),
           'accuracy':TP/(TP+FP+FN+TN),'precision':TP/(TP+FP),'recall':TP/(TP+FN)}
    return out

def sequenceConfusion(outPath):
    fpResults = "../images/results"

    aDice = {}
    aPrecision = {}
    aRecall = {}
    aAccuracy = {}

    for file in sorted(glob.glob(fpResults + "/*")):
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
    diceAx.axhline(np.mean(y), linestyle='--', color='r')
    diceAx.axvline(10,linestyle='--',color='b',alpha=0.5)
    diceAx.set_xticks(np.arange(0, 21, 2))
    dice.savefig(outPath+'/summaryDICE.png',bbox_inches='tight')

    precision, precAx = plt.subplots(1)
    x, y = zip(*sorted(aPrecision.items()))
    precAx.plot(x, y, '-ok')
    precAx.set(title="Precision Across Sequence")
    precAx.set_ylabel('Precision Score')
    precAx.set_xlabel('Target Frame')
    precAx.axhline(np.mean(y), linestyle='--', color='r')
    precAx.axvline(10,linestyle='--',color='b',alpha=0.5)
    precAx.set_xticks(np.arange(0, 21, 2))
    precision.savefig(outPath+'/summaryPrecision.png',bbox_inches='tight')

    recall, recallAx = plt.subplots(1)
    x, y = zip(*sorted(aPrecision.items()))
    recallAx.plot(x, y, '-ok')
    recallAx.set(title="Recall Across Sequence")
    recallAx.set_ylabel('Recall Score')
    recallAx.set_xlabel('Target Frame')
    recallAx.axhline(np.mean(y), linestyle='--', color='r')
    recallAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    recallAx.set_xticks(np.arange(0, 21, 2))
    recall.savefig(outPath + '/summaryRecall.png', bbox_inches='tight')

    accuracy, accAx = plt.subplots(1)
    accAx.plot(x, y, '-ok')
    accAx.set(title="Accuracy Across Sequence")
    accAx.set_ylabel('Accuracy Score')
    accAx.set_xlabel('Target Frame')
    accAx.axhline(np.mean(y), linestyle='--', color='r')
    accAx.axvline(10, linestyle='--', color='b', alpha=0.5)
    accAx.set_xticks(np.arange(0, 21, 2))
    accuracy.savefig(outPath + '/summaryAccuracy.png', bbox_inches='tight')
