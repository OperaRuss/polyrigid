'''
Russell Wustenberg, rw2873
polyrigid refactor
start date 20220220
'''

import os
import glob
import random
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import utilities as utils
from sklearn.metrics import f1_score
import Polyrigid as pr
import Evaluation as eval

def get_model(imgFloating: torch.tensor, componentSegmentations: dict,
              componentWeights: dict, learningRate: float = 0.005):
    model = pr.Polyrigid(imgFloating, componentSegmentations, componentWeights).to('cuda')
    return model, torch.optim.Adam(model.parameters(), learningRate)


# SECTION 1: Model Parameters
# vMaxItrs and vLambda may be set to list objects in order to run
# multiple consecutive runs using different parameters.


def estimateKinematics(inFolder: str, inPrefixImg: str, inPrefixSeg:str, imgFloat: str,
                       imgTarget: str, outFolder: str, numComponents: int, maxItrs: int,
                       learningRate: float, stopLoss: float, updateRate: int,
                       alpha: float, beta: float, gamma: float, delta: float, epsilon: float, zeta: float):
    '''
    Extracts kinematic estimates based on a polyrigid registration model.
    :param inFolder: Path to the folder with the input information.
    :param inPrefix: String for leading prefix for input files.
    :param imgFloat: String of the format 'frame_i' for files of type (e.g. frame_i.nii)
    :param imgTarget: String for the target image.
    :param numComponents: Number of rigid components in the image.  Segmentations must exist for these.
    :param maxItrs: Maximum number of iterations desired for registration
    :param learningRate: Learning rate for the model.
    :param stopLoss: Loss cutoff for an acceptable model.
    :param alpha: Hyperparameter. Throttles all regulation scores.
    :param beta: Hyperparameter. Controls strength of smoothness regulation.
    :param gamma: Hyperparameter. Controls strength of negative Jacobian determinant regulation.
    :param delta: Hyperparameter. Controls strength of rigidity regularation on rotations.
    :param epsilon: Hyperparameter.  Controls weight decay for rigid components in weight volume.
    :param zeta: Hyperparameter. Controls segmentation thresholding for shape consistency in propagation.
    '''
    print("====================================================")
    print("Estimating registration of ",imgFloat,"to",imgTarget)
    print("\tModel running with:")
    print("\tAlpha:",alpha,"Beta:",beta,"Gamma:",gamma)
    print("\tDelta:",delta,"Zeta:",zeta,"Epsilon:",epsilon)
    print("====================================================")

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    vSubFolder = imgFloat + '_to_' + imgTarget
    vOutPath = os.path.join(outFolder, vSubFolder)

    if not os.path.exists(vOutPath):
        os.makedirs(vOutPath)

    # SECTION 2: READING IN IMAGE DATA
    imgSITK_moving = sitk.ReadImage(inFolder + inPrefixImg
                                    + imgFloat + ".nii")
    tImgMoving = utils.normalizeImage(
        sitk.GetArrayFromImage(imgSITK_moving)[10:70, 60:160, 50:150])
    tImgMoving = torch.tensor(tImgMoving, dtype=torch.float32)
    tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

    imgSITK_target = sitk.ReadImage(inFolder + inPrefixImg
                                    + imgTarget + ".nii")
    tImgTarget = utils.normalizeImage(
        sitk.GetArrayFromImage(imgSITK_target)[10:70, 60:160, 50:150])
    tImgTarget = torch.tensor(tImgTarget, dtype=torch.float32)
    tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

    assert tImgMoving.shape == tImgTarget.shape, \
        "Images must be of the same dimensions. Got %s != %s" % (
            tImgMoving.shape, tImgTarget.shape)

    aComponentSegmentations_Float = {}
    aComponentWeightValues = {}
    aComponentSegmentations_Target = {}

    # If we have previously registered to the floating image, there must exist a set of warped
    # segmentations from our original segmentation.  We seek to propagate these, so use them as
    # the floating image segmentations.
    if glob.glob(outFolder + "/*_to_" + imgFloat):
        previousRegistration = glob.glob(outFolder + "/*_to_" + imgFloat)[0]
        for i in range(1, numComponents):
            temp = sitk.ReadImage(os.path.join(previousRegistration,
                                               "warped_seg_" + str(i) + ".nii"))
            aComponentSegmentations_Float[i] = sitk.GetArrayFromImage(temp)

            temp = sitk.ReadImage(inFolder + inPrefixSeg
                                  + vInFrame_Target + "_seg_" + str(i) + ".nii")
            aComponentSegmentations_Target[i] = sitk.GetArrayFromImage(temp)[10:70, 60:160, 50:150]
    else:
        for i in range(1, numComponents):
            temp = sitk.ReadImage(inFolder + inPrefixSeg
                                  + vInFrame_Float + "_seg_" + str(i) + ".nii.gz")
            aComponentSegmentations_Float[i] = sitk.GetArrayFromImage(temp)[10:70, 60:160, 50:150]

            temp = sitk.ReadImage(inFolder + inPrefixSeg
                                  + vInFrame_Target + "_seg_" + str(i) + ".nii.gz")
            aComponentSegmentations_Target[i] = sitk.GetArrayFromImage(temp)[10:70, 60:160, 50:150]

    for idx, img in aComponentSegmentations_Float.items():
        aComponentSegmentations_Float[idx] = utils.normalizeImage(img)

    for idx, img in aComponentSegmentations_Float.items():
        aComponentWeightValues[idx] = epsilon

    model, optimizer = get_model(tImgMoving, aComponentSegmentations_Float,
                                 aComponentWeightValues, learningRate=learningRate)

    tComponentSegImg_Target = np.zeros(model.mImageDimensions)
    for label, img in aComponentSegmentations_Target.items():
        if 1 <= label <= 8:
            tComponentSegImg_Target += img
    tComponentSegImg_Target = torch.tensor(tComponentSegImg_Target, dtype=torch.float32).cuda()

    vLossHistory = {}

    # STEP 7: ITERATION TOWARDS OPTIMIZATION
    for itr in range(maxItrs):
        optimizer.zero_grad()

        tImgWarped = model.forward()
        tImgFixed = 1.0 * tImgTarget

        DICE_loss = model._getLoss_DICE(tComponentSegImg_Target)
        similarity_loss = alpha * utils._getMetricMSE(tImgWarped, tImgFixed)
        smooth_loss = beta * utils._loss_Smooth(model.tDisplacementField)
        jdet_loss = gamma * utils._loss_JDet(model.tDisplacementField)
        rigid_loss = delta * model._getLoss_Rigidity()
        trans_loss = zeta * model._getLoss_Translation_L2()
        loss = DICE_loss + similarity_loss + rigid_loss + smooth_loss + jdet_loss + trans_loss

        # Calculate gradients wrt parameters
        loss.backward()

        # Update parameters based on gradients
        optimizer.step()
        vLossHistory[str(itr)] = loss.item()

        # check for exit conditions.  Still not clear on what constitutes provable convergence.

        if itr % updateRate == 0:
            print(
                "Itr: {}, Total {} DICE {} Sim {} Smooth {} Jdet {} Rigid {} Trans {}".format(
                    itr, loss.item(), DICE_loss.item(),similarity_loss.item(), smooth_loss.item(),
                    jdet_loss.item(), rigid_loss.item(), trans_loss.item()
                )
            )

        if abs(loss) < stopLoss:
            print("Model converged at iteration ", itr, " with loss score ", loss)
            break

    # SECTION 8: MODEL OUTPUT
    # The below sections produce results from the run of the model and put them in the specified
    # output folder.  Starts with establishing a folder based on the current model parameters then
    # outputs all output into that folder.

    tImgWarped_Seg = F.grid_sample(model.tImgSegmentation,
                                   model.tDisplacementField,
                                   mode='nearest', padding_mode='zeros',
                                   align_corners=False)

    for label, seg in aComponentSegmentations_Float.items():
        tTemp = F.grid_sample(torch.tensor(seg, dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0),
                              model.tDisplacementField,
                              mode='nearest', padding_mode='zeros',
                              align_corners=False)
        npTemp = tTemp.detach().squeeze().cpu().numpy()

        aComponentCfsn = eval.confusionMatrix(aComponentSegmentations_Target[label],
                                              npTemp,vOutPath,int(imgTarget.split('_')[-1]),
                                              "Component_"+str(label)+"_")
        np.savez(vOutPath+'/component_'+str(label)+'_cfsn',**aComponentCfsn)

        sitkTemp = sitk.GetImageFromArray(npTemp)
        sitk.WriteImage(sitkTemp, vOutPath + '/warped_seg_' + str(label) + '.nii')

    sitkDVF = sitk.GetImageFromArray(model._getLEPT().detach().squeeze().cpu().numpy(),
                                     isVector=True)
    sitk.WriteImage(sitkDVF, vOutPath
                    + '/LEPT_frame_' + str(source) + '_to_' + str(target) + '.nii')
    npPredSeg = tImgWarped_Seg.detach().cpu().numpy()
    npTargSeg = tComponentSegImg_Target.detach().cpu().numpy()

    cfsn = eval.confusionMatrix(npTargSeg,npPredSeg,vOutPath,imgTarget)

    tImgWarped = \
        sitk.GetImageFromArray(tImgWarped.detach().squeeze().cpu().numpy(), False)
    sitk.WriteImage(tImgWarped, vOutPath + "/imgWarped.nii")

    vDICE_Before = \
        f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1, 1),
                 model.tImgSegmentation.detach().cpu().numpy().reshape(-1, 1), average='macro')
    vDICE_After = \
        f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1, 1),
                 tImgWarped_Seg.detach().cpu().view((-1, 1)).numpy(), average="macro")

    tComponentSegImg_Float = \
        sitk.GetImageFromArray(model.tImgSegmentation.detach().cpu().numpy(), False)
    sitk.WriteImage(tComponentSegImg_Float, vOutPath + "/float_seg_binary.nii")

    tImgWarped_Seg = \
        sitk.GetImageFromArray(tImgWarped_Seg.detach().squeeze().cpu().numpy(), False)

    sitk.WriteImage(tImgWarped_Seg, vOutPath + "/warped_seg_binary.nii")

    tComponentSegImg_Target = \
        sitk.GetImageFromArray(tComponentSegImg_Target.detach().cpu().numpy(), False)
    sitk.WriteImage(tComponentSegImg_Target, vOutPath + "/target_seg_binary.nii")

    # Due to the Grid Sample coordinate system scaling, we need to evaluate
    # The DVF on a synthetic point system.  This is scaled to the original
    # volume dimensions and gives gradient information in expected values.
    # We use a custom function for JDet taken from the LapriMRI github repository.
    # The results were validated with a built-in calculation in SITK.
    tSamplePoints_Depth = torch.linspace(0, 20, steps=model.mImageDimensions[2])
    tSamplePoints_Height = torch.linspace(0, 200, steps=model.mImageDimensions[3])
    tSamplePoints_Width = torch.linspace(0, 200, steps=model.mImageDimensions[4])
    tSamplePoints = torch.cartesian_prod(tSamplePoints_Depth,
                                         tSamplePoints_Height,
                                         tSamplePoints_Width)
    tOnes = torch.ones(np.prod(model.mImageDimensions), dtype=torch.float32)
    tSamplePoints = torch.cat((tSamplePoints, tOnes.unsqueeze(-1)), dim=1).unsqueeze(-1).cuda()
    tTestDVF = model._getLEPT(tSamplePoints)

    tDeterminantMap = utils.jacobian_determinant_3d(tTestDVF)
    tDeterminantMap = tDeterminantMap.detach().squeeze().cpu().numpy()
    vNumNeg = (tDeterminantMap <= 0.0).sum()
    print("Num neg dets: ", vNumNeg)

    # Here we output the log of the determinant map to visualize contraction
    # and expansion more easily.  We assume that there is a low percentage
    # of non-invertable points.
    tDeterminantMap_log = np.log(tDeterminantMap)
    tDeterminantMap = sitk.GetImageFromArray(tDeterminantMap_log, False)
    sitk.WriteImage(tDeterminantMap, vOutPath + "/log_jacobian_determinants.nii")

    vComponentFinalTransforms_Euclidean = {}
    vComponentFinalTransforms_Log = {}
    vModelResults = {'target': int(imgTarget.split('_')[-1]),
                     'numNeg': vNumNeg,
                     'finalLoss': loss.item(),
                     'netDICE': vDICE_After - vDICE_Before}
    cfsn['target'] = int(imgTarget.split('_')[-1])
    vItrRigidityScores = []

    with open(vOutPath + "/res_transforms.txt", "w") as out:
        print(f"MaxIterations: {maxItrs}", file=out)
        print(f"Learning Rate: {learningRate}", file=out)
        print(f"Similarity Parameter: {alpha}",file=out)
        print(f"Smoothness Parameter: {beta}", file=out)
        print(f"Jacobian Regularization Parameter: {gamma}", file=out)
        print(f"Rigidity Regularization Parameter: {delta}",file=out)
        print(f"DICE score before registration: {vDICE_Before:.4f}", file=out)
        print(f"DICE score after registration: {vDICE_After:.4f}", file=out)
        print(f"Loss achieved: {loss:.4f}", file=out)
        percJDET = (vNumNeg / (np.prod(model.mImageDimensions)) * 100)
        vModelResults['percentageNegJDets'] = percJDET
        print(f"Percentage of Jacobian determinants negative: " +
              f"{percJDET:.2f}%", file=out)
        print("Confusion Matrix for all carpals:")
        print("\tP\t\tN")
        print(f"P\t{cfsn['TP']}\t{cfsn['FN']}")
        print(f"N\t{cfsn['FP']}\t{cfsn['TN']}\t")
        print("Final parameter Estimations:\n", file=out)
        for i in range(0, 8):
            aCompTransforms = model._getLogComponentTransforms()
            print("Component " + str(i + 1), file=out)
            transform = torch.matrix_exp(torch.reshape(aCompTransforms[i],
                                                       (model.mNDims + 1, model.mNDims + 1)))
            transform = transform[0:3, 0:3]
            RRT = torch.sub(torch.matmul(transform, transform.T), torch.eye(3, device='cuda'))
            RTR = torch.sub(torch.matmul(transform.T, transform), torch.eye(3, device='cuda'))
            Rdet = torch.det(transform) - 1.0
            rigidity = torch.frobenius_norm(RRT) + torch.frobenius_norm(RTR) + Rdet
            vItrRigidityScores.append(rigidity.item())
            print("Rigidity Score: ", rigidity.item(), file=out)
            vCompTransform_Log = torch.reshape(aCompTransforms[i],
                                               (model.mNDims + 1,
                                                model.mNDims + 1))
            vComponentFinalTransforms_Log[str(i)] = \
                vCompTransform_Log.detach().cpu().numpy()
            vCompTransform = torch.matrix_exp(vCompTransform_Log)
            vCompTransform = vCompTransform.detach().cpu().numpy()
            vComponentFinalTransforms_Euclidean[str(i)] = vCompTransform
            print(vCompTransform, file=out)

    vModelResults['meanRigidityScore'] = np.mean(vItrRigidityScores)
    np.savez(vOutPath+'/confusionMatrix',**cfsn)
    np.savez(vOutPath + '/component_transforms_final_log',
             **vComponentFinalTransforms_Log)
    np.savez(vOutPath + '/component_transforms_final_euclidean',
             **vComponentFinalTransforms_Euclidean)
    np.savez(vOutPath + '/model_results', **vModelResults)
    np.savez(vOutPath + '/model_loss',**vLossHistory)

    imgTarget = sitk.GetImageFromArray(tImgTarget.detach().squeeze().cpu().numpy()
                                       ,isVector=False)
    sitk.WriteImage(imgTarget,vOutPath + '/imgReference.nii')
    imgFloat = sitk.GetImageFromArray(tImgMoving.detach().squeeze().cpu().numpy()
                                      ,isVector=False)
    sitk.WriteImage(imgFloat,vOutPath + '/imgFloat.nii')

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    vStopLoss = 1e-5
    vLearningRate = 0.01
    vMaxItrs = 100
    vUpdateRate = 10
    vNumComponents = 15
    vInFolder = "../images/input/rave/"
    vInFrame_Float_Prefix = "iso_"
    vInSeg_Float_Prefix = "em_"
    vStartFrame = 10
    vEndFrame_hi = -1
    vEndFrame_low = -1
    vNumFrames = 20
    vRunID = '1'
    vOutFile = "../images/results/"
    vStride = 1
    vAlpha = [1.0] # Signal strength for all regularization
    vBeta = [0.0]  # Signal strength for smoothness regularization
    vGamma = [0.0]  # Signal strength for negative JD regularization
    vDelta = [0.] # Signal strength for rigidity regularization
    vEpsilon = [0.] # Translation Regularization
    vZeta = [0.] # Component weighting parameter


    if vEndFrame_hi == -1:
        vEndFrame_hi = vNumFrames - 1
    if vEndFrame_low == -1:
        vEndFrame_low = 0

    for alpha in vAlpha:
        for beta in vBeta:
            for gamma in vGamma:
                for delta in vDelta:
                    for zeta in vZeta:
                        for epsilon in vEpsilon:
                            for source in range(vStartFrame, vEndFrame_hi, vStride):
                                #div = abs(beta)+abs(gamma)+abs(delta)+abs(epsilon)
                                #beta = abs(beta)/div
                                #gamma = abs(gamma)/div
                                #delta = abs(delta)/div
                                #epsilon = abs(epsilon)/div

                                if (source + vStride > vNumFrames - 1):
                                    print("Cannot register outside of frame sequence.")
                                    break
                                target = source + vStride
                                vInFrame_Float = "frame_" + str(source)
                                vInFrame_Target = "frame_" + str(target)

                                estimateKinematics(vInFolder, vInFrame_Float_Prefix, vInSeg_Float_Prefix, vInFrame_Float,
                                                   vInFrame_Target, vOutFile, vNumComponents, vMaxItrs, vLearningRate,
                                                   vStopLoss, vUpdateRate, alpha, beta, gamma, delta, zeta, epsilon)

                            for source in range(vStartFrame, vEndFrame_low, -vStride):
                                if (source - vStride < 0):
                                    print("Cannot register outside of frame sequence.")
                                    break
                                target = source - vStride
                                vInFrame_Float = "frame_" + str(source)
                                vInFrame_Target = "frame_" + str(target)

                                estimateKinematics(vInFolder, vInFrame_Float_Prefix, vInSeg_Float_Prefix, vInFrame_Float,
                                                   vInFrame_Target, vOutFile, vNumComponents, vMaxItrs, vLearningRate,
                                                   vStopLoss, vUpdateRate, alpha, beta, gamma, delta, zeta, epsilon)


                            label = f"_a_{str(alpha).replace('.','_')}_b_{str(beta).replace('.','_')}"\
                                    F"_g_{str(gamma).replace('.','_')}_d_{str(delta).replace('.','_')}"\
                                    f"_e_{str(epsilon).replace('.', '_')}"
                            eval.getEvaluationPlots(label)
                            eval.getStaticMeshes(params=label)
