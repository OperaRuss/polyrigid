'''
Russell Wustenberg, rw2873
polyrigid refactor
start date 20220220

README
There are a number of I/O steps which are simply hard-coded in for rapid prototyping
at this time.  If you want to skip them, comment them out.  Else, make sure to change these things:

1) File Structure   The expected file structure for the I/O is as follows
                        images
                            cropped
                                float_frame
                                    frame_data.nii
                                    frame_seg_0.nii
                                    frame_seg_1.nii
                                    ...
                                    frame_seg_N.nii
                                target_frame
                                    frame_data.nii
                                    frame_seg_0.nii
                                    frame_seg_1.nii
                                    ...
                                    frame_seg_N.nii
                            full-size frames
                                * not used in current implementation
                                * Follows same format as cropped
                            results
                                * automatically created based on the run
                                date
                                    run_results


2) vDate  --  Change to today's date
3) Paths  --  Most of these
4) Results -- The program expects there to exist a 'results' folder in the 'images' folder.
              This is used for output of final charts.  Note that the error graph is NOT
              automatically saved at this time.

Note on interpreting error: the NCC function used here was developed by the Voxel Morph team
to massage NCC into a learning metric.  As such, they compute the NCC coefficients for all blocks
across the image (in a one-to-one mapping) and then take their average to produce an error score.
In order for the metric to be learnable, they add a negative sign BEFORE returning it.  So, the
error is bounded [0,1] but it will report [0,-1] with the error trending toward -1 with a better fit.
'''

import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utilities as utils
from sklearn.metrics import f1_score
import Polyrigid as pr


def get_model(imgFloating: torch.tensor, componentSegmentations: dict,
              componentWeights: dict, learningRate: float = 0.005):
    model = pr.Polyrigid(imgFloating, componentSegmentations, componentWeights).to('cuda')
    return model, torch.optim.Adam(model.parameters(), learningRate)


# SECTION 1: Model Parameters
# vMaxItrs and vLambda may be set to list objects in order to run
# multiple consecutive runs using different parameters.
vStop_Loss = 1e-5
vStep_Size = [0.01]
vMaxItrs = [200]
vUpdateRate = 10
vItrHistory = {}
vLongNegJDHistory = {}
vLongLossHistory = {}
vLongRigidityScores = {}
vLongRelativeDice = {}
vNumComponents = [15]
vInFolder = "../images/input/rave/"
vInFrame_Float_Prefix = "iso_"
vInSeg_Float_Prefix = "em_"
vStartFrame = 10
vEndFrame_hi = 11
vEndFrame_low = 9
vNumFrames = 20
vOutFile = "../images/results/"
vDate = "20220412"
vBeta = [0.75]  # 0.0 for full Jdet, 1.0 for full smoothness
vAlpha = [0.75]  # on/off for disp field regularization
vGamma = [0.015]  # [0,1] for rigid transformation estimation
vDelta = [0.5]

if vEndFrame_hi == -1:
    vEndFrame_hi = vNumFrames - 1
if vEndFrame_low == -1:
    vEndFrame_low = 0

for source in range(vStartFrame,vEndFrame_hi):
    if (source+1 > vNumFrames - 1):
        print("Cannot register outside of frame sequence.")
        exit(1)
    target = source + 1
    vInFrame_Float = "frame_" + str(source)
    vInFrame_Target = "frame_" + str(target)

    for pMaxItrs in vMaxItrs:
        for pBeta in vBeta:
            for pAlpha in vAlpha:
                for pGamma in vGamma:
                    for pDelta in vDelta:
                        for pStepSize in vStep_Size:
                            for pNumComps in vNumComponents:
                                vCurrTestKey = pAlpha

                                if not os.path.exists(vOutFile):
                                    os.makedirs(vOutFile)

                                if not os.path.exists(os.path.join(vOutFile, vDate)):
                                    os.makedirs(os.path.join(vOutFile, vDate))

                                # SECTION 2: READING IN IMAGE DATA
                                imgSITK_moving = sitk.ReadImage(vInFolder + vInFrame_Float_Prefix
                                                                + vInFrame_Float + ".nii")
                                imgMoving = utils.normalizeImage(
                                    sitk.GetArrayFromImage(imgSITK_moving)[10:70, 60:160, 50:150])
                                tImgMoving = torch.tensor(imgMoving, dtype=torch.float32)
                                tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

                                imgSITK_target = sitk.ReadImage(vInFolder + vInFrame_Float_Prefix
                                                                + vInFrame_Target + ".nii")
                                imgTarget = utils.normalizeImage(
                                    sitk.GetArrayFromImage(imgSITK_target)[10:70, 60:160, 50:150])
                                tImgTarget = torch.tensor(imgTarget, dtype=torch.float32)
                                tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

                                assert imgMoving.shape == imgTarget.shape, \
                                    "Images must be of the same dimensions. Got %s != %s" % (
                                    imgMoving.shape, imgTarget.shape)

                                aComponentSegmentations_Float = {}
                                aComponentWeightValues = {}
                                aComponentSegmentations_Target = {}

                                for i in range(1, pNumComps):
                                    temp = sitk.ReadImage(vInFolder + vInSeg_Float_Prefix
                                                          + vInFrame_Float + "_seg_" + str(i) + ".nii.gz")
                                    aComponentSegmentations_Float[i] = sitk.GetArrayFromImage(temp)[10:70, 60:160, 50:150]

                                    temp = sitk.ReadImage(vInFolder + vInSeg_Float_Prefix
                                                          + vInFrame_Target + "_seg_" + str(i) + ".nii.gz")
                                    aComponentSegmentations_Target[i] = sitk.GetArrayFromImage(temp)[10:70, 60:160, 50:150]

                                for idx, img in aComponentSegmentations_Float.items():
                                    aComponentSegmentations_Float[idx] = utils.normalizeImage(img)

                                for idx, img in aComponentSegmentations_Float.items():
                                    aComponentWeightValues[idx] = pDelta

                                model, optimizer = get_model(tImgMoving, aComponentSegmentations_Float,
                                                             aComponentWeightValues, learningRate=pStepSize)

                                tComponentSegImg_Target = np.zeros(model.mImageDimensions)
                                for label, img in aComponentSegmentations_Target.items():
                                    if 1 <= label <= 8:
                                        tComponentSegImg_Target += img
                                tComponentSegImg_Target = torch.tensor(tComponentSegImg_Target, dtype=torch.float32).cuda()

                                # STEP 7: ITERATION TOWARDS OPTIMIZATION
                                print("Running with Smoothness Parameter " + str(pBeta)
                                      + " and JD Regularization parameter " + str(1.0 - pBeta))
                                print("Running model for maximum of " + str(pMaxItrs) + " iterations.")
                                print("Running model with weight parameter ", str(pDelta))
                                print("Running model with rigidity gain of ", str(pGamma))
                                print("Running model with ", pNumComps, "components.")

                                for itr in range(pMaxItrs):
                                    optimizer.zero_grad()

                                    tImgWarped = model.forward()
                                    tImgFixed = 1.0 * tImgTarget

                                    reg_loss = utils._getMetricMSE(tImgWarped, tImgFixed)
                                    smooth_loss = pAlpha * pBeta * utils._loss_Smooth(model.tDisplacementField)
                                    jdet_loss = pAlpha * (1 - pBeta) * utils._loss_JDet(model.tDisplacementField)
                                    rigid_loss = pGamma * model._getLoss_Rigidity()
                                    loss = reg_loss + rigid_loss + smooth_loss + jdet_loss

                                    # Calculate gradients wrt parameters
                                    loss.backward()

                                    # Update parameters based on gradients
                                    optimizer.step()
                                    vItrHistory[itr] = loss.item()

                                    # check for exit conditions.  Still not clear on what constitutes provable convergence.

                                    if itr % vUpdateRate == 0:
                                        print(
                                            "Itr: {}, Total {} Reg {} Smooth {} Jdet {}".format(
                                                itr, loss.item(), reg_loss.item(), smooth_loss.item(), jdet_loss.item(),
                                            )
                                        )

                                    if abs(loss) < vStop_Loss:
                                        print("Model converged at iteration ", itr, " with loss score ", loss)
                                        break

                                    if itr > 10:
                                        if abs((loss + vItrHistory[itr - 1] + vItrHistory[itr - 2]) / 3.0) < 1e-5:
                                            print("Model growth slowed at iteration", itr, "with loss score ", loss)
                                            break

                                # SECTION 8: MODEL OUTPUT
                                # The below sections produce results from the run of the model and put them in the specified
                                # output folder.  Starts with establishing a folder based on the current model parameters then
                                # outputs all output into that folder.

                                vSubFolder = vInFrame_Float + '_to_' + vInFrame_Target
                                vOutPath = os.path.join(vOutFile + vDate, vSubFolder)

                                if not os.path.exists(vOutPath):
                                    os.makedirs(vOutPath)

                                tImgWarped_Seg = F.grid_sample(model.tImgSegmentation,
                                                               model.tDisplacementField,
                                                               mode='nearest', padding_mode='zeros',
                                                               align_corners=False)

                                for label, seg in aComponentSegmentations_Float.items():
                                    tTemp = F.grid_sample(torch.tensor(seg,dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(0),
                                                          model.tDisplacementField,
                                                          mode='nearest', padding_mode='zeros',
                                                          align_corners=False)
                                    sitkTemp = sitk.GetImageFromArray(tTemp.detach().squeeze().cpu().numpy())
                                    sitk.WriteImage(sitkTemp, vOutPath + '/warped_seg_'+str(label)+'.nii')

                                sitkDVF = sitk.GetImageFromArray(model._getLEPT().detach().squeeze().cpu().numpy(),
                                                                 isVector=True)
                                sitk.WriteImage(sitkDVF, vOutPath
                                                + '/LEPT_frame_' + str(source) + '_to_' + str(target) + '.nii')

                                tImgWarped = \
                                    sitk.GetImageFromArray(tImgWarped.detach().squeeze().cpu().numpy(), False)
                                sitk.WriteImage(tImgWarped, vOutPath + "/imgWarped.nii")
                                tImgWarped = sitk.GetArrayFromImage(tImgWarped)

                                vDICE_Before = \
                                    f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1, 1),
                                             model.tImgSegmentation.detach().cpu().numpy().reshape(-1, 1), average='macro')
                                vDICE_After = \
                                    f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1, 1),
                                             tImgWarped_Seg.detach().cpu().view((-1, 1)).numpy(), average="macro")

                                tComponentSegImg_Float = \
                                    sitk.GetImageFromArray(model.tImgSegmentation.detach().cpu().numpy(), False)
                                sitk.WriteImage(tComponentSegImg_Float, vOutPath + "/float_seg_binary.nii")
                                tComponentSegImg_Float = sitk.GetArrayFromImage(tComponentSegImg_Float)

                                tImgWarped_Seg = \
                                    sitk.GetImageFromArray(tImgWarped_Seg.detach().squeeze().cpu().numpy(), False)

                                sitk.WriteImage(tImgWarped_Seg, vOutPath + "/warped_seg_binary.nii")

                                tComponentSegImg_Target = \
                                    sitk.GetImageFromArray(tComponentSegImg_Target.detach().cpu().numpy(), False)
                                sitk.WriteImage(tComponentSegImg_Target, vOutPath + "/target_seg_binary.nii")
                                tComponentSegImg_Target = sitk.GetArrayFromImage(tComponentSegImg_Target)

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

                                vLongNegJDHistory[vCurrTestKey] = vNumNeg
                                vLongLossHistory[vCurrTestKey] = loss.item()
                                vLongRelativeDice[vCurrTestKey] = vDICE_After - vDICE_Before

                                # Here we output the log of the determinant map to visualize contraction
                                # and expansion more easily.  We assume that there is a low percentage
                                # of non-invertable points.
                                tDeterminantMap_log = np.log(tDeterminantMap)
                                tDeterminantMap = sitk.GetImageFromArray(tDeterminantMap, False)
                                sitk.WriteImage(tDeterminantMap, vOutPath + "/log_jacobian_determinants.nii")

                                vComponentFinalTransforms_Euclidean = {}
                                vComponentFinalTransforms_Log = {}
                                vModelResults = {'numNeg':vNumNeg,
                                                 'finalLoss':loss.item(),
                                                 'netDICE':vDICE_After-vDICE_Before}
                                vItrRigidityScores = []
                                with open(vOutPath + "/res_transforms.txt", "w") as out:
                                    print(f"MaxIterations: {pMaxItrs}", file=out)
                                    print(f"Learning Rate: {pStepSize}", file=out)
                                    print(f"Smoothness Parameter: {pAlpha * pBeta}", file=out)
                                    print(f"Jacobian Regularization Parameter: {pAlpha * (1.0 - pBeta)}", file=out)
                                    print(f"DICE score before registration: {vDICE_Before:.4f}", file=out)
                                    print(f"DICE score after registration: {vDICE_After:.4f}", file=out)
                                    print(f"Target Loss: {vStop_Loss:.4f}", file=out)
                                    print(f"Loss achieved: {loss:.4f}", file=out)
                                    percJDET = (vNumNeg / (np.prod(model.mImageDimensions)) * 100)
                                    vModelResults['percentageNegJDets'] = percJDET
                                    print(f"Percentage of Jacobian determinants negative: " +
                                          f"{percJDET:.2f}%", file=out)
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
                                vLongRigidityScores[vCurrTestKey] = np.mean(vItrRigidityScores)
                                np.savez(vOutPath + '/component_transforms_final_log',
                                         **vComponentFinalTransforms_Log)
                                np.savez(vOutPath + '/component_transforms_final_euclidean',
                                         **vComponentFinalTransforms_Euclidean)
                                np.savez(vOutPath + '/model_results', **vModelResults)

for source in range(vStartFrame,vEndFrame_low,-1):
    pass

data = sorted(vLongNegJDHistory.items())
x, y = zip(*data)
fig = plt.plot(x, y, marker=".", markersize=10, color='tab:blue')
plt.ylabel("Num. Vector Displacements with Neg. Determinant")
plt.xlabel("Number of Components Included")
plt.tick_params(axis='y', labelcolor='tab:blue')
plt.title("Neg. Determinants by Num. Components")
plt.savefig(vOutPath + "/plot_Comps2negJD.png", bbox_inches='tight')
plt.close()
'''
data = sorted(vLongNegJDHistory.items())
x, y = zip(*data)
fig, ax1 = plt.subplots()
ax1.plot(x, y, marker=".", markersize=10, color='tab:blue')
ax1.set_ylabel("Num. Vector Displacements with Neg. Determinant")
ax1.set_xlabel("Signal Gain of Rigidity Regularization")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
data = sorted(vLongRigidityScores.items())
x, y = zip(*data)
ax2.set_ylabel("Mean Rigidity Score for Carpal Bone Transforms")
ax2.plot(x, y, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.savefig(vOutPath + "/plot_Rigid2negJD.png", bbox_inches='tight')
plt.close()
'''

data = sorted(vLongLossHistory.items())
x, y = zip(*data)
fig = plt.plot(x, y, marker=".", markersize=10, color='tab:blue')
plt.ylabel("MSE Loss")
plt.xlabel("Number of Components Included")
plt.savefig(vOutPath + "/plot_Comps2Loss.png", bbox_inches='tight')
plt.close()

'''
data = sorted(vLongLossHistory.items())
x, y = zip(*data)
fig, ax1 = plt.subplots()
ax1.plot(x, y, marker=".", markersize=10, color='tab:blue')
ax1.set_ylabel("MSE Loss")
ax1.set_xlabel("Signal Gain of Rigidity Regularization")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
data = sorted(vLongRigidityScores.items())
x, y = zip(*data)
ax2.set_ylabel("Mean Rigidity Score for Carpal Bone Transforms")
ax2.plot(x, y, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.savefig(vOutPath + "/plot_Rigid2Loss.png", bbox_inches='tight')
plt.close()
'''

data = sorted(vLongRelativeDice.items())
x, y = zip(*data)
fig = plt.plot(x, y, marker=".", markersize=10, color='tab:blue')
plt.ylabel("Change in DICE After Registration")
plt.xlabel("Number of Components Included")
plt.title("Change in DICE by Num. Components")
plt.savefig(vOutPath + "/plot_Comps2DICE.png", bbox_inches='tight')
plt.close()

'''
data = sorted(vLongRelativeDice.items())
x, y = zip(*data)
fig, ax1 = plt.subplots()
ax1.plot(x, y, marker=".", markersize=10, color='tab:blue')
ax1.set_ylabel("Change in DICE After Registration")
ax1.set_xlabel("Signal Gain of Rigidity Regularization")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
data = sorted(vLongRigidityScores.items())
x, y = zip(*data)
ax2.set_ylabel("Mean Rigidity Score for Carpal Bone Transforms")
ax2.plot(x, y, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.savefig(vOutPath + "/plot_Rigid2DICE.png", bbox_inches='tight')
plt.close()
'''

data = sorted(vLongRigidityScores.items())
x, y = zip(*data)
fig = plt.plot(x, y, marker=".", markersize=10, color='tab:blue')
plt.ylabel("Avg. Rigidity Score After Registration")
plt.xlabel("Number of Components Included")
plt.title("Rigidity Score")
plt.savefig(vOutPath + "/plot_Comps2Rigidity.png", bbox_inches='tight')
plt.close()