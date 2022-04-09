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
              componentWeights: dict, learningRate: float=0.005):
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
vTestHistory = {}
vNumComponents = 15
vInFolder = "../images/input/rave/"
vInFrame_Float = "iso_frame_6"
vInFrame_Target = "iso_frame_5"
vOutFile = "../images/results/"
vDate = "20220408"
vLambda = [1.0]
vAlpha = [0.0]

for pMaxItrs in vMaxItrs:
    for pLamda in vLambda:
        for pAlpha in vAlpha:
            for pStepSize in vStep_Size:
                if not os.path.exists(vOutFile):
                    os.makedirs(vOutFile)

                if not os.path.exists(os.path.join(vOutFile,vDate)):
                    os.makedirs(os.path.join(vOutFile,vDate))

                # SECTION 2: READING IN IMAGE DATA
                imgSITK_moving = sitk.ReadImage(vInFolder+vInFrame_Float+".nii")
                imgMoving = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_moving)[23:53,60:160,50:150])
                tImgMoving = torch.tensor(imgMoving, dtype=torch.float32)
                tImgMoving = tImgMoving.unsqueeze(0).unsqueeze(0).cuda()

                imgSITK_target = sitk.ReadImage(vInFolder+vInFrame_Target+".nii")
                imgTarget = utils.normalizeImage(sitk.GetArrayFromImage(imgSITK_target)[23:53,60:160,50:150])
                tImgTarget = torch.tensor(imgTarget, dtype=torch.float32)
                tImgTarget = tImgTarget.unsqueeze(0).unsqueeze(0).cuda()

                assert imgMoving.shape == imgTarget.shape, \
                    "Images must be of the same dimensions. Got %s != %s"%(imgMoving.shape,imgTarget.shape)

                aComponentSegmentations_Float = {}
                aComponentWeightValues = {}
                aComponentSegmentations_Target = {}

                for i in range(1,vNumComponents):
                    temp = sitk.ReadImage(vInFolder+vInFrame_Float+"_seg_"+str(i)+".nii.gz")
                    aComponentSegmentations_Float[i] = sitk.GetArrayFromImage(temp)[23:53,60:160,50:150]

                    temp = sitk.ReadImage(vInFolder+vInFrame_Target+"_seg_"+str(i)+".nii.gz")
                    aComponentSegmentations_Target[i] = sitk.GetArrayFromImage(temp)[23:53,60:160,50:150]

                for idx,img in aComponentSegmentations_Float.items():
                    aComponentSegmentations_Float[idx] = utils.normalizeImage(img)

                for idx, img in aComponentSegmentations_Float.items():
                    aComponentWeightValues[idx] = 1/2

                model, optimizer = get_model(tImgMoving,aComponentSegmentations_Float,
                                             aComponentWeightValues,learningRate=pStepSize)

                tComponentSegImg_Target = np.zeros(model.mImageDimensions)
                for img in aComponentSegmentations_Target.values():
                    tComponentSegImg_Target += img
                tComponentSegImg_Target = torch.tensor(tComponentSegImg_Target, dtype=torch.float32).cuda()

                # STEP 7: ITERATION TOWARDS OPTIMIZATION
                print("Running with Smoothness Parameter " + str(pLamda)
                      + " and JD Regularization parameter " + str(1.0 - pLamda))
                print("Running model for maximum of " + str(pMaxItrs) + " iterations.")
                for itr in range(pMaxItrs):
                    optimizer.zero_grad()

                    tImgWarped = model.forward()
                    tImgFixed = 1.0 * tImgTarget

                    reg_loss = utils._getMetricMSE(tImgWarped, tImgFixed)
                    #reg_loss = -utils._getMetricNCC(tImgWarped, tImgFixed)
                    smooth_loss = pAlpha * pLamda * utils._loss_Smooth(model.tDisplacementField)
                    jdet_loss = pAlpha * (1-pLamda) * utils._loss_JDet(model.tDisplacementField)
                    rigid_loss = model._getLoss_Rigidity()
                    loss = reg_loss + 0.3 * rigid_loss #+ smooth_loss + jdet_loss

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
                        if abs((loss + vItrHistory[itr-1] + vItrHistory[itr-2 ]) / 3.0) < 1e-5:
                            print("Model growth slowed at iteration", itr, "with loss score ", loss)
                            break

                # SECTION 8: MODEL OUTPUT
                # The below sections produce results from the run of the model and put them in the specified
                # output folder.  Starts with establishing a folder based on the current model parameters then
                # outputs all output into that folder.

                str_smooth = str(pLamda).replace('.','_')
                str_JD = str(1.0-pLamda).replace('.','_')
                str_itrs = str(pMaxItrs)
                str_lr = str(pStepSize).replace('.','_')
                str_alph = str(pAlpha).replace('.','_')

                vSubFolder = "/obj_" + str(vNumComponents) + "_itrs_" + str_itrs + "_Alph_" + str_alph + "_sm_" + str_smooth + "_JD_" + str_JD + "_lr_" + str_lr
                vOutPath = vOutFile + vDate + vSubFolder

                if not os.path.exists(vOutPath):
                    os.makedirs(vOutPath)

                tImgWarped_Seg = F.grid_sample(model.tImgSegmentation,
                                               model.tDisplacementField,
                                               mode='nearest', padding_mode='zeros', align_corners=False)

                utils.pltImage(tImgMoving.detach().squeeze().cpu().numpy()[tImgMoving.shape[2]//2, :, :],
                               title="Moving Image",toShow=False,toSave=True,outPath=vOutPath,outFile="/img_moving.png")

                plt.imshow(tImgMoving.detach().squeeze().cpu().numpy()[tImgMoving.shape[2]//2, :, :],cmap='gray')
                plt.title("Moving Image")
                plt.axis('off')
                plt.savefig(vOutPath + "/img_moving.png", bbox_inches='tight')
                #plt.show()
                plt.close()

                plt.imshow(tImgWarped.detach().squeeze().cpu().numpy()[tImgWarped.shape[2]//2, :, :],cmap='gray')
                plt.title("Warped Image")
                plt.axis('off')
                plt.savefig(vOutPath + "/img_warped.png", bbox_inches='tight')
                #plt.show()
                plt.close()

                plt.imshow(tImgTarget.squeeze().cpu().numpy()[tImgTarget.shape[2]//2, :, :],cmap='gray')
                plt.title("Target Image")
                plt.axis('off')
                plt.savefig(vOutPath + "/img_target.png", bbox_inches='tight')
                #plt.show()
                plt.close()

                data = sorted(vItrHistory.items())
                x,y = zip(*data)
                fig = plt.plot(x,y,marker =".",markersize=10)
                plt.title("Loss by Iterations")
                #plt.ylim(-1.0,0.0)
                plt.savefig(vOutPath + "/plot_NCC.png", bbox_inches='tight')
                #plt.show()
                plt.close()

                tImgWarped = \
                    sitk.GetImageFromArray(tImgWarped.detach().squeeze().cpu().numpy(),False)
                sitk.WriteImage(tImgWarped, vOutPath + "/nii_warped.nii")
                tImgWarped = sitk.GetArrayFromImage(tImgWarped)

                vDICE_Before = \
                    f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1,1),
                             model.tImgSegmentation.detach().cpu().numpy().reshape(-1,1),average='macro')
                vDICE_After = \
                    f1_score(tComponentSegImg_Target.detach().cpu().numpy().reshape(-1,1),
                             tImgWarped_Seg.detach().cpu().view((-1,1)).numpy(),average="macro")

                tComponentSegImg_Float = \
                    sitk.GetImageFromArray(model.tImgSegmentation.detach().cpu().numpy(),False)
                sitk.WriteImage(tComponentSegImg_Float, vOutPath + "/nii_float_seg.nii")
                tComponentSegImg_Float = sitk.GetArrayFromImage(tComponentSegImg_Float)

                tImgWarped_Seg = \
                    sitk.GetImageFromArray(tImgWarped_Seg.detach().squeeze().cpu().numpy(),False)
                sitk.WriteImage(tImgWarped_Seg, vOutPath + "/nii_warped_seg.nii")

                tComponentSegImg_Target = \
                    sitk.GetImageFromArray(tComponentSegImg_Target.detach().cpu().numpy(),False)
                sitk.WriteImage(tComponentSegImg_Target, vOutPath + "/nii_target_seg.nii")
                tComponentSegImg_Target = sitk.GetArrayFromImage(tComponentSegImg_Target)
        
                tDeterminantMap = utils.jacobian_determinant_3d(model.tDisplacementField)
                tDeterminantMap = tDeterminantMap.detach().squeeze().cpu().numpy()
                tDeterminantMap = tDeterminantMap

                sitkDisplacementField = sitk.GetImageFromArray(model.tDisplacementField.detach().squeeze().cpu().numpy(),
                                                               isVector=True)
                tJacDetVerification = sitk.DisplacementFieldJacobianDeterminant(sitkDisplacementField)
                npDeterminantMap = sitk.GetArrayFromImage(tJacDetVerification)

                vLim = int
                if abs(tDeterminantMap.min()) >= abs(tDeterminantMap.max()):
                    vLim = tDeterminantMap.min()
                else:
                    vLim = tDeterminantMap.max()

                for i in range(tDeterminantMap.shape[0]):
                    fig = plt.figure()
                    plt.title("LapIRN Determinant Map For Volume Slice "+str(i))
                    im = plt.imshow(tDeterminantMap[i,:,:])#,cmap='bwr_r')
                    #plt.clim(-vLim,vLim)
                    plt.axis('off')
                    fig.colorbar(im)
                    plt.savefig(vOutPath + "/img_LapIRN_det_small_slice_" + str(i) + ".png", bbox_inches='tight')
                    #plt.show()
                    plt.close()

                if abs(npDeterminantMap.min()) >= abs(npDeterminantMap.max()):
                    vLim = npDeterminantMap.min()
                else:
                    vLim = npDeterminantMap.max()

                for i in range(npDeterminantMap.shape[0]):
                    fig = plt.figure()
                    plt.title("SITK Determinant Map For Volume Slice "+str(i))
                    im = plt.imshow(npDeterminantMap[i,:,:])#,cmap='bwr_r')
                    #plt.clim(-vLim,vLim)
                    plt.axis('off')
                    fig.colorbar(im)
                    plt.savefig(vOutPath + "/img_sitk_det_small_slice_" + str(i) + ".png", bbox_inches='tight')
                    #plt.show()
                    plt.close()

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

                sitkDisplacementField = sitk.GetImageFromArray(
                    tTestDVF.detach().squeeze().cpu().numpy(),
                    isVector=True)
                tJacDetVerification = sitk.DisplacementFieldJacobianDeterminant(sitkDisplacementField)
                npDeterminantMap = sitk.GetArrayFromImage(tJacDetVerification)

                vLim = int
                if abs(tDeterminantMap.min()) >= abs(tDeterminantMap.max()):
                    vLim = tDeterminantMap.min()
                else:
                    vLim = tDeterminantMap.max()

                for i in range(tDeterminantMap.shape[0]):
                    fig = plt.figure()
                    plt.title("LapIRN Determinant Map For Volume Slice " + str(i))
                    im = plt.imshow(tDeterminantMap[i, :, :])  # ,cmap='bwr_r')
                    # plt.clim(-vLim,vLim)
                    plt.axis('off')
                    fig.colorbar(im)
                    plt.savefig(vOutPath + "/img_LapIRN_det_large_slice_" + str(i) + ".png", bbox_inches='tight')
                    # plt.show()
                    plt.close()

                if abs(npDeterminantMap.min()) >= abs(npDeterminantMap.max()):
                    vLim = npDeterminantMap.min()
                else:
                    vLim = npDeterminantMap.max()

                for i in range(npDeterminantMap.shape[0]):
                    fig = plt.figure()
                    plt.title("SITK Determinant Map For Volume Slice " + str(i))
                    im = plt.imshow(npDeterminantMap[i, :, :])  # ,cmap='bwr_r')
                    # plt.clim(-vLim,vLim)
                    plt.axis('off')
                    fig.colorbar(im)
                    plt.savefig(vOutPath + "/img_sitk_det_large_slice_" + str(i) + ".png", bbox_inches='tight')
                    # plt.show()
                    plt.close()

                vNumNeg_LapIRN = (tDeterminantMap <= 0.0).sum()
                vNumNeg_SITK = (npDeterminantMap <= 0.0).sum()

                tDeterminantMap = sitk.GetImageFromArray(tDeterminantMap,False)
                sitk.WriteImage(tDeterminantMap, vOutPath + "/nii_determinant.nii")

                with open(vOutPath + "/res_transforms.txt", "w") as out:
                    print(f"MaxIterations: {pMaxItrs}",file=out)
                    print(f"Learning Rate: {pStepSize}",file=out)
                    print(f"Smoothness Parameter: {pAlpha*pLamda}",file=out)
                    print(f"Jacobian Regularization Parameter: {pAlpha*(1.0-pLamda)}",file=out)
                    print(f"DICE score before registration: {vDICE_Before:.4f}",file=out)
                    print(f"DICE score after registration: {vDICE_After:.4f}",file=out)
                    print(f"Target Loss: {vStop_Loss:.4f}",file=out)
                    print(f"Loss achieved: {loss:.4f}",file=out)
                    print(f"Percentage of Jacobian determinants negative by LapIRN: "+
                          f"{(vNumNeg_LapIRN/(np.prod(model.mImageDimensions))*100):.2f}%",file=out)
                    print(f"Percentage of Jacobian determinants negative by SITK: " +
                          f"{(vNumNeg_LapIRN / (np.prod(model.mImageDimensions)) * 100):.2f}%", file=out)
                    print("Final parameter Estimations:\n",file=out)
                    for i in range(0,vNumComponents-1):
                        aCompTransforms = model._getLogComponentTransforms()
                        print("Component transformation "+str(i),file=out)
                        transform = torch.matrix_exp(torch.reshape(aCompTransforms[i],
                                                                   (model.mNDims+1,model.mNDims+1)))
                        transform = transform[0:3,0:3]
                        RRT = torch.sub(torch.matmul(transform, transform.T),torch.eye(3,device='cuda'))
                        RTR = torch.sub(torch.matmul(transform.T,transform),torch.eye(3,device='cuda'))
                        Rdet = torch.det(transform) - 1.0
                        rigidity = torch.frobenius_norm(RRT) + torch.frobenius_norm(RTR) + Rdet
                        print("Rigidity Score: ",rigidity.item(),file=out)
                        print(torch.matrix_exp(torch.reshape(aCompTransforms[i],
                                                             (model.mNDims+1,model.mNDims+1))),file=out)
