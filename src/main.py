'''
Russell Wustenberg, rw2873
Code for the Thesis
"Carpal Bone Rigid-Body Kinematic Estimation
by Log-Euclidean Registration"
Submitted in partial fulfillment for the requirements
of the master's of computer science degree at
New York University's Tandon School of Engineering.
Supervisor: Dr. Guido Gerig of the Visual Imaging and Data Analysis Lab.
'''

import os
import random
import numpy as np
import torch
import Evaluation as eval
import Kinematics as K
import Registration as R

if __name__ == "__main__":
    '''
    This is a driver function for the registration pipeline.  The user should first set all paths
    and hyper-parameters using the variables preceeding the loop.  Hyper-parameters should be passed
    as an iterable (even if only one is desired).  This format can quickly be used to output several
    grid search results over the passed parameters.
    
    A word of warning: the output of each run of the model is quite large.  Six runs was sufficient
    to use 100% of my secondary storage.  Also, it should be noted that around 5 GB of vRAM are required
    to run the model on the current data set.  This can be reduced by making the cropped images smaller.
    
    In order to run this program, your file structure should look as follows
    Project Root
        Images
            input
                rave
                    [frame and segmentation .nii or .nii.gz files]
            results // Created and populated at run time
                Model Labeled Folder // indexed by hyperparameter values
                    Model Labeled folder 
                        // Designed to be dragged-and-dropped elsewhere for record keeping
                        imgErrors           // TP FP FN image showing IoU stats for this model
                        imgFloat            // Slices of the floating image as a .png
                        imgFloatOverRef     // Untouched segmentations from original floating image over target
                        imgRef              // Slices of the reference image as a .png
                        imgRefOverRef       // The Ground Truth segmentations laid over the reference image
                        imgWarped           // Slices of the warped image as a .png
                        imgWarpOverRef      // The Warped (propagated) segmentations laid over the target image.
                        meshes              // VTK static meshes of the bones, viewable in Paraview or VTK
                        plots               // Results of Kinematics.py, which compare the C_T^i and warp field Phi
                            estimations
                            modelAccuray_Centroids
                                // The accuracy of the centroid placement when Phi, the warp field is
                                // applied to each component segmentation
                            modelAccuracy_EstimatedRotations
                                // The accuracy of the rotations estimated when each individual component's estimated
                                // transform is applied to the respective component.
                            modelAccuracy_EstimatedTranslations
                                // The accuracy of the centroid placement when each individual component's transform
                                // is applied to the respective component.
                            modelAccuracy_Rotations
                                // The accuracy of the rotations estimated when Phi, the warp field is
                                // applied to each component segmenation
                        summary_metrics.png // The following are longitudinal evaluation charts for the model
                        summary_MSE.png
                        summaryAccuracy.png
                        summaryComponents_Accuracy.png
                        summaryComponents_DICE.png
                        summaryComponents_Precision.png
                        summaryComponents_Recall.png
                        summaryDICE.png
                        summaryLoss.png
                        summaryNJD.png
                        summaryPrecision.png
                        summaryRecall.png
                        summaryRigid.png
                frame_K_to_frame_(K+/-1)
                    // The following are individual and batch files for various statistics on each frame registration
                    component_N_cfsn.npz    //confusion matrix for component N
                    component_transforms_final_euclidean.npz    // The estimated component transforms in SE(3)
                    component_transforms_final_log.npz          // The estimated component transforms in the tangent space
                    confusionMatrix.npz     //general confusion matrix for the registration on all bones
                    float_seg_binary.nii    //The unmodified carpal segmentations in 3D
                    imgComponent_N_FN.nii   // The Nth component's 3D False Negatives
                    imgComponent_N_FP.nii   // The Nth component's 3D False Positives
                    imgComponent_N_TN.nii   // The Nth component's 3D True Negatives
                    imgComponent_N_TP.nii   // The Nth component's 3D True Positives
                    imgFloat.nii            // The floating Image
                    imgFN.nii               // Total False Negatives for the registration
                    imgFP.nii               // All False Positives for the registration
                    imgReference.nii        // Reference image
                    imgTN.nii               // All true negatives for the registration (not very helpful)
                    imgTP.nii               // All true positives for the registration
                    imgWarped.nii           // The warped image from the registration
                    LEPT_frame_K_to_frame_(K+/-1).nii   // The warp field estimated for the registration
                    log_jabocian_determinants.nii   // The log jacobian determinant image for the registration
                    model_loss.npz          // The loss statistics for this pariwise registration
                    model_results.npz       // All result metrics for this registration as a dictionary.
                    res_transforms.txt      // Text file showing resulting transformations and applicable statistics. 
                                            // (The above file is very useful!)
                    target_seg_binary.nii   // Target image segmentations as a binary image.
                    warped_seg_N.nii        // Warped component segmentation for the Nth component. Used in next round
                                            // Of registration if frame K+/-1 is not a terminal frame.
                    warped_seg_binary.nii   // All components for the warped image composed into a single binary image.             
        src
            Evaluation.py
            Kinematics.py
            main.py
            Polyrigid.py
            Preprocess.py
            Registration.py
            utilities.py
        venv
        .gitignore
        requirements.txt
        
    The working directory should be set to the source folder, and use main.py to drive the pipeline.
    
    To trace the code, begin in main.py.
    Main.py calls the following sequence:
        Registration.py
            This file contains the log-Euclidean polyrigid registration framework.
            It was consturcted using PyTorch and uses an Adam optimizer.  As such,
            it relies upon the Polyrigid class in Polyrigid.py to perform gradient descent.
        Evaluation.py
            This file uses the results of the pair-wise registration to calculate metrics for
            sequence-scale evaluation plots.  The source files will be contained within each pairwise
            registration folder under the model's results folder.
        Kinematics.py
            This file takes in the series of estimated component transformations, composes them and
            compares the results of their application to their respective components to the same component's
            behavior when warped using the displacement vector field estimated using the LEP registration framework.
    '''
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    vStopLoss = 1e-5
    vLearningRate = 0.01
    vMaxItrs = 200
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
    vAlpha = [1.0] # Signal strength for MSE loss
    vEta = [1.0]  # Signal strength for DICE loss
    vBeta = [1.0]  # Signal strength for smoothness regularization
    vGamma = [0.0]  # Signal strength for negative JD regularization
    vDelta = [1.0] # Signal strength for rigidity regularization
    vEpsilon = [1.0] # Translation Regularization
    vZeta = [0.5] # Component weighting parameter


    if vEndFrame_hi == -1:
        vEndFrame_hi = vNumFrames - 1
    if vEndFrame_low == -1:
        vEndFrame_low = 0

    for a in vAlpha:
        for b in vBeta:
            for g in vGamma:
                for d in vDelta:
                    for zeta in vZeta:
                        for e in vEpsilon:
                            for h in vEta:
                                alpha = abs(a)
                                beta = abs(b)
                                gamma = abs(g)
                                delta = abs(d)
                                epsilon = abs(e)
                                eta = abs(h)
                                div = alpha + beta + gamma + delta + epsilon + eta

                                if div <= 0.0:
                                    print("Weighting of hyper parameters requires a positive value!")
                                    exit(1)

                                alpha = abs(alpha)/div
                                beta = abs(beta)/div
                                gamma = abs(gamma)/div
                                delta = abs(delta)/div
                                epsilon = abs(epsilon)/div
                                eta = abs(eta)/div

                                def _formatLabelVal(val: float):
                                    out = round(val,4)
                                    first, last = str(out).split('.')
                                    if len(last) < 4:
                                        last = last + '000'
                                    return f"{int(first):02}_{last[0:4]}"

                                label = f"DICE_{_formatLabelVal(eta)}_MSE_{_formatLabelVal(alpha)}" \
                                        f"_smooth_{_formatLabelVal(beta)}_nJD_{_formatLabelVal(gamma)}"\
                                        f"_rigid_{_formatLabelVal(delta)}_trans_{_formatLabelVal(epsilon)}"\
                                        f"_weight_{_formatLabelVal(zeta)}"

                                fpOut = os.path.join(vOutFile,label)
                                if not os.path.exists(fpOut):
                                    os.mkdir(fpOut)

                                for source in range(vStartFrame, vEndFrame_hi, vStride):
                                    if (source + vStride > vNumFrames - 1):
                                        print("Cannot register outside of frame sequence.")
                                        break
                                    target = source + vStride
                                    vInFrame_Float = "frame_" + str(source)
                                    vInFrame_Target = "frame_" + str(target)

                                    R.estimateKinematics(vInFolder, vInFrame_Float_Prefix, vInSeg_Float_Prefix,
                                                       vInFrame_Float,vInFrame_Target, fpOut, vNumComponents,
                                                       vMaxItrs, vLearningRate,vStopLoss, vUpdateRate,
                                                       alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                                       epsilon=epsilon,zeta=zeta,eta=eta)

                                for source in range(vStartFrame, vEndFrame_low, -vStride):
                                    if (source - vStride < 0):
                                        print("Cannot register outside of frame sequence.")
                                        break
                                    target = source - vStride
                                    vInFrame_Float = "frame_" + str(source)
                                    vInFrame_Target = "frame_" + str(target)

                                    R.estimateKinematics(vInFolder, vInFrame_Float_Prefix, vInSeg_Float_Prefix,
                                                       vInFrame_Float,vInFrame_Target, fpOut, vNumComponents,
                                                       vMaxItrs, vLearningRate,vStopLoss, vUpdateRate,
                                                       alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                                       epsilon=epsilon, zeta=zeta, eta=eta)
                                print("Generating evaluation metrics...")
                                eval.getEvaluationPlots(label)
                                print("Generating static meshes...")
                                eval.getStaticMeshes(params=label)
                                print("Estimating Kinematics...")
                                K.getPlots(label=label)
                                print("Model complete.")

    print("All model runs complete.")