'''
Russell Wustenberg, rw2873
polyrigid refactor
start date 20220220
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