import torch
import numpy as np
import SimpleITK as sitk
import torchmaxflow as tmf
import os
import re


def PreProcessing_MaxFlow(fpIN, fpOUT):
    # Uncomment for preprocessing applied to input data
    inPrefix = "../images/input/rave"
    outPrefix = "../images/input/rave/em_"
    patternFrames = 'iso_frame_([0-9]|1[0-9]).nii.gz'
    patternSegs = '_seg_(.*)\.nii\.gz'
    regexFrames = re.compile(patternFrames)
    aFrames = []

    for file in os.listdir(fpIN):
        if regexFrames.match(file):
            aFrames.append(file)

    for frame in aFrames:
        # Get names of all segmentations
        niiName, extGZ = os.path.splitext(frame) # os.path.basename(os.path.join(inDir,frame))
        basename, extNii = os.path.splitext(niiName)
        aSegs = []
        regexSegs = re.compile(basename+patternSegs)
        i = 0
        for file in os.listdir(inPrefix):
            if regexSegs.match(file):
                aSegs.append(file)

        imgFrame = sitk.ReadImage(os.path.join(inPrefix, frame),sitk.sitkFloat32)
        npFrame = sitk.GetArrayFromImage(imgFrame)
        for seg in aSegs:
            file1, ext1 = os.path.splitext(seg)
            fileBase, ext2 = os.path.splitext(file1)
            outName = fileBase.removeprefix('iso_') + ext2 + ext1

            imgSeg = sitk.ReadImage(os.path.join(inPrefix, seg),sitk.sitkFloat32)
            npSeg = sitk.GetArrayFromImage(imgSeg)

            fP = 0.5 + (npSeg - 0.5) * 0.8
            bP = (1.0 - fP)
            Prob = np.asarray([bP,fP])
            lam = 1.25
            sig = 1.0

            tImg = torch.from_numpy(npFrame).unsqueeze(0).unsqueeze(0)
            tLabel = torch.from_numpy(Prob).unsqueeze(0)

            result = tmf.maxflow(tImg, tLabel, lam, sig).squeeze().cpu().numpy()

            out = sitk.GetImageFromArray(np.multiply(result,npSeg))

            out.CopyInformation(imgSeg)
            sitk.WriteImage(out, outPrefix + outName)

def MaxFlow(fpIN,fpOUT):
    patternSegs = 'warped_seg_(.*)\.nii'
    regexSegs = re.compile(patternSegs)
    aSegs = []

    for file in os.listdir(fpIN):
        if regexSegs.match(file):
            aSegs.append(file)

        imgSeg = sitk.ReadImage(os.path.join(fpIN, file), sitk.sitkFloat32)
        npFrame = sitk.GetArrayFromImage(imgSeg)
        for seg in aSegs:
            fileBase, ext1 = os.path.splitext(seg)
            outName = fileBase + ext1

            imgSeg = sitk.ReadImage(os.path.join(fpIN, seg), sitk.sitkFloat32)
            npSeg = sitk.GetArrayFromImage(imgSeg)

            fP = 0.5 + (npSeg - 0.5) * 0.8
            bP = (1.0 - fP)
            Prob = np.asarray([bP, fP])
            lam = 1.25
            sig = 1.0

            tImg = torch.from_numpy(npFrame).unsqueeze(0).unsqueeze(0)
            tLabel = torch.from_numpy(Prob).unsqueeze(0)

            result = tmf.maxflow(tImg, tLabel, lam, sig).squeeze().cpu().numpy()

            out = sitk.GetImageFromArray(np.multiply(result, npSeg))

            out.CopyInformation(imgSeg)
            sitk.WriteImage(out, outPrefix + outName)
