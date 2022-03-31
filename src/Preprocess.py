import fnmatch

import SimpleITK as sitk
import numpy as np
import os
import argparse
import utilities as utils

# TODO This file should contain an Atlas-building function
# TODO This file should contain a "Warp Back to Source" function
# TODO This file should contain a Bias Field Correction function


def setTrueVoxelSize3D(fpFILENAME:str=None,fpIN:str="..\\images\\input",
                       fpOUT:str="..\\images\\test\\unpacked",
                       trueVoxelDims=None):
    if (fpFILENAME is not None):
        out = os.path.join(fpOUT, "unpacked", fpFILENAME)
    else:
        out = os.path.join(fpOUT, "unpacked")

    if not os.path.exists(out):
        os.makedirs(out)

    img = sitk.ReadImage(fpIN)

    if not (type(trueVoxelDims) is None):
        img.SetSpacing(trueVoxelDims)

    sitk.WriteImage(img, os.path.join(out, fpFILENAME + ".nii"))

    return out


def getFramesFrom4D(fpFILENAME:str=None,fpIN:str="..\\images\\input",
                    fpOUT:str="..\\images\\test\\unpacked",
                    trueVoxelDims=None):
    '''
    :param fpFILENAME: If not None, a string to make a subdirectory for this .nii.
    :param fpIN: File path (as string) to a 4D NIfTI-1 file.
    :param fpOUT: File path (as string) to an output folder.
    :param trueVoxelDims: The true voxel dimensions of the image if different from those in the NIfTI header.
    :return:
    '''
    img = sitk.ReadImage(fpIN)
    if (fpFILENAME is not None):
        out = os.path.join(fpOUT,"unpacked",fpFILENAME)
    else:
        out = os.path.join(fpOUT,"unpacked")

    if not os.path.exists(out):
        os.makedirs(out)

    data = sitk.GetArrayFromImage(img)
    numFrames = img.GetSize()[3]

    # Check to See if the
    if img.GetDimension() == 3:
        print("This function should only be called on 4D data.")
        exit(3)
    else:
        for i in range(0,numFrames):
            fName = "frame_" + str(i) + ".nii"
            newImg = sitk.GetImageFromArray(data[i,:,:,:])
            newImg.SetOrigin(img.GetOrigin())
            newImg.SetOrigin(np.eye(3).flatten())
                # The data coming in is set to identity anyway in this set but
                # this would constitute a tricky thing to encode if this function
                # is ever applied to a different input set.  Just so you future
                # users are aware, you'll need to take the source's 4D direction
                # and project it to a 3D direction.
            if not (type(trueVoxelDims) is None):
                newImg.SetSpacing(trueVoxelDims)
            sitk.WriteImage(newImg,os.path.join(out,fName))

        return out, numFrames


def _getResampledImageDims(oldImgDims: list,oldVoxDims: list,newVoxDims):
    aspect = np.divide(oldVoxDims,newVoxDims)
    return np.round(np.dot(oldImgDims,np.diag(aspect)))


def _getDimString(dims: list):
    temp = ""
    for i in range(len(dims)):
        temp += str(dims[i])
        if i < len(dims)-1:
            temp += "x"
    return temp


def getIsotropicResampling(fpFILENAME:str=None,fpIN="..\\images\\input\\unpacked",
                           fpOUT="..\\images\\test\\iso",isoVoxelDim: float=0.5):
    if type(isoVoxelDim) is None:
        isoVoxelDim = 0.5

    if (fpFILENAME is not None):
        out = os.path.join(fpOUT,"iso",fpFILENAME)
    else:
        out = os.path.join(fpOUT,"iso")

    if not os.path.exists(out):
        os.makedirs(out)

    for filename in os.listdir(fpIN):
        iImg = os.path.join(fpIN,filename)
        oImg = os.path.join(out,filename)

        img = sitk.ReadImage(os.path.join(fpIN,filename))
        iDim = img.GetSpacing()
        oDim = _getResampledImageDims(img.GetSize(),iDim,isoVoxelDim)
        oDim = _getDimString(oDim)
        oVox = _getDimString([isoVoxelDim]*3)

        os.system("c3d " + iImg + " --resample " + oDim + " --spacing " + oVox + "mm --o " + oImg)

    return out


def getBiasFieldCorrection(fpMASK: str=None,fpFILENAME:str=None,fpIN="..\\images\\input\\iso",
                           fpOUT="..\\images\\test\\unbiased"):
    if (fpFILENAME is not None):
        out = os.path.join(fpOUT,"unbiased",fpFILENAME)
    else:
        out = os.path.join(fpOUT,"unbiased")

    if not os.path.exists(out):
        os.makedirs(out)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    for filename in os.listdir(fpIN):
        if fnmatch.fnmatch(filename,"*.nii"):
            print("Now unbiasing ",filename)
            if fpMASK is not None:
                img = sitk.ReadImage(os.path.join(fpIN, filename))
                mask = sitk.ReadImage(os.path.join(fpMASK, filename), sitk.sitkUInt8)
            else:
                img = sitk.Normalize(sitk.ReadImage(os.path.join(fpIN, filename)))
                mask = sitk.OtsuThreshold(img, 0, 1, 5)
            corrected = corrector.Execute(img,mask)
            sitk.WriteImage(corrected,os.path.join(out,filename))
        else:
            img = sitk.ReadImage(os.path.join(fpIN, filename))
            sitk.WriteImage(img,os.path.join(out,filename))

    return out


def _getCropDictionary(dLow: int, dHigh: int, hLow: int, hHigh: int, wLow: int, wHigh: int):
    return {'d':[dLow,dHigh],'h':[hLow,hHigh],'w':[wLow,wHigh]}


def getCroppedROI(crop: dict,fpFILENAME:str=None, numFrames=20,fpIN="..\\images\\input\\unbiased",
                  fpOUT="..\\images\\test\\cropped"):
    if (fpFILENAME is not None):
        out = os.path.join(fpOUT,"crop",fpFILENAME)
    else:
        out = os.path.join(fpOUT,"crop")

    if not os.path.exists(out):
        os.makedirs(out)

    for filename in os.listdir(fpIN):
        filePath = os.path.join(fpIN, filename)
        img = sitk.ReadImage(filePath)
        img = sitk.GetArrayFromImage(img)
        cropImg = img[crop['d'][0]:crop['d'][1],
                      crop['h'][0]:crop['h'][1],
                      crop['w'][0]:crop['w'][1]]
        cropImg = sitk.GetImageFromArray(cropImg, isVector=False)
        sitk.WriteImage(cropImg, os.path.join(out, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3D and 4D .nii files for use in image processing.")
    parser.add_argument("-n","--name",metavar="Name",help="A reference name for the file, used for file system and "
                                                          "labelling.")
    parser.add_argument("inFilePath",help="Input file path pointing to a 3D or 4D .nii file.")
    parser.add_argument("outFilePath",help="Output file path pointing to a directory.")
    parser.add_argument('-o','--OriginalVoxelDims',type=float,nargs=3,help="Original Voxel Dimensions in the order ("
                                                                           "H,W,D)",
                        metavar=("H","W","D"))
    parser.add_argument("-u","--unpack",action='store_true',help="Pass to unpack a 4D tensor into consecutive 3D "
                                                                 "frames.")
    parser.add_argument("-i","--isotropic",type=float,nargs='?',const=0.5,
                        help="Pass to resample volume to an isotropic voxel size. "
                             "If a float value follows the argument, voxels will be "
                             "set to that length on each size",
                        metavar="sideLen")
    parser.add_argument('-b','--biasFieldCorrection',help="Pass to perform bias field correction.",
                        metavar="mask_file_path",type=str,nargs='?',const="Otsu")
    parser.add_argument('-c','--crop',type=int,nargs=6,
                        help="Pass along with six integers for the region of interest in the sequence. "
                             "Order for arguments is \n [depth_low, depth_high, height_low, height_high, "
                             "width_low, width_high].",
                        metavar=("D_low","D_hi","H_low","H_hi","W_low","W_hi"))

    args = parser.parse_args()

    if args.unpack:
        print("Unpacking frames....")
        fpUnpack, numFrames = getFramesFrom4D(fpFILENAME=args.name,
                                              fpIN=args.inFilePath, fpOUT=args.outFilePath,
                                              trueVoxelDims=args.OriginalVoxelDims)
    else:
        if args.OriginalVoxelDims:
            print("Setting original voxel dimensions...")
            fpUnpack = setTrueVoxelSize3D(fpFILENAME=args.name,
                                          fpIN=args.inFilePath, fpOUT=args.outFilePath,
                                          trueVoxelDims=args.OriginalVoxelDims)
            numFrames = 1
        else:
            fpUnpack = args.inFilePath
            numFrames = 1

    if args.isotropic:
        print("Resampling to isotropic voxel size...")
        fpIso = getIsotropicResampling(fpFILENAME=args.name,
                                       fpIN=fpUnpack, fpOUT=args.outFilePath,
                                       isoVoxelDim=args.isotropic)
    else:
        fpIso = fpUnpack

    if args.biasFieldCorrection == "Otsu":
        print("Applying bias field correction...")
        fpUnbiased = getBiasFieldCorrection(fpMASK=None,fpFILENAME=args.name,
                                            fpIN=fpIso, fpOUT=args.outFilePath)
    else:
        fpUnbiased = getBiasFieldCorrection(fpMASK=args.biasFieldCorrection, fpFILENAME=args.name,
                                            fpIN=fpIso, fpOUT=args.outFilePath)

    if args.crop:
        print("Cropping images to the region of interest...")
        getCroppedROI(crop=_getCropDictionary(*args.crop),fpFILENAME=args.name,
                      numFrames=numFrames,
                      fpIN=fpUnbiased, fpOUT=args.outFilePath)









