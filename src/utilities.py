import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def showNDA_InEditor_BW(img: np.ndarray, title: str=""):
    try:
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except:
        if len(img.shape) != 2:
           print("Ensure that you are passing an image slice and not a volume.")
        else:
            print("Something is not quite right about your image.")

def showSITK_InEditor_BW(img: sitk.Image, title: str=""):
    try:
        ndarr = sitk.GetArrayViewFromImage(img)
        plt.imshow(ndarr, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except:
        if img.GetDimension() != 2:
           print("Ensure that you are passing an image slice and not a volume.")
        else:
            print("Something is not quite right about your image.")

def showInEditor_RGB(img: sitk.Image, title: str=""):
    try:
        ndarr = sitk.GetArrayViewFromImage(img)
        plt.imshow(ndarr)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except:
        if img.GetDimension() != 2:
           print("Ensure that you are passing an image slice and not a volume.")
        else:
            print("Something is not quite right about your image.")

def convertSITK2numpy(img: sitk.Image):
    nda = sitk.GetArrayFromImage(img)
    return nda

def saveSITKimage(img: sitk.Image, fileName:str, outputFile: str="./images/results"):
    sitk.WriteImage(img, os.path.join(outputFile, fileName))