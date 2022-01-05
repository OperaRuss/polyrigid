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

def makeHomogenous(point: list):
    temp = []
    if (type(point) == np.ndarray):
        temp = point.tolist()
        temp.append(1)
        temp = np.array(temp)
    else:
        temp = point
        temp.append(1)
    return temp

def makeCartesian(point: list):
    temp = []
    if(type(point) == np.ndarray):
        tempPoint = point.tolist()
        temp = [x / tempPoint[-1] for x in tempPoint[:-1]]
    else:
        temp = [x / point[-1] for x in point[:-1]]
    return temp

def getRadiansFromDegrees(degrees):
    '''
    Converts degrees into radians.

    :param degrees: Can be of type float or list of floats.
    :return: Returns a single float or list of floats in the same order as the input argument.
    '''
    if type(degrees) == float:
        return degrees * (np.pi / 180.0)
    elif type(degrees) == list:
        temp = []
        for degree in degrees:
            temp.append(degree * (np.pi / 180.0))
        return temp
    else:
        print("This function takes in arguments of type float or list of floats.")
        return None

def getRotationMatrixFromDegrees(radians, dimensions: int=2):
    '''
    This function takes in angles of rotation and outputs a rotation matrix of the same dimension specified.

    :param radians: Argument for angles of X, Y, Z rotation angles relative to identity. Can be float or list of floats in the format (X, Y, Z).
    :param dimensions: Argument indicating the number of dimensions for the current coordinate system.
    :return: Returns a right-hand rotation matrix or list of three right-hand rotation matricies (X, Y, Z), each of the specified dimension.
    '''
    if dimensions not in [2,3]:
        print("Current implementation is only specified for dimensions 2 and 3.")
        return -1

    outRotationMatricies = []

    try:
        # Assumes that all three dimensions are receiving the same angle of rotation, ie. X=Y=Z.
        if type(radians) == float:
            if(dimensions == 3):
                outRotationX = [[1, 0,                0              ],
                                [0, np.cos(radians), -np.sin(radians)],
                                [0, np.sin(radians),  np.cos(radians)]]
                outRotationMatricies.append(outRotationX)

                outRotationY = [[np.cos(radians),   0,  np.sin(radians)],
                                [0,                 1,  0              ],
                                [-np.sin(radians),  0,  np.cos(radians)]]
                outRotationMatricies.append(outRotationY)

                outRotationZ = [[np.cos(radians),  -np.sin(radians), 0],
                                [np.sin(radians),   np.cos(radians), 0],
                                [0,                 0,               1]]
                outRotationMatricies.append(outRotationZ)

                return np.array(outRotationMatricies, dtype=np.float64)
            else:
                outRotation2D = [[np.cos(radians), -np.sin(radians)],
                                 [np.sin(radians),  np.cos(radians)]]
                return np.array(outRotation2D, dtype=np.float64)

        elif type(radians) == list:
            if len(radians) != 3:
                print("Input for using a list of floats must be of format (X, Y, Z).")
                return -1
            else:
                outRotationX = [[1, 0,                0              ],
                                [0, np.cos(radians[0]), -np.sin(radians[0])],
                                [0, np.sin(radians[0]),  np.cos(radians[0])]]
                outRotationMatricies.append(outRotationX)

                outRotationY = [[np.cos(radians[1]),   0,  np.sin(radians[1])],
                                [0,                 1,  0              ],
                                [-np.sin(radians[1]),  0,  np.cos(radians[1])]]
                outRotationMatricies.append(outRotationY)

                outRotationZ = [[np.cos(radians[2]),  -np.sin(radians[2]), 0],
                                [np.sin(radians[2]),   np.cos(radians[2]), 0],
                                [0,                 0,               1]]
                outRotationMatricies.append(outRotationZ)

                return np.array(outRotationMatricies, dtype=np.float64)
        else:
            print("Input for 'radians' must be either a single float or list of 3 floats (X, Y, Z).")
            return -1
    except:
        print(  "Input should be of type float when angle of rotation is the same for X, " +
                "Y and Z or list of floats in the order (X, Y, Z) when different." )
        return -1