import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def showNDA_InEditor_BW(img: np.ndarray, title: str=""):
    '''
    Basic 'show' function for a numpy type array.

    :param img: 2- or 3-dimensional numpy array.
    :param title: If you want to label the image, include a string.
    :return:
    '''
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
    '''
    Basic display function for an SITK type image.  Sometimes does not work with
    my IDE but does not thrown an error.

    :param img: SimpleITK Image.
    :param title: Title, if one is wished.
    :return:
    '''
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

def convertSITK2numpy(img: sitk.Image):
    '''
    :param img: A SimpleITK Image
    :return: Returns the contents of the SITK image as a numpy array.  Discards header and affine.
    '''
    nda = sitk.GetArrayFromImage(img)
    return nda

def convertNumpy2SITK(img: np.ndarray):
    '''
    :param img: A 2- or 3-dimensional numpy array to be converted.
    :return: Returns a SimpleITK image.  Assumes affine transform is identity.
    '''
    img = sitk.GetImageFromArray(img)
    return img

def saveSITKimage(img: sitk.Image, fileName:str, outputFile: str="../images/results"):
    sitk.WriteImage(img, os.path.join(outputFile, fileName))

def saveNPimage_2D_BW(img: np.ndarray, fileName:str,outputFile: str="../images/results"):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(outputFile,fileName), bbox_inches='tight')

def makeHomogeneous(array):
    '''
    :param point: List of coordinates [X, Y, Z, etc] or a multi-dimensional array
    :return: Same point or matrix but now in homogeneous coordinates.
    '''
    if not type(array) == np.ndarray:
        temp = np.array(array,dtype=np.float64)
    else:
        temp = copy.deepcopy(array)
    if len(temp.shape) == 1:
        # If the length of the array is 1, it is a vector
        temp = temp.tolist()
        temp.append(1)
        temp = np.array(temp)
    else:
        # If the length of the array is greater than 1, it is a matrix
        if not all(x==temp.shape[0] for x in temp.shape):
            print("Your matrix must be square.\nYou passed in:\n",array)
            temp = None
        else:
            # the matrix is square, we add a zero to each row
            zeros = np.zeros((temp.shape[-1],1),dtype=np.float64)
            temp = np.concatenate((temp,zeros),axis=1)
            homoRow = np.concatenate((np.zeros(temp.shape[0]),[1]),axis=0)
            temp = np.vstack((temp,homoRow))
    if not type(array) == np.ndarray:
        temp = temp.tolist()
    return temp

def decomposeHomographyMatrix(array):
    '''
    This is the same function as 'make Cartesian' but with a name that implies what it does.
    :param array: A square numpy.ndarray
    :return: The Euclidean decomposition as a dictionary with keys { 'translation', 'rotation' }
    '''
    # Decompose the homogeneous array into a rotation matrix and a translation vector
    divisor = array[-1,-1]
    temp = np.divide(array, divisor)
    translation = temp[:,-1]
    translation = translation[:-1]
    rotation = temp[:-1,:-1]
    temp = {'translation': translation, 'rotation': rotation}
    return temp

def makeCartesian(array):
    if not type(array) == np.ndarray:
        temp = np.array(array,dtype=np.float64)
    else:
        temp = copy.deepcopy(array)
    if len(temp.shape) == 1:
        # Array is a vector and we can divide by the final element and exclude it
        temp = [x / temp[-1] for x in temp[:-1]]
    else:
        temp = decomposeHomographyMatrix(array)
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

def getRotationMatrixFromRadians(radians, dimensions: int=2):
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
                outRotationX = [[1, 0,                0              , 0],
                                [0, np.cos(radians), -np.sin(radians), 0],
                                [0, np.sin(radians),  np.cos(radians), 0],
                                [0, 0,                0,               1]]
                outRotationMatricies.append(outRotationX)

                outRotationY = [[np.cos(radians),   0,  np.sin(radians), 0],
                                [0,                 1,  0              , 0],
                                [-np.sin(radians),  0,  np.cos(radians), 0],
                                [0, 0,                0,                 1]]

                outRotationMatricies.append(outRotationY)

                outRotationZ = [[np.cos(radians),  -np.sin(radians), 0, 0],
                                [np.sin(radians),   np.cos(radians), 0, 0],
                                [0,                 0,               1, 0],
                                [0,                 0,               0, 1]]
                outRotationMatricies.append(outRotationZ)

                outMat = np.dot(outRotationMatricies[0], 
                                       np.dot(outRotationMatricies[1], outRotationMatricies[2]))

                return np.array(outMat, dtype=np.float64)
            else:
                outMat = [[np.cos(radians), -np.sin(radians), 0],
                                 [np.sin(radians),  np.cos(radians), 0],
                                 [0,                0,               1]]
                return np.array(outMat, dtype=np.float64)

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

def normalizeImage(img: np.ndarray):
    '''
    :param img: A numpy array of un-normalized values of any range.
    :return: The same image, normalized to the range [0.0,1.0]
    '''
    max = np.max(img)
    min = np.min(img)
    temp = np.subtract(img, min)
    temp = np.divide(img, (max - min))
    return temp

def normalizeForImageOutput(img: np.ndarray):
    '''
    :param img: A numpy array with values of any range.
    :return: The same image quantized to the standard [0,255] intensity scale.
    '''
    temp = normalizeImage(img)
    temp = np.multiply(temp, 255)
    temp = np.floor(temp)
    temp = np.array(temp,np.ushort)
    return temp

def resampleImage(image: sitk.SimpleITK.Image, transform):
    '''
    This function was borrowed from the Kitware SimpleITK tutorial notebooks.

    :param image: A SimpleITK Image.
    :param transform: A displacement field transform with the same dimensions as the image.
    :return: The input image under the provided displacement transform.
    '''
    if type(image) != sitk.SimpleITK.Image:
        image = sitk.GetImageFromArray(image, isVector=False)
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)