import numpy as np
import utilities as utils
from scipy import ndimage


def getDistanceToComponentRegion(componentSegmentation: np.ndarray):
    '''
    Comes from Makki, Borotikar, Garetier, et al., "A fast and memory-efficient algorithm for smooth
    interpolation of polyrigid transformations: application to human joint tracking." arXiv preprint
    arXiv:2005.02159. Source code found here: https://github.com/rousseau/dynMRI

    This function utilizes scipy.ndimage.distance_transform_edt() to calculate the distance
    between each pixel in the image and the region of interest.  The expectation is that
    the input has high values within the region of interest and zero (or negative) values elsewhere.

    :param componentSegmentation: A numpy array containing a segmentation of the region of interest
    :return: The distance from each pixel to the nearest in-region pixel.
    '''

    maxIntensity = np.max(componentSegmentation)
    invertedImage = np.subtract(maxIntensity, componentSegmentation)
    return ndimage.distance_transform_edt(invertedImage)

def getCommowickWeight(componentSegmentation: np.ndarray, gamma: float):
    '''
    This function comes from Commowick, Arsigny, Isambert, et al. "An efficient locally affine
    framework for the smooth registration of anatomical structures," Medical Image Analysis 12
    (2008), p427-441.

    :param componsnetSegmentation: Image as numpy array containing the binary label of
    :param gamma: User hyperparameter controlling rate of decay of component's region of influence.
    :return: The weight of contribution the component has at each pixel in the image.
    '''
    # Makki's implementation proposed an alternative function with steeper drop off
    # 2.0 / ( 1.0 + np.exp(0.4 * getDistanceToComponentRegion(componentSegmentation))
    return (1.0 / (1.0 + (gamma * pow(getDistanceToComponentRegion(componentSegmentation), 2))))

def getNormalizedCommowickWeight(listComponentSegmentation: list, listParamsRateOfDecay: list, is3D: bool=False):
    '''
    Function computes the normalized contribution weight at each pixel for each component. First,
    the raw contribution is contributed using the Commowick weight function.  Then, a sum is calculated
    over all contributions at each pixel.  Finally, each weight image is normalized by the sum of weights
    present at that pixel.  For pixels far from any component in the image, behavior may be unrealistic,
    and care must be taken when evaluating object behavior near the border of the region of interest.

    :param listComponentSegmentation: a list of images, as numpy arrays, containing component segmentation labels
    :param listParamsRateOfDecay: a list of float-type hyperparameters specifying the rate of decay of influence for each component
    :param is3D: boolean argument for whether images are 3D (True) or 2D (False).
    :return: a list of numpy arrays containing normalized contribution weight of each component
    '''
    numComponents = len(listComponentSegmentation)
    listCommowickWeights = []
    for idx in range(numComponents):
        listCommowickWeights.append(getCommowickWeight(listComponentSegmentation[idx], listParamsRateOfDecay[idx]))
    
    imWidth = listComponentSegmentation[0].shape[0]
    imHeight = listComponentSegmentation[0].shape[1]
    if(is3D):
        imDepth = listComponentSegmentation[0].shape[3]
        sumImage = np.zeros((imWidth,imHeight),dtype=np.float64)

        for x in range(imWidth):
            for y in range(imHeight):
                for z in range(imDepth):
                    for n in range(numComponents):
                        sumImage[x,y] += listCommowickWeights[n][x,y]
    else:
        sumImage = np.zeros((imWidth, imHeight), dtype=np.float64)

        for x in range(imWidth):
            for y in range(imHeight):
                for n in range(numComponents):
                    sumImage[x,y] += listCommowickWeights[n][x,y]
    
    listNormalizedWeights = []

    for n in range(numComponents):
        listNormalizedWeights.append(np.divide(listComponentSegmentation[n], sumImage))
        utils.showNDA_InEditor_BW(listNormalizedWeights[n])
    
    return listNormalizedWeights
    

def testFunction():
    import Shape

    img = np.zeros((64,64), dtype=np.float64)
    c = Shape.Ellipse([32,32],30,10,2)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(c.isWithin([x,y])):
                img[x,y] = 1.0

    img1 = np.zeros((64,64), dtype=np.float64)
    c = Shape.Ellipse([12,12],10,10,2)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(c.isWithin([x,y])):
                img1[x,y] = 1.0

    listImages = [img, img1]
    listRatesOfDecay = [0.25, 0.1]

    for n in range(len(listImages)):
        utils.showNDA_InEditor_BW(listImages[n], "Image " + str(n))

    listNormalizedWeights = getNormalizedCommowickWeight(listImages, listRatesOfDecay, is3D=False)

    for n in range(len(listImages)):
        utils.showNDA_InEditor_BW(listNormalizedWeights[n], "Normalized " + str(n) + ", Max: " + str(np.max(listNormalizedWeights[n])))