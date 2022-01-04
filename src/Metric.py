import numpy as np
import skimage

def ComputeNormalizedMutualInformation(image1: np.ndarray, image2: np.ndarray, bins: int=50):
    '''
    This function is a wrapper that calls SciKit image's normalized mutual information function.
    Sourced from the SciKit image documentation cited here: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.normalized_mutual_information
    Uses the method cited from C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap invariant entropy measure of 3D medical image alignment. 
    Pattern Recognition 32(1):71-86 DOI:10.1016/S0031-3203(98)00091-0

    :param image1: N-dimensional numpy array containing intensity information.
    :param image2: N-dimensional numpy array of same dimension as image1.
    :param bins: Integer value representing granularity of similarity calculation.
    :return: Float valued normalized mutual information score. Higher indicates greater similarity.
    '''
    return skimage.metrics.normalized_mutual_information(image1,image2,bins)

def ComputeNormalizedRMSE(fixedImage: np.ndarray, warpedImage: np.ndarray, normalization: str='euclidean'):
    return skimage.metrics.normalized_root_mse(fixedImage, warpedImage, normalization)







