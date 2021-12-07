import SimpleITK
import numpy as np
import matplotlib.pylab as plt
import SimpleITK as sitk
import utilities as utils

img = np.zeros([128,128,128])

class Circle():
    def __init__(self, center: list, radius: float):
        self.center = center
        self.radius = radius

    def isWithin(self, point: list):
        radSqr = self.radius ** 2
        xSqr = (point[0] - self.center[0]) ** 2
        ySqr = (point[1] - self.center[1]) ** 2
        zSqr = (point[2] - self.center[2]) ** 2
        return (xSqr + ySqr + zSqr) < radSqr

class Ellipse():
    def __init__(self, center: list, axisMajor: float, axisMinor: float, axisWonky: float):
        self.center = center
        self.axisMajor = axisMajor
        self.axisMinor = axisMinor
        self.axisWonky = axisWonky

    def isWithin(self, point:list):
        epsilon = 1e-5
        majRadSqr = self.axisMajor ** 2
        minRadSqr = self.axisMinor ** 2
        wokRadSqr = self.axisWonky ** 2
        xSqr = (point[0] - self.center[0]) ** 2
        ySqr = (point[1] - self.center[1]) ** 2
        zSqr = (point[2] - self.center[2]) ** 2
        return ((xSqr/(majRadSqr + epsilon)) + \
               (ySqr/(minRadSqr + epsilon)) + \
               (zSqr/(wokRadSqr + epsilon))) <= 1


c = Circle([64,64,64], 20)

e = Ellipse([64,64,64], 20,10,30)

utils.showNDA_InEditor_BW(img[:,:,64])

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        for z in range(img.shape[2]):
            if(e.isWithin([x,y,z])):
                img[x,y,z] = 1.0

sitk.Show(SimpleITK.GetImageFromArray(img))