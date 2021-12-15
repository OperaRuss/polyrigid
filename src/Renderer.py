import SimpleITK
import numpy as np
import matplotlib.pylab as plt
import SimpleITK as sitk
import utilities as utils

point = [1.0,1.0]

print(point)
foo = utils.makeHomogenous(point)
print(foo)
foo[2] = 0.5
foo = utils.makeCartesian(foo)
print(foo)
