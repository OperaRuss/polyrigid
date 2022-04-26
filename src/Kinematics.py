import SimpleITK as sitk
import numpy as np
from skimage.measure import label, regionprops

img = sitk.ReadImage("../images/results/frame_1_to_frame_0/warped_seg_6.nii")
npImg = sitk.GetArrayFromImage(img)
label = label(npImg)
region = regionprops(label)
props = region[0]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
cd,ch,cw = props.centroid
cd = int(cd)
ch = int(ch)
cw = int(cw)
ax.imshow(label[cd,:,:])
ax.plot(cw,ch,'.g',markersize=15)

plt.show()


