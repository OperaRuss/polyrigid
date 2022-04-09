import torch
import numpy as np
import SimpleITK as sitk
import torchmaxflow as tmf
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch is using", device)
print()

img = sitk.ReadImage('../images/input/rave/iso_frame_8.nii.gz',sitk.sitkFloat32)
nImg = sitk.GetArrayFromImage(img)

label = sitk.ReadImage('../images/input/rave/iso_frame_0_seg_binary.nii.gz',sitk.sitkFloat32)
nLabel = sitk.GetArrayFromImage(label)
fP = 0.5 + (nLabel - 0.5) * 0.95
bP = 1.0 - fP
Prob = np.asarray([bP, fP])

lamda = 0.5
sigma = 1.0



tImg = torch.from_numpy(nImg).unsqueeze(0).unsqueeze(0)
tLabel = torch.from_numpy(Prob).unsqueeze(0)

print(tLabel.max(), tLabel.min())
result = tmf.maxflow(tImg,tLabel,lamda,sigma).squeeze().cpu().numpy()

slice = 38

print(result.max(),result.min())
plt.imshow(result[slice,:,:])
plt.title("lamda: "+str(lamda)+",sigma: "+str(sigma))
plt.show()

plt.imshow(nLabel[slice,:,:])
plt.title("Original")
plt.show()

print(np.linalg.norm(result - nLabel))

