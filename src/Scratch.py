import torch
import numpy as np
import SimpleITK as sitk
import torchmaxflow as tmf
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch is using", device)
print()

frame = 8

img = sitk.ReadImage('../images/input/rave/iso_frame_'+str(frame)+'.nii.gz',sitk.sitkFloat32)
nImg = sitk.GetArrayFromImage(img)

label = sitk.ReadImage('../images/input/rave/iso_frame_'+str(frame)+'_seg_binary.nii.gz',sitk.sitkFloat32)
nLabel = sitk.GetArrayFromImage(label)
fP = 0.5 + (nLabel - 0.5) * 0.8
bP = 1.0 - fP
Prob = np.asarray([bP, fP])

plt.imshow(fP[38,:,:])
plt.show()
plt.close()


plt.imshow(bP[38,:,:])
plt.show()
plt.close()

lamda = [1.0,1.25]
sigma = [1.0]



tImg = torch.from_numpy(nImg).unsqueeze(0).unsqueeze(0)
tLabel = torch.from_numpy(Prob).unsqueeze(0)

plt.imshow(nLabel[38,:,:])
plt.title("Original")
plt.show()

for lam in lamda:
    for sig in sigma:
        result = tmf.maxflow(tImg,tLabel,lam,sig).squeeze().cpu().numpy()

        slice = 38

        print(result.max(),result.min())
        plt.imshow(result[slice,:,:])
        plt.title("lamda: "+str(lam)+",sigma: "+str(sig))
        plt.show()

        out = sitk.GetImageFromArray(np.multiply(result,nLabel))
        out.CopyInformation(img)
        sitk.WriteImage(out,"../images/results/20220410/result_"+str(lam).replace('.','_')+"_"+str(sig).replace('.','_')+".nii")

        print("lam: ",lam," sig: ",sig)
        print("metric: ",np.linalg.norm(result - nLabel))

