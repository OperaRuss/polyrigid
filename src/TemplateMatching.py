'''
This is a program re-implementing a template matching assignment from the
Intro to CV course at NYU.  The goal was to accomplish the matching of a
template to a location in an image from which it was lifted. The extension
of the assignment, here, is to do so in PyTorch in an effective way.

There are two implementations here.  First, the basic cross-correlation matching
and then a (very memory hungry!) second correlation which sets the stage for a
possible block-matching algorithm explored for this project.
'''

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import utilities as utils

date = "20220308"

def showImg(img,title="",save=None):
    plt.imshow(img.detach().squeeze().cpu().numpy()[:,:], cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.savefig("../images/results/"+date+"/"+save+".png",bbox_inches='tight')
    plt.show()
    img.cuda().unsqueeze(0).unsqueeze(0)

def thresh(img, above=1.0, below=0.0):
    return np.where(img < 0.9, above, below)

# Data Loading
source = sitk.ReadImage("../images/input/keys/keys_multiple.png")
source = utils.normalizeImage(sitk.GetArrayFromImage(source))
kernel = sitk.ReadImage("../images/input/keys/keys_crop_unique.png")
kernel = utils.normalizeImage(sitk.GetArrayFromImage(kernel))
kernel = kernel[0:49,0:49,0]


'''
The first section of this code replicates the Intro to CV key template matching assignment.
The goal was to achieve basic cross correlation of a template against an image, but using PyTorch instead of
the for-loop driven model used in the course.  

For straight CC, delete the subtraction by the mean.
For zero-normalized CC, keep the mean subtraction.
'''
th_k = thresh(kernel)
th_img = thresh(source)

tKernel = torch.tensor(th_k).cuda().unsqueeze(0).unsqueeze(0)
tImg = torch.tensor(th_img).cuda().unsqueeze(0).unsqueeze(0)

tKernel_mean = torch.mean(tKernel)
tImg_mean = torch.mean(tImg)

ndims = len(tKernel.shape)-2

if(tKernel.shape[2] % 2 == 0):
    padH = (tKernel.shape[2] - 1)//2
else:
    padH = tKernel.shape[2]//2
if tKernel.shape[3] // 2 == 0:
    padW = (tKernel.shape[3] - 1)//2
else:
    padW = tKernel.shape[3]//2

padding = (padH, padW)
stride = (1,1)
conv_fn = getattr(F,'conv%dd' % ndims)

test = conv_fn(tImg - tImg_mean,tKernel - tKernel_mean,stride=stride,padding=padding)

showImg(test,"Template Matching by Cross-Corr",save="res_keys_CC")

print((test==torch.max(test)).nonzero())

'''
WARNING: THIS PART OF THE PROGRAM IS EXCEPTIONALLY MEMORY HUNGRY
    For this reason it has been moved to the CPU. I have 16 Gigs on my computer and it is almost
    entirely occupied while running.  This is non-optimal, but it was a way to test the theory.
    
This second section implements Voxel Morph's Normalized Cross Correlation algorithm which
standardizes the output relative to the variance of intensity.  It outputs the exact same
results but with a much sharper peak where the template match occurs.

Basically, in order to obtain the local patch variance, we have to respect the spatiality of the
patch system.  Convolution layers do this naturally, but there is data loss wrt the kernel when
the kernel shifts between several windows.  

In the implementation below, steps [a:i] calculate the numerator of the NCC equation and steps [j:n] 
calculate the denominator.  Of these [a:c] and [j:k], which are among the more expensive, can be calculated 
once and held static for the remainder of the program. The remaining can be calculated for each block at the start
of the iteration (is the idea).  We basically have 'opened' the neighborhoods into columns and then use them
as references for each consecutive calculation.  It is, in theory, a way to speed up what may be a slow process
of repeatedly feeding the whole system through several convolutional layers.  However, it turned out
to be much more memory expensive than I had thought it would.

I also had thought that the NCC metric would be rotation invariant, based on something said in an article I read.
This was a false presumption and I will need to return to the source and re-evaluate.  It drives me to wonder
if mutual information is a better metric in this regard or if NCC (with its peaky quality on near matches) may
still be serviceable.

The implementation here is explained in a document curated on overleaf.  Please request it if
you are curious about what it is trying to do beyond what's written here.
'''
import unfoldNd as uf

dilation = 1
stride = 1
padding = (tKernel.shape[2]//2,tKernel.shape[3]//2)

unf = uf.UnfoldNd(tKernel.squeeze().shape,dilation,padding,stride)
a = unf(tImg.cpu()).cpu()
b = torch.mean(a,dim=1,keepdim=True)
c = a - b
d = torch.mean(tKernel)
e = torch.subtract(tKernel,d).cpu()
f = torch.reshape(e,(1,np.prod(tKernel.shape)))
g = c * f[:,:,None]
h = torch.sum(g,axis=1,keepdim=True)
i = torch.square(h)
j = torch.square(c)
k = torch.sum(j, axis=1,keepdim=True)
l = torch.square(f)
m = torch.sum(l)
n = k * m
o = torch.div(i,torch.add(n,1e-5))

result = o.reshape(tImg.shape)

showImg(result,"Template Matching by NCC",save="res_keys_NCC")
print((result==torch.max(result)).nonzero())



