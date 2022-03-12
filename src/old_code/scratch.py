import torch
import torch.nn
import torch.nn.functional as F

a = torch.tensor(torch.randn(3,9),requires_grad=True)
b = torch.tensor(torch.randn((3,3)),requires_grad=True)
t = torch.randn((3,1))
x = torch.randn((4,1))
x[3] = 1.0
w = torch.tensor([[0.66,0.23,0.11]])
n = 3

for i in range(n):
    print("A"+str(i),a[i].view(3,3))
    print("B"+str(i),b[i].view(3,1))
    print()

import torch.optim as op

optim = op.Adam([a,b],lr=0.01)

for i in range(10):
    temp = torch.cat((a,b),axis=1)
    temp = torch.cat((temp,torch.zeros(3,4)),axis=1)
    temp[:,-1] = 1.0
    curr = w @ temp
    out = curr.view(4,4) @ x
    out = (torch.div(out,out[3]))[0:3]
    loss = F.mse_loss(out,t)
    loss.backward()

    optim.step()

    print("Loss: ",loss)

for i in range(n):
    print("A"+str(i),a[i])
    print("B"+str(i),b[i])
    print()