import torch
import scipy.linalg

'''
Implementation sourced from https://github.com/pytorch/pytorch/issues/9983#issuecomment-891777620
Currently only known method for implmenting matrix logarithm within pytorch.
'''
def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)

def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)

logm = Logm.apply

A = torch.rand(4,4, dtype=torch.float64, requires_grad=True)
B = torch.rand(4,4,dtype=torch.float64, requires_grad=True)
func = 0.5 * logm(A) + 0.5 * logm(B)
loss = torch.norm(func,p=2)
loss.backward()
print(loss)
print(A)
print(A.grad)
print(B)
print(B.grad)

A.grad.data.zero_()
B.grad.data.zero_()
print()

func = 0.80 * logm(A) + 0.2 * logm(B)
func2 = torch.matrix_exp(func)
loss = torch.norm(func2,p=2)
loss.backward()
print(loss)
print(A)
print(A.grad)
print(B)
print(B.grad)


