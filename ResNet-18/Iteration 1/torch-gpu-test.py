#! /usr/bin/env python 
# -------------------------------------------------------
import torch
# -------------------------------------------------------
print("Defining torch tensors:")
x = torch.Tensor(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)

# -------------------------------------------------------
# let us run the following only if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    x = x.cuda()
    y = y.cuda()
    print(x + y)
else:
    print("CUDA is NOT available.")

# -------------------------------------------------------
