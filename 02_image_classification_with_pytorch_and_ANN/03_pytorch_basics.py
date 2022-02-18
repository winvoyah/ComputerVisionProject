# This section covers:
# Converting NumPy arrays to PyTorch tensors
# Creating tensors from scratch

# import libs torch, numpy
import torch
import numpy as np

print ( 'converting NumPy arrays to PyTorch tensors' )
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# Calculations between tensors can only happen if the tensors share the same dtype.

print ( 'lets simply create a numpy array with value 1,2,3,4,5' )
arr = np.array ( [1, 2, 3, 4, 5] )
print ( arr )
print ( arr.dtype )
print ( type ( arr ) )

arr1 = torch.tensor ( arr )
print ( arr1.dtype )
print ( type ( arr1 ) )

print ( 'now lets create a 4x3 2D array (matrix)' )
arr = np.arange ( 0, 12 ).reshape ( 4, 3 )
print ( arr )
print ( 'lets convert this 2D array into a torch tensor' )
arr1 = torch.tensor ( arr )
print ( arr1 )

print ( 'lets create a tensor from scratch' )
#   Uninitialized tensors with .empty()
z = torch.empty ( 3, 4 )
print ( z.dtype )

#   Initialized tensors with .zeros() and .ones()
z = torch.ones ( 3, 4 )
print ( z.dtype )

z = torch.zeros ( 3, 4, dtype=torch.int32)
print ( z.dtype )
print ( z )

print ( 'Tensors from ranges' )

z1 = torch.arange ( 0, 12, dtype=torch.int64 ).reshape ( 3, 4 )
print ( z1, z1.dtype, type ( z1 ) )

print ( 'Tensors from data' )
z1 = torch.tensor ( (0, 1, 2, 3, 4, 5, 6) )
print ( z1 )
print ( 'Random number tensors that follow the input size' )
z1 = torch.randint ( 3, 7, [3, 4] )
print ( z1 )

z1 = torch.randn ( 5, 5 )
print ( z1 )
print ( 'Set random seed which allows us to share the same "random" results.' )
np.random.seed ( 1 )
torch.manual_seed ( 1 )
print(np.random.randint(0,5,4))
print(torch.randint(0,5,[2,2]))


print ( 'Tensor attributes' )

print(z1.shape)
print(z1.size())
print(z1.device)
# PyTorch supports use of multiple devices, harnessing the power of one or more GPUs in addition to the CPU.
# We won't explore that here, but you should know that operations between tensors can only happen for tensors installed on the same device.
