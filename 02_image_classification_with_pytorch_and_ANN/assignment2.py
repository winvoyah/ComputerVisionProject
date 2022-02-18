# Import torch and NumPy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional

# Set the random seed for NumPy and PyTorch both to "42"
#   This allows us to share the same "random" results.
torch.manual_seed(42)
np.random.seed(42)

# Create a NumPy array called "arr" that contains 6 random integers between 0 (inclusive) and 5 (exclusive)
arr = np.random.randint(0, 5, 6, dtype=int)
print(arr, arr.dtype)
# Create a tensor "x" from the array above
x_var = torch.tensor(arr, dtype=torch.int32)
print(x_var, x_var.dtype)

# Change the dtype of x from 'int32' to 'int64'
x_var = x_var.type(torch.int64)
print(x_var, x_var.dtype)

# Reshape x into a 3x2 tensor
x_var = x_var.reshape(3, 2)
print(x_var)

# Return the left-hand column of tensor x
print (f'from X: {x_var} \nleft hand side of x:\n {x_var[:, 0]}')

# Without changing x, return a tensor of square values of x
# x_sqrd = torch.pow(x_var, 2)
print (f' for the values of X: {x_var} \n the square of the vlaues of x is:{torch.pow ( x_var, 2 )}')

# Create a tensor "y" with the same number of elements as x, that can be matrix-multiplied with x
# reshape y such that it can be multiplied with x. 3x2 vs 2x3 matrix
y = x_var.reshape(2, 3)
print (y)
# Find the matrix product of x and y
print (f'product of x:{x_var} \n  '
        f'and y:{y} \n '
        f'is X * y: {torch.mm ( x_var, y )}')

# Create a Simple linear model using torch.nn
# the model will take 1000 input and output 20 multi-class classification results.
# the model will have 3 hidden layers which include 200, 120, 60 respectively.


class SimpleLinearModel(nn.Module):
    def __init__(self, in_sz=1000, out_sz=20, layers=[200, 120, 60]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.fc4 = nn.Linear(layers[2], out_sz)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        # Improving neural networks by preventing co-adaptation of feature detectors


    def forward(self, x):
        x = torch.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        # multi-class classification with log_softmax
        return x



# initiate the model and printout the number of parameters
model = SimpleLinearModel()
print(model)



