import torch

def activation(x):
    '''
    Using sigmoid function as activation function
    '''
    return 1/(1 - torch.exp(x * -1))
'''
Generating random features, weights and bias
'''
features = torch.rand((1,5))
weights = torch.rand((5,1))
bias = torch.rand((1,1))

h = torch.mm(features,weights) + bias

y = activation(h)
'''
print("features",features)
print("weights",weights)
print("bias",bias)
print("h",h)
'''
print("y: ",y)
