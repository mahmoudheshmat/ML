'''
Coding Softmax
'''
import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    s=0
    for x in L:
        s=s+np.exp(x)
    for x in range(len(L)):
        L[x]=np.exp(L[x])/s
    
    return L
        
