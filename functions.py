import numpy as np

def soft_threshold(x0,lmbda):
    x0 = np.sign(x0)*np.maximum(np.abs(x0)-lmbda,0)
    return(x0)

#x0 = np.random.rand(1,4)
x0 = ([-3,-1,0,2,5])
print(x0)

y = soft_threshold(x0,1)
print(y)

