import numpy as np
from numpy.typing import NDArray

# checked!
def soft_threshold(x0,lmbda):
    x0 = np.sign(x0)*np.maximum(np.abs(x0)-lmbda,0)
    return(x0)

# checked!
def grad(x0):
    # Description: this function compute horizontal and vertifcal derivaties using finite difference
    #              approximation, s.t. f(x0+h)-f(x0). In this way we have that the gradient in the
    #              pixel x0_11 will be x_12-x_11 (horizontal) and x_21-x_11 (vertical). For the BC's
    #              we reflect the last row and column.
    matrix_v = np.zeros_like(x0)
    matrix_v[:-1,:] = x0[1:,:]
    matrix_v[-1,:] = x0[-1,:]
    matrix_h = np.zeros_like(x0)
    matrix_h[:,:-1] = x0[:,1:]
    matrix_h[:,-1] = x0[:,-1]
    # compute vertical derivative with Neumann boundary conditions (symmetric BC's)
    gradient_y = matrix_v - x0
    # compute horizontal derivative with Neumann boundary conditions (symmetric BC's)
    gradient_x = matrix_h -x0
    return(gradient_x,gradient_y)

# checked!
def div(fx: np.ndarray,fy: np.ndarray) -> np.ndarray:
    n1 = fx.shape[0]
    I = np.eye(n1)
    fx = fx.reshape(-1,1,order = 'F')
    fy = fy.reshape(-1,1,order = 'F')
    # kernel define the block structure of the transpose of the horizontal and vertical derivative
    kernel = -np.eye(n1)
    kernel[np.arange(1,n1),np.arange(0,n1-1)] = 1
    kernel[-1,-1] = 0
    print(kernel)
    # the full matrix is a tensor product of matrices
    LvT = np.kron(I,kernel)
    LhT = np.kron(kernel,I)
    divergence = np.dot(LvT, fy) + np.dot(LhT, fx)
    divergence = - divergence.reshape([n1,n1],order = 'F')
    return(divergence)
