import numpy as np
from numpy.typing import NDArray

# checked!
def soft_threshold(x0: np.ndarray,lmbda: int)-> np.ndarray:
    x0 = np.sign(x0)*np.maximum(np.abs(x0)-lmbda,0)
    return(x0)

# checked!
def grad(x0: np.ndarray) -> np.ndarray:
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

# checked! (From a discrete viewpoint this is indeed the transpose operator of the gradient)
def div(fx: np.ndarray,fy: np.ndarray) -> np.ndarray:
    n1 = fx.shape[0]
    I = np.eye(n1)
    fx = fx.reshape(-1,1,order = 'F')
    fy = fy.reshape(-1,1,order = 'F')
    # kernel define the block structure of the transpose of the horizontal and vertical derivative
    kernel = -np.eye(n1)
    kernel[np.arange(1,n1),np.arange(0,n1-1)] = 1
    kernel[-1,-1] = 0
    # the full matrix is a tensor product of matrices
    LvT = np.kron(I,kernel)
    LhT = np.kron(kernel,I)
    divergence = np.dot(LvT, fy) + np.dot(LhT, fx)
    divergence = - divergence.reshape([n1,n1],order = 'F')
    return(divergence)

# checked! 
def ProxHstar(x0: np.ndarray,lmbda: float) -> np.ndarray:
    gradient_norm = np.sqrt(np.sum(x0**2,axis=2))
    gradient_norm = np.repeat(gradient_norm[:,:,np.newaxis],2,axis = 2)
    prox = x0 / np.maximum(gradient_norm/lmbda,1)
    return(prox)

# NOT checked yet: to be added the choice for computing the preconditioned gradient
def gradient_datafidelity(eigA: np.ndarray, x0: np.ndarray, bdelta: np.ndarray)-> np.ndarray:
    bhat = np.fft.fft2(bdelta)
    xhat = np.fft.fft2(x0)
    eig_xhat = eigA * xhat
    grad = np.real(np.fft.ifft2(np.conj(eigA) *(eig_xhat - bhat)))


    