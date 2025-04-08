import numpy as np

# checked!
def soft_threshold(x0,lmbda):
    x0 = np.sign(x0)*np.maximum(np.abs(x0)-lmbda,0)
    return(x0)

def grad(x0):
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


A = np.array([[1,2],[3,4]])
print(A)
[fx,fy] = grad(A)
print(type(fx))
print(type(fy))
print(fx)
print(fy)