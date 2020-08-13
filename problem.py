import numpy as np

def gen_well_cond_matrix(xdim = 6, ydim=10):
    A = np.random.rand(ydim, xdim)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    B = np.matmul(u, np.matmul(np.diag(s), v))
    return B.T

def gen_ill_cond_matrix(xdim = 6, ydim=10):
    A = np.random.rand(ydim, xdim)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    s[0] *= 1e5
    B = np.matmul(u, np.matmul(np.diag(s), v))
    return B.T

xdim = 6
ydim=10
W = gen_well_cond_matrix(xdim, ydim)
# W = gen_ill_cond_matrix(xdim, ydim)
print('cond(W): ', np.linalg.cond(W))

def gen_data(nsamples):
    X = np.random.randn(nsamples, xdim)
    Y = np.matmul(X, W)
    return X, Y

