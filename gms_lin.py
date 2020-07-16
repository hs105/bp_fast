import numpy as np
import matplotlib.pyplot as plt

#similar ill-conditioned experiments as in https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb
#first imitating a one-layer net

def gen_ill_cond_matrix(xdim = 6, ydim=10):
    A = np.random.rand(ydim, xdim)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    s[0] *= 1e5
    B = np.dot(u, np.dot(np.diag(s), v))
    return B.T
    # print(np.allclose(A, Atrue)

xdim = 6
ydim=10
nsamples = 1000
W = gen_ill_cond_matrix(xdim, ydim)
print('shape of W: ', W.shape)

def row_wise_dot(a, b):
    '''row-wise dot'''
    result = np.empty((a.shape[0],))
    for i, (row1, row2) in enumerate(zip(a, b)):
        result[i] = np.dot(row1, row2)
    return result

def gen_data(W, nsamples, xdim):
    X = np.random.randn(nsamples, xdim)
    Y = X.dot(W)
    return X, Y

def min_grad(W, num_steps):
    m, n = (xdim, ydim)
    w = np.random.rand(m, n)
    final_acc = None
    st = []
    err = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        A = X.T.dot(X)
        b = X.T.dot(Y)

        R = A.dot(w) - b
        P = A.dot(R)

        alpha_n = row_wise_dot(P.T, R.T)
        alpha_d = row_wise_dot(P.T, P.T)

        # could alpha_d contain 0?
        alpha = alpha_n / alpha_d

        w = w - R.dot(np.diag(alpha))

        st.append(alpha)
        err.append(np.linalg.norm(R))
    return w, st, err, final_acc

w, st, err, final_acc = min_grad(W, num_steps=100)
plt.plot(err, '-b+')
plt.xlabel('steps')
plt.ylabel('loss')
plt.yscale('log')
plt.show()

