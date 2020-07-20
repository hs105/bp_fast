import numpy as np
import matplotlib.pyplot as plt
import sys

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

def last_layer_1step(X, Y, w):
    '''
    our algorithm for updating the last layer ( only one step).
    '''
    A = X.T.dot(X)
    b = X.T.dot(Y)

    R = A.dot(w) - b
    P = A.dot(R)

    alpha_n = row_wise_dot(P.T, R.T)
    alpha_d = row_wise_dot(P.T, P.T)

    # could alpha_d contain 0?
    alpha = alpha_n / alpha_d

    w = w - R.dot(np.diag(alpha))

    #recalculate; R1 should always be smaller
    R1 = A.dot(w) - b

    return w, alpha, R, R1

def ill_example1(W, num_steps):
    '''this illustrate our algorithm for solving an ill-conditioned learning example with only one layer '''
    m, n = (xdim, ydim)
    w = np.random.rand(m, n)
    final_acc = None
    st = np.empty((num_steps, w.shape[1]))
    err = []
    err1 = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        w, alpha, R, R1 = last_layer_1step(X, Y, w)

        st[k, :] = alpha
        err.append(np.linalg.norm(R))
        err1.append(np.linalg.norm(R1))
    return w, st, err, err1, final_acc

def min_delta(W, num_steps):
    '''minimizing the prediction error'''
    m, n = (xdim, ydim)
    w = np.random.rand(m, n)
    final_acc = None
    st = np.empty((num_steps, w.shape[1]))
    err = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        delta = X.dot(w) - Y
        print('X.shape:', X.shape)
        print('delta.shape', delta.shape)
        dotx = row_wise_dot(X, delta)
        print('dotx.shape:', dotx.shape)
        sys.exit(1)

# min_delta(W, num_steps=100)

w, st, err, err1, final_acc = ill_example1(W, num_steps=100)
plt.figure(1)
plt.plot(err, '-b+', label='BEFORE weight update')
plt.plot(err1, '-k.', label='AFTER weight update')
plt.xlabel('steps')
plt.ylabel('loss (squared norm of gradient)')
plt.yscale('log')
plt.show()


plt.figure(2)
markers = ['-b+', '--ko', ':g<', '--rs']
for i in range(len(markers)):#range(st.shape[1]):
    plt.plot(st[:, i], markers[i])
plt.xlabel('steps')
plt.ylabel('step-sizes')
# plt.yscale('log')
plt.show()


