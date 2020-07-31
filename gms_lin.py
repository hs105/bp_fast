import numpy as np
import matplotlib.pyplot as plt
import sys

#similar ill-conditioned experiments as in https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb
#first imitating a one-layer net

def gen_ill_cond_matrix(xdim = 6, ydim=10):
    A = np.random.rand(ydim, xdim)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    s[0] *= 1e5
    B = np.matmul(u, np.matmul(np.diag(s), v))
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
    Y = np.matmul(X, W)
    return X, Y

def rate_of_mr(A):
    mu = min(np.linalg.eigvals(A + A.T)) / 2
    sigma = np.linalg.norm(A, 2)
    return np.sqrt(1 - (mu / sigma) ** 2)

def last_layer_1step(X, Y, w):
    '''
    our algorithm for updating the last layer (only one step).
    '''

    A = np.matmul(X.T, X)
    rate = rate_of_mr(A)
    # print('cond(A)=', np.linalg.cond(A))
    # print('norm(A)=', np.linalg.norm(A))
    b = np.matmul(X.T, Y)

    R = A.dot(w) - b
    P = A.dot(R)

    err_Y = X.dot(w) - Y

    alpha_n = row_wise_dot(P.T, R.T)
    alpha_d = row_wise_dot(P.T, P.T)

    # could alpha_d contain 0?
    alpha = alpha_n / alpha_d

    w = w - R.dot(np.diag(alpha))

    #recalculate; R1 should always be smaller than R
    R1 = A.dot(w) - b

    return w, alpha, R, R1, err_Y, rate

def ill_example1(W, num_steps):
    '''this illustrate our algorithm for solving an ill-conditioned learning example with only one layer '''
    m, n = (xdim, ydim)
    w = np.random.rand(m, n)
    st = np.empty((num_steps, w.shape[1]))
    err = []
    err1 = []
    err_Y = []
    rates = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        w, alpha, R, R1, errY, rate = last_layer_1step(X, Y, w)

        st[k, :] = alpha
        err.append(np.linalg.norm(R))
        err1.append(np.linalg.norm(R1))
        err_Y.append(np.linalg.norm(errY))
        rates.append(rate)
    return w, st, err, err1, err_Y, rates

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

w, st, err, err1, err_Y, rates = ill_example1(W, num_steps=100)
plt.figure(1)
plt.plot(err, '-b+', label='BEFORE weight update')
plt.plot(err1, '-k.', label='AFTER weight update')
plt.plot(err_Y, '--ro', label='error of predicting Y')
plt.plot(rates, '-g<', label='rates')
plt.legend()
plt.xlabel('steps')
plt.ylabel('loss (squared norm of gradient)')
plt.yscale('log')
plt.title('the \'AFTER\' curve should always be smaller')


plt.figure(2)
markers = ['-b+', '--ko', ':g<', '--rs']
for i in range(len(markers)):#range(st.shape[1]):
    plt.plot(st[:, i], markers[i])
plt.xlabel('steps')
plt.ylabel('step-sizes')
# plt.yscale('log')

#similar ill-conditioned experiment 2: two-layer linear
#for the same ill-conditioned example
#use W2W1X to approximate Y

def last_layer_1step2(X, Y, w1, w2):
    '''
    our algorithm for updating the last layer (only one step), w2
    w1 is fixed for now
    '''
    print('cond(X.T*X):', np.linalg.cond(np.matmul(X.T, X)))
    X = X.dot(w1)
    XTX = np.matmul(X.T, X)
    print('cond(w1.T*X.T*X*w1):', np.linalg.cond(XTX))
    _, sg, _ = np.linalg.svd(XTX)
    #looks the smaller matrix, k by k conditioning is important.
    print('eig(w1.T*X.T*X*w1):, top-k:', sg[:xdim], ', remaining: ', sg[xdim:])

    w2, alpha, R, R1, err_Y, rate = last_layer_1step(X, Y, w2)
    return w2, alpha, R, R1, err_Y, rate


def ill_example1_2layers(W, num_steps):
    '''this illustrate our algorithm for solving the same ill-conditioned learning example but with two layers '''
    m, n = (xdim, ydim)
    w1 = np.random.rand(m, m)
    #w1 = np.eye(m) + 0.1 * np.random.rand(m, m)
    # print('cond(w1)=', np.linalg.cond(w1))
    w2 = np.random.rand(m, n)
    st = np.empty((num_steps, w.shape[1]))
    err = []
    err1 = []
    err_Y = []
    rates = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        w2, alpha, R, R1, errY, rate = last_layer_1step2(X, Y, w1, w2)

        st[k, :] = alpha
        err.append(np.linalg.norm(R))
        err1.append(np.linalg.norm(R1))
        err_Y.append(np.linalg.norm(errY))
        rates.append(rate)
    return w, st, err, err1, err_Y, rates


# w, st, err, err1, err_Y, rates = ill_example1_2layers(W, num_steps=100)
# plt.figure(3)
# plt.plot(err, '-b+', label='2-layer w2: BEFORE weight update')
# plt.plot(err1, '-k.', label='2-layer w2: AFTER weight update')
# plt.plot(err_Y, '-ro', label='2-layer w2: error of predicting Y')
# plt.plot(rates, '-g<', label='rates')
# plt.legend()
# plt.xlabel('steps')
# plt.ylabel('loss (squared norm of gradient)')
# plt.yscale('log')
# plt.title('two-layer linear system: the \'AFTER\' curve should always be smaller')
#
# plt.figure(4)
# markers = ['-b+', '--ko', ':g<', '--rs']
# for i in range(len(markers)):#range(st.shape[1]):
#     plt.plot(st[:, i], markers[i])
# plt.xlabel('steps')
# plt.ylabel('step-sizes')
#
# plt.show()

def ill_example1_2layers_1(W, n1, num_steps):
    ''' n1: output size of layer 1
    n1 is smaller or larger than m
    '''
    m, n = (xdim, ydim)
    if m > n1:
        w1 = np.eye(n1) + 0.1 * np.random.rand(n1, n1)
        #w1 = np.vstack((w1, np.zeros((m-n1, n1))))
        w1 = np.vstack((w1, 0.1 * np.random.rand(m-n1, n1)))
    elif m < n1:
        w1 = np.eye(m) + 0.9 * np.random.rand(m, m)
        w1 = np.hstack((w1, 0.1 * np.random.rand(m, n1-m)))
    else:
        w1 = np.eye(m) + 0.1 * np.random.rand(m, m)

    assert w1.shape == (m, n1)
    print('cond(w1)=', np.linalg.cond(w1))
    w2 = np.random.rand(n1, n)
    st = np.empty((num_steps, w.shape[1]))
    err = []
    err1 = []
    err_Y = []
    rates = []
    for k in range(num_steps):
        X, Y = gen_data(W, nsamples, xdim)

        w2, alpha, R, R1, errY, rate = last_layer_1step2(X, Y, w1, w2)

        st[k, :] = alpha
        err.append(np.linalg.norm(R))
        err1.append(np.linalg.norm(R1))
        err_Y.append(np.linalg.norm(errY))
        rates.append(rate)
    return w2, st, err, err1, err_Y, rates

# n1 = xdim - 2 #condition number being small, but the error is high because of compressing information.
n1 = xdim + 2 #overdetermined. the underlying matrix is not invertible, so matrix is ill-conditioned. --confirm?
w, st, err, err1, err_Y, rates = ill_example1_2layers_1(W, n1, num_steps=100)

plt.figure(3)
plt.plot(err, '-b+', label='2-layer w2: BEFORE weight update')
plt.plot(err1, '-k.', label='2-layer w2: AFTER weight update')
plt.plot(err_Y, '-ro', label='2-layer w2: error of predicting Y')
plt.plot(rates, '-g<', label='rates')
plt.legend()
plt.xlabel('steps')
plt.ylabel('loss (squared norm of gradient)')
plt.yscale('log')
plt.title('two-layer linear system: the \'AFTER\' curve should always be smaller')

plt.figure(4)
markers = ['-b+', '--ko', ':g<', '--rs']
for i in range(len(markers)):#range(st.shape[1]):
    plt.plot(st[:, i], markers[i])
plt.xlabel('steps')
plt.ylabel('step-sizes')
plt.show()