import matplotlib.pyplot as plt
import numpy as np

# plt.figure(1)
# plt.subplot(221)
# plt.plot(delta1, '--b+', label='delta_1')
# plt.plot(delta2, '-bs', label='delta_2')
# plt.yscale('log')
# plt.legend()
#
# plt.subplot(222)
# plt.plot(grad1, '--k+', label='grad_1')
# plt.plot(grad2, '-ks', label='grad_2')
# plt.yscale('log')
#
# plt.subplot(223)
# plt.plot(nn.step_sizes, '-ko', label='step-sizes')
# plt.title('step-sizes')
#
# plt.subplot(224)
# diff = np.asarray(nn.delta1_before) - np.asarray(nn.delta1_after1)
# # diff = np.asarray(nn.delta1_before) - np.asarray(nn.delta1_after2)
# plt.plot(diff, '--bs', label='difference')
# # plt.yscale('log')
# plt.title('before/after norm of delta')
# plt.legend()
#
# plt.figure(2)
# plt.subplot(221)
# plt.plot(nn.delta1_before, '--r+', label='||delta1|| before')
# plt.plot(nn.delta1_after1, '-bo', label='||delta1|| after1')
# # plt.plot(nn.delta1_after2, '-.k*', label='||delta1|| after2(from measure on the new w)')
# # this should be 0; but why not?
# print('difference between two new deltas is:', norm(np.asarray(nn.delta1_after1) - np.asarray(nn.delta1_after2)))
# plt.title('before/after norm of delta')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()
#
# print('#negatitive{} out of {}'.format(len(diff[np.where(diff < 0)]), len(diff)))
# assert (all(d >= 0.0 for d in diff))


def proc_data(learning_rate):
    loss_const = np.load('loss_const_' + str(learning_rate) + '.npy')
    #return np.mean(loss_const, axis=0), 'const lr=' + str(learning_rate)
    return loss_const, 'const lr=' + str(learning_rate)

# plt.subplot(121)
fig = plt.figure()
markers=['-k+', '--rx', '-.bs', ':g<', '--kh', '--yp']
for i, lr in enumerate([0.0625, 0.1, 0.2, 0.25, 0.3, 0.4]):
    loss, label1 = proc_data(learning_rate=lr)
    plt.plot(loss, markers[i], label=str(lr))

plt.legend()
plt.yscale('log')
plt.xlabel('epochs (x 10)')
plt.ylabel('loss')
plt.show()
fig.savefig('const_lr.png')


# plt.subplot(122)
# plt.plot(nn.ratios, '-bo'),
# plt.title('contraction ratios')
# print('average contraction ratio:', np.mean(nn.ratios))