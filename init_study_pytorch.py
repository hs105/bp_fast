import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from problem import gen_data
from models import nn_1layer_Xavier, NN_2layers_Xavier, NN_nlayers_Xavier, \
    NN_2layers_Uniform, NN_nlayers_Uniform, \
    NN_2layers_Freeze, NN_nlayers_Freeze, \
    nn_1layer_iDelta, NN_2layers_iDelta, NN_nlayers_iDelta


'''initialization experiment using pytorch'''

#todo: see if reducing step-size for 10-layer net will lead to better curve.

num_epochs = 1500
learning_rate = 0.1#tuned a bit
nsamples = 1000

def train(model, step_size = learning_rate):

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #train the model
    a_losses = []
    for epoch in range(num_epochs):
        X, Y = gen_data(nsamples)

        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(X).float()
        targets = torch.from_numpy(Y).float()

        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #backward pass to train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))

        a_losses.append(loss.item())
    return a_losses



# a_losses_1layer = train(nn_1layer_Xavier())
a_losses_2layers = train(NN_2layers_Xavier())
a_losses_10layers = train(NN_nlayers_Xavier(10))
a_losses_20layers = train(NN_nlayers_Xavier(20))

# a_losses_2layer_uniform = train(NN_2layers_Uniform())
# a_losses_10layers_uniform = train(NN_nlayers_Uniform(10))

nn2_freeze = NN_2layers_Freeze()
#check point: the w1 if it's frozen
w1_before = nn2_freeze.fc1.weight.clone()#clone produces another object
w2_before = nn2_freeze.fc2.weight.clone()
a_losses_2layers_freeze = train(nn2_freeze)
w1_after = nn2_freeze.fc1.weight
w2_after = nn2_freeze.fc2.weight
# this shows w1 is indeed frozen, but w2 is not
print('change in w1:', torch.norm(w1_before - w1_after))
print('change in w2:', torch.norm(w2_before - w2_after))

# a_losses_5layers_freeze = train(NN_nlayers_Freeze(5))

# a_losses_10layers_freeze = train(NN_nlayers_Freeze(10))
# a_losses_1layer_iDelta=train(nn_1layer_iDelta())
a_losses_2layers_iDelta = train(NN_2layers_iDelta())
a_losses_10layers_iDelta = train(NN_nlayers_iDelta(10))
a_losses_20layers_iDelta = train(NN_nlayers_iDelta(20))

#plot graphs
# plt.plot(a_losses_1layer, '-b+', label='1 layer -- xavier')
plt.plot(a_losses_2layers, '--ro', label = '2 layers -- xavier')
plt.plot(a_losses_10layers, '-.rp', label = '10 layers -- xavier')
plt.plot(a_losses_20layers, ':rx', label='20 layers -- xavier')

# plt.plot(a_losses_2layer_uniform, '-yh', label= '2 layers --uniform')
# plt.plot(a_losses_10layers_uniform, '-mo', label= '10 layers --uniform')#diverges

# plt.plot(a_losses_2layers_freeze, '-g+', label= '2 layers --freeze')
# plt.plot(a_losses_5layers_freeze, ':mo', label = '5 layers -- freeze')
# plt.plot(a_losses_10layers_freeze, '--gs', label = '10 layers -- freeze')

# plt.plot(a_losses_1layer_iDelta, '-yh', label='1 layer -- iDelta')
plt.plot(a_losses_2layers_iDelta, '--bp', label='2 layers -- iDelta')
plt.plot(a_losses_10layers_iDelta, '-bx', label='10 layers -- iDelta')
plt.plot(a_losses_20layers_iDelta, '-m.', label='20 layers -- iDelta')

plt.yscale('log')
plt.legend()
plt.show()



