import torch
import torch.nn.init as init
import math

'''this script contains our method of initialization weights'''

def init_iDelta(tensor, delta=0.01):
    '''initialize the tensor with I + \delta * random '''
    r = torch.empty(tensor.shape)
    i = init.eye_(tensor)
    # fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    # a = math.sqrt(6/(fan_in+fan_out))# adding this scaling factor doesn't seem very different. same fast.
    # return torch.mul(a, i) + torch.mul(delta, init.xavier_uniform(r))
    return i + torch.mul(delta, init.uniform_(r, 0, 1.0))
    # return i + torch.mul(delta, init.xavier_uniform(r))
    # return i - torch.mul(delta, init.xavier_uniform(r))

# w = init_iDelta(torch.empty(3, 5), 0.01)
# w = init_iDelta(torch.empty(10, 5), 0.01)
# print(w)


