import torch.nn as nn
import torch.nn.init as init
from problem import xdim, ydim

(input_size, output_size) = (xdim, ydim)

hidden_size = input_size + 10

def nn_1layer_Xavier():
    model = nn.Linear(input_size, output_size)
    init.xavier_uniform_(model.weight)
    init.constant_(model.bias, 0.1)
    return model

class NN_2layers_Xavier(nn.Module):
    def __init__(self):
        super(NN_2layers_Xavier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class NN_nlayers_Xavier(nn.Module):
    def __init__(self, num_hidden_layers):
        super(NN_nlayers_Xavier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)

        self.num_hidden_layers = num_hidden_layers
        self.fc_hidden = {}
        for i in range(self.num_hidden_layers):
            self.fc_hidden[i] = nn.Linear(hidden_size, hidden_size)
            init.xavier_uniform_(self.fc_hidden[i].weight)

        self.fc_last = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc_last.weight)

    def forward(self, x):
        out = self.fc1(x)
        for i in range(self.num_hidden_layers):
            out = self.fc_hidden[i](out)
        out = self.fc_last(out)
        return out

class NN_2layers_Uniform(nn.Module):
    def __init__(self):
        super(NN_2layers_Uniform, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, output_size)
        init.uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class NN_nlayers_Uniform(nn.Module):
    def __init__(self, num_hidden_layers):
        super(NN_nlayers_Uniform, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.uniform_(self.fc1.weight)

        self.num_hidden_layers = num_hidden_layers
        self.fc_hidden = {}
        for i in range(self.num_hidden_layers):
            self.fc_hidden[i] = nn.Linear(hidden_size, hidden_size)
            init.uniform_(self.fc_hidden[i].weight)

        self.fc_last = nn.Linear(hidden_size, output_size)
        init.uniform_(self.fc_last.weight)

    def forward(self, x):
        out = self.fc1(x)
        for i in range(self.num_hidden_layers):
            out = self.fc_hidden[i](out)
        out = self.fc_last(out)
        return out

def freeze_a_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class NN_2layers_Freeze(nn.Module):
    def __init__(self):
        super(NN_2layers_Freeze, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)

        #freeze layer 1
        freeze_a_layer(self.fc1)

        self.fc2 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class NN_nlayers_Freeze(nn.Module):
    def __init__(self, num_hidden_layers):
        super(NN_nlayers_Freeze, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        freeze_a_layer(self.fc1)

        self.num_hidden_layers = num_hidden_layers
        self.fc_hidden = {}
        for i in range(self.num_hidden_layers):
            self.fc_hidden[i] = nn.Linear(hidden_size, hidden_size)
            init.xavier_uniform_(self.fc_hidden[i].weight)
            freeze_a_layer(self.fc_hidden[i])

        self.fc_last = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc_last.weight)

    def forward(self, x):
        out = self.fc1(x)
        for i in range(self.num_hidden_layers):
            out = self.fc_hidden[i](out)
        out = self.fc_last(out)
        return out