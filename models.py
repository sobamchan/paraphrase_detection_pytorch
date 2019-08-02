import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, nlayers, dropouts, output_dims):
        super().__init__()
        self.layers = []
        self.dropouts = []

        input_dim = 300 * 2

        for output_dim in output_dims:
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropouts[0]))
            self.dropouts.append(nn.Dropout(dropouts[1]))
            input_dim = output_dim

        self.final_fc = nn.Linear(input_dim, 2)

        for idx, layer in enumerate(self.layers):
            setattr(self, f'fc{idx}', layer)

        for idx, dropout in enumerate(self.dropouts):
            setattr(self, f'dropout{idx}', dropout)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1s, sent2s):
        '''
        sent1s: [B, H]
        sent2s: [B, H]
        '''
        concated = torch.cat((sent1s, sent2s), dim=1)  # [B, H * 2]

        x = concated
        x = self.dropouts[0](x)

        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.final_fc(x)
        x = self.dropouts[1](x)

        return x
