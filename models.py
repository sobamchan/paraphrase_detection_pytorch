import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, trial):
        super().__init__()
        self.layers = []
        self.dropouts = []
        nlayers = trial.suggest_int('nlayers', 3, 8)
        dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
        input_dim = 300 * 2

        for i in range(nlayers):
            output_dim = int(trial.suggest_categorical(f'n_units_l{i}', list(range(100, 1000, 100))))
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, 2))

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
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.relu(layer(x))
            x = dropout(x)

        return x
