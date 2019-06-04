import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self):
        self.linear1 = nn.Linear(300 * 2, 200)
        self.linear2 = nn.Linear(200, 300)
        self.linear3 = nn.Linear(300, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1s, sent2s):
        '''
        sent1s: [B, H]
        sent2s: [B, H]
        '''

        concated = torch.cat((sent1s, sent2s), dim=1)  # [B, H * 2]
        x = self.relu(self.linear1(concated))
        x = self.relu(self.linear2(x))
        out = self.softmax(self.linear3(x))
        return out
