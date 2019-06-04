from pathlib import Path

import fire
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data import get
from models import MLP


def train(ddir: str, savedir: str, bsize: int, ft_path: str, use_cuda: bool, epoch: int, lr: float, seed: int = 1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    print('Loading dataset...')
    dataloader = get(ddir, savedir, bsize, ft_path)

    print('Setting up models...')
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()

    print('Start training...')
    for i_epoch in range(1, epoch+1):
        losses = []
        for tgts, sent1s, sent2s in dataloader:
            tgts = tgts.to(device)
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)

            preds = model(sent1s, sent2s)
            loss = criteria(preds, tgts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f'Train loss: {np.mean(losses)}')

    print('Dumping the model...')
    savedir = Path(savedir)
    torch.save(model, savedir / 'model.pth')


if __name__ == '__main__':
    fire.Fire()
