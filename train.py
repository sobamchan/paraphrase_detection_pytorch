import sys
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data import get
from models import MLP


def train(ddir: str, data_cache_dir: str, savedir: str, bsize: int,
          ft_path: str, use_cuda: bool, epoch: int, lr: float, seed: int = 1111):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    savedir = Path(savedir)
    savedir = savedir / datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir.mkdir()

    logf = open(savedir / 'log.txt', 'w')
    logf.write(' '.join(sys.argv) + '\n')

    print('Loading dataset...')
    train_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='train')
    valid_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='valid')

    print('Setting up models...')
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()

    best_acc = 0

    print('Start training...')
    for i_epoch in range(1, epoch+1):
        losses = []
        model.train()
        for tgts, sent1s, sent2s in train_dataloader:
            tgts = tgts.to(device)
            sent1s = sent1s.to(device)
            sent2s = sent2s.to(device)

            preds = model(sent1s, sent2s)
            loss = criteria(preds, tgts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        valid_losses = []
        valid_accs = []
        with torch.no_grad():
            for tgts, sent1s, sent2s in valid_dataloader:
                tgts = tgts.to(device)
                sent1s = sent1s.to(device)
                sent2s = sent2s.to(device)

                preds = model(sent1s, sent2s)
                pred_idxs = preds.argmax(dim=1).tolist()

                loss = criteria(preds, tgts)
                acc = len([1 for p, t in zip(pred_idxs, tgts.tolist()) if p == t]) / len(tgts.tolist())
                valid_losses.append(loss.item())
                valid_accs.append(acc)

        print(f'Train loss: {np.mean(losses)}')
        logf.write(f'Train loss: {np.mean(losses)}\n')

        _loss = np.mean(valid_losses)
        _acc = np.mean(valid_accs)
        print(f'Valid loss: {_loss}')
        logf.write(f'Valid loss: {_loss}\n')
        print(f'Valid accuracy: {_acc}')
        logf.write(f'Valid accuracy: {_acc}\n')

        if _acc > best_acc:
            best_acc = _acc
            print('Best acc')
            print(f'Dumping the model to {savedir}...')
            torch.save(model, savedir / 'best.pth')

        print()
        logf.write('\n')

    print(f'Dumping the model to {savedir}...')
    torch.save(model, savedir / 'model.pth')


if __name__ == '__main__':
    fire.Fire()
