# import sys
# from datetime import datetime
# from pathlib import Path

import fire
import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from data import get
from models import MLP


# def train(ddir: str, data_cache_dir: str, _savedir: str, bsize: int,
#           ft_path: str, use_cuda: bool, epoch: int, lr: float, seed: int = 1111):
def train(ddir: str, data_cache_dir: str, _savedir: str, bsize: int,
          ft_path: str, use_cuda: bool, epoch: int, seed: int = 1111):

    # print('Loading dataset...')
    train_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='train')
    valid_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='valid')

    def objective(trial):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # savedir = Path(_savedir)
        # savedir = savedir / datetime.now().strftime('%Y%m%d_%H%M%S')
        # savedir.mkdir()

        # logf = open(savedir / 'log.txt', 'w')
        # logf.write(' '.join(sys.argv) + '\n')

        # optuna settings
        lr = trial.suggest_uniform('lr', 1e-5, 1e-1)

        # print('Setting up models...')
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = MLP(trial).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criteria = nn.CrossEntropyLoss()

        best_acc = 0

        # print('Start training...')
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

            # print(f'Train loss: {np.mean(losses)}')
            # logf.write(f'Train loss: {np.mean(losses)}\n')

            # _loss = np.mean(valid_losses)
            _acc = np.mean(valid_accs)
            # print(f'Valid loss: {_loss}')
            # logf.write(f'Valid loss: {_loss}\n')
            # print(f'Valid accuracy: {_acc}')
            # logf.write(f'Valid accuracy: {_acc}\n')

            if _acc > best_acc:
                best_acc = _acc
                # print('Best acc')
                # print(f'Dumping the model to {savedir}...')
                # torch.save(model, savedir / 'best.pth')

            # print()
            # logf.write('\n')

        # print(f'Dumping the model to {savedir}...')
        # torch.save(model, savedir / 'model.pth')
        return best_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    trial = study.best_trial

    print(f'    Value: {trial.value}')
    print(f'    Params: ')
    for k, v in trial.params.items():
        print(f'      {k}: {v}')

    # return objective


if __name__ == '__main__':
    fire.Fire()
