import sys
from typing import List
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import optuna

import torch
import torch.nn as nn
import torch.optim as optim

from data import get
from models_optuna import MLP
from logger import get_logger


global model


def train(ddir: str, data_cache_dir: str, _savedir: str, bsize: int,
          ft_path: str, use_cuda: bool, epoch: int, seed: int = 1111,
          use_optuna: bool = True, n_trials: int = 100,  # with optuna
          lr: float = 1e-5, output_dims: List = [100, 200, 100], dropout: float = 0.5  # without optuna
          ):

    print('Loading dataset...')
    train_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='train')
    valid_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='valid')

    savedir = Path(_savedir)
    savedir = savedir / datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir.mkdir()
    logf = open(savedir / 'log.txt', 'w')
    logger = get_logger(logf, False)
    logger(' '.join(sys.argv))

    def objective(trial, save=False):
        global model

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        if trial:
            if not isinstance(trial, optuna.Trial):
                # Best
                lr = trial.params['lr']
                nlayers = trial.params['nlayers']
                dropout = trial.params['dropout']
                output_dims = [trial.params[f'n_units_l{i}'] for i in range(nlayers)]
            else:
                # optuna settings
                lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
                nlayers = trial.suggest_int('nlayers', 3, 8)
                dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
                output_dims = [
                        int(trial.suggest_categorical(f'n_units_l{i}', list(range(100, 1000, 100))))
                        for i in range(nlayers)
                        ]
        else:
            nlayers = len(output_dims)
        # else:
        #     lr = 1e-5
        #     output_dims = [200, 100, 200]
        #     nlayers = len(output_dims)
        #     dropout = 0.5

        # print('Setting up models...')
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = MLP(nlayers, dropout, output_dims).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criteria = nn.CrossEntropyLoss()

        best_acc = 0
        n_fail_in_a_raw = 0
        limit_n_fail_in_a_raw = 10

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

            logger(f'Train loss: {np.mean(losses)}')

            _loss = np.mean(valid_losses)
            _acc = np.mean(valid_accs)
            logger(f'Valid loss: {_loss}')
            logger(f'Valid accuracy: {_acc}')

            if _acc > best_acc:
                best_acc = _acc
                n_fail_in_a_raw = 0
            else:
                n_fail_in_a_raw += 1

            if n_fail_in_a_raw >= limit_n_fail_in_a_raw:
                break

            logger('\n')

        return best_acc

    if use_optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        logger(f'Number of finished trials: {len(study.trials)}', True)
        logger('Best trial:', True)

        # Dump models with best trial
        final_acc = objective(study.best_trial)

        logger(f'    Value: {study.best_trial.value}', True)
        logger(f'    Params: ', True)
        for k, v in study.best_trial.params.items():
            logger(f'      {k}: {v}', True)

        logger(f'Final accuracy: {final_acc}', True)
    else:
        objective(None)

    # Dump model
    global model
    logger(f'Dumping the model to {savedir}...', True)
    torch.save(model, savedir / 'best.pth')

    logf.close()


if __name__ == '__main__':
    fire.Fire()
