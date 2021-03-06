import sys
from typing import List
from datetime import datetime
from pathlib import Path
from collections import Counter

import fire
import numpy as np
import optuna
import lineflow.cross_validation as lfcv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get, get_collate_fn
from models import MLP
from logger import get_logger


global model


def train(ddir: str, data_cache_dir: str, _savedir: str, bsize: int,
          ft_path: str, use_cuda: bool, epoch: int, seed: int = 1111,
          use_optuna: bool = True, n_trials: int = 100,  # with optuna
          lr_lower_bound: float = 1e-5, lr_upper_bound: float = 1e-1,
          nlayers_lower_bound: int = 3, nlayers_upper_bound: int = 10,
          dropout_lower_bound: int = 0.2, dropout_upper_bound: int = 0.5,
          odim_start: int = 100, odim_end: int = 1000, odim_step: int = 100,
          lr: float = 1e-5, output_dims: List = [100, 200, 100], dropout: float = 0.5  # without optuna
          ):

    savedir = Path(_savedir)
    data_cache_dir = Path(data_cache_dir)
    savedir = savedir / datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir.mkdir()
    logf = open(savedir / 'log.txt', 'w')
    logger = get_logger(logf, False)
    logger(' '.join(sys.argv))

    logger('Loading dataset...', True)
    dataset = get(ddir, ft_path, split='train')
    train_size = int(len(dataset) * 0.8)
    train_dataset, valid_dataset = lfcv.split_dataset_random(dataset, train_size, 42)
    logger(f'Train dataset size: {len(train_dataset)}', True)
    logger(f'Valid dataset size: {len(valid_dataset)}', True)
    train_dataloader = DataLoader(
            train_dataset.save(data_cache_dir / 'swem.train.cache'),
            batch_size=bsize,
            shuffle=True,
            num_workers=4,
            collate_fn=get_collate_fn()
            )
    valid_dataloader = DataLoader(
            valid_dataset.save(data_cache_dir / 'swem.valid.cache'),
            batch_size=bsize,
            shuffle=False,
            num_workers=4,
            collate_fn=get_collate_fn()
            )

    # Prepare weights for loss function.
    logger("Preparing weight for loss function.", True)
    labels = [sample[0] for sample in train_dataset]
    label_counter = Counter(labels)  # {1: number of class 1 samples, 0: number of class 0 samples,}
    loss_weight = torch.zeros(len(label_counter))
    loss_weight[0] = 1 / label_counter[0]
    loss_weight[1] = 1 / label_counter[1]
    logger(f"Number of samples for each classes: {label_counter}", True)
    logger(f"Weight: loss_weight: {loss_weight}", True)

    def objective(trial: optuna.Trial,  # with optuna
                  lr: int = None, output_dims: List = None, dropout: float = None  # without optuna
                  ):
        assert not (trial is not None and lr is not None)
        assert not (trial is not None and output_dims is not None)
        assert not (trial is not None and dropout is not None)
        assert not (trial is None and lr is None)
        assert not (trial is None and output_dims is None)
        assert not (trial is None and dropout is None)

        global model

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        if trial is not None:
            if not isinstance(trial, optuna.Trial):
                # Best
                lr = trial.params['lr']
                nlayers = trial.params['nlayers']
                dropouts = trial.params['dropout']
                output_dims = [trial.params[f'n_units_l{i}'] for i in range(nlayers)]
            else:
                # In study.
                logger(f'{"-" * 10} Trial #{trial.number} {"-" * 10}')

                # optuna settings
                # lr = trial.suggest_uniform('lr', lr_lower_bound, lr_upper_bound)
                lr = trial.suggest_categorical('lr', [1e-3, 3e-4, 2e-4, 1e-5])
                nlayers = trial.suggest_int('nlayers', nlayers_lower_bound, nlayers_upper_bound)
                dropouts = [
                        trial.suggest_categorical(f'dropout_l{i}', [0.2, 0.5, 0.7])
                        for i in range(2)
                        ]
                output_dims = [
                        int(trial.suggest_categorical(f'n_units_l{i}', list(range(odim_start, odim_end, odim_step))))
                        for i in range(nlayers)
                        ]
        else:
            nlayers = len(output_dims)

        logger('Setting up models...')
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = MLP(nlayers, dropouts, output_dims).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criteria = nn.CrossEntropyLoss(weight=loss_weight.to(device))

        best_acc = 0
        n_fail_in_a_raw = 0
        limit_n_fail_in_a_raw = 5

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

            logger(f"{'-' * 25}\n")

        return best_acc

    if use_optuna:
        logger('With optuna.', True)
        logger("Let's go.", True)
        study = optuna.create_study(direction='maximize')

        try:
            study.optimize(objective, n_trials=n_trials)
        except KeyboardInterrupt:
            logger('Keyboard Interrupted', True)

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
        logger('Without optuna.', True)
        objective(None, lr, output_dims, dropout)

    # Dump model
    global model
    logger(f'Dumping the model to {savedir}...', True)
    torch.save(model, savedir / 'best.pth')

    logf.close()


if __name__ == '__main__':
    fire.Fire()
