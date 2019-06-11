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
from models_optuna import MLP


def train(ddir: str, data_cache_dir: str, _savedir: str, bsize: int,
          ft_path: str, use_cuda: bool, epoch: int, seed: int = 1111):

    print('Loading dataset...')
    train_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='train')
    valid_dataloader = get(ddir, data_cache_dir, bsize, ft_path, split='valid')

    def objective(trial, save=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        # savedir = Path(_savedir)
        # savedir = savedir / datetime.now().strftime('%Y%m%d_%H%M%S')
        # savedir.mkdir()

        # logf = open(savedir / 'log.txt', 'w')
        # logf.write(' '.join(sys.argv) + '\n')

        if trial:
            if save:
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
            lr = 1e-5
            output_dims = [200, 100, 200]
            nlayers = len(output_dims)
            dropout = 0.5

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
                n_fail_in_a_raw = 0
                # print('Best acc')
                # print(f'Dumping the model to {savedir}...')
                # torch.save(model, savedir / 'best.pth')
            else:
                n_fail_in_a_raw += 1

            if n_fail_in_a_raw >= limit_n_fail_in_a_raw:
                break

            # print()
            # logf.write('\n')

        # print(f'Dumping the model to {savedir}...')
        # torch.save(model, savedir / 'model.pth')
        return best_acc

    use_optuna = True
    if use_optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trial:')

        # Dump models with best trial
        final_acc = objective(study.best_trial, save=True)

        print(f'    Value: {study.best_trial.value}')
        print(f'    Params: ')
        for k, v in study.best_trial.params.items():
            print(f'      {k}: {v}')

        print(f'Final accuracy: {final_acc}')
    else:
        objective(None, save=True)


if __name__ == '__main__':
    fire.Fire()
