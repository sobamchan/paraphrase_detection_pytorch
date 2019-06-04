import fire

import torch

from data import get


def eval(ddir: str, model_path: str,
         use_cuda: bool = True, ft_path: str = '/home/takeshita/mnt/DATA/NLP/fasttext_pretrained/wiki.en.bin',
         split='test'):
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = torch.load(model_path).to(device)

    dataloader = get(ddir, './test', bsize=128, ft_path=ft_path, split=split, shuffle=False)

    pred_idxs = []
    tgts = []
    for _tgts, sent1s, sent2s in dataloader:
        sent1s = sent1s.to(device)
        sent2s = sent2s.to(device)

        preds = model(sent1s, sent2s)
        pred_idxs += preds.argmax(dim=1).tolist()
        tgts += _tgts.tolist()

    assert len(pred_idxs) == len(tgts), f'{len(pred_idxs)} != {len(tgts)}'

    acc = len([1 for p, t in zip(pred_idxs, tgts) if p == t]) / len(tgts)
    print(f'Accuracy: {acc}')


if __name__ == '__main__':
    fire.Fire()
