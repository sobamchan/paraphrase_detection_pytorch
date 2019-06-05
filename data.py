import os.path as osp
from typing import List, Dict
from pathlib import Path
from collections import Counter
import pickle
import random

import fire
import spacy
import lineflow as lf
import fastText

import torch
from torch.utils.data import DataLoader

from swem import SWEM

NLP = spacy.load('en_core_web_sm')
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'


def sent_preprocess(swem):
    def _f(sent):
        tokens = [
                token.text.lower()
                for token in NLP(sent)
                if not token.is_space
                ]
        vec = swem.average_pooling(tokens)
        return vec
    return _f


def build_vocab(tokens: List, cache: str, max_size: int = 5000) -> (Dict, List):
    if not osp.isfile(cache):
        counter = Counter(tokens)
        words, _ = zip(*counter.most_common(max_size))
        words = [PAD_TOKEN, UNK_TOKEN] + list(words)
        token_to_index = dict(zip(words, range(len(words))))
        with open(cache, 'wb') as f:
            pickle.dump((token_to_index, words), f)
    else:
        with open(cache, 'rb') as f:
            token_to_index, words = pickle.load(f)

    return token_to_index, words


def get_collate_fn():

    def _f(batch):
        tgts, sent1s, sent2s = zip(*batch)
        return (
                torch.LongTensor(tgts),
                torch.FloatTensor(sent1s),
                torch.FloatTensor(sent2s)
                )

    return _f


def get(ddir: str, savedir: str, bsize: int, ft_path: str, split: str):
    random.seed(1111)
    ddir = Path(ddir)
    savedir = Path(savedir)

    ft_model = fastText.load_model(ft_path)
    swem = SWEM(ft_model)

    quality = lf.TextDataset(str(ddir / (f'quality.{split}.txt'))).map(int)
    sent1 = lf.TextDataset(str(ddir / (f'sent1.{split}.txt'))).map(sent_preprocess(swem))
    sent2 = lf.TextDataset(str(ddir / (f'sent2.{split}.txt'))).map(sent_preprocess(swem))

    ds = lf.zip(quality, sent1, sent2)

    dataloader = DataLoader(
            ds.save(savedir / f'swem.{split}.cache'),
            batch_size=bsize,
            shuffle=True,
            num_workers=4,
            collate_fn=get_collate_fn()
            )
    return dataloader


def test_get(ddir: str, savedir: str, bsize: int, ft_path: str):
    ddir = Path(ddir)
    savedir = Path(savedir)

    ft_model = fastText.load_model(ft_path)
    swem = SWEM(ft_model)

    quality = lf.TextDataset(str(ddir / ('quality.test.txt'))).map(int)
    sent1 = lf.TextDataset(str(ddir / ('sent1.test.txt'))).map(sent_preprocess(swem))
    sent2 = lf.TextDataset(str(ddir / ('sent2.test.txt'))).map(sent_preprocess(swem))

    ds = lf.zip(quality, sent1, sent2)

    test_dataloader = DataLoader(
            ds.save(savedir / 'swem.test.cache'),
            batch_size=bsize,
            shuffle=False,
            num_workers=4,
            collate_fn=get_collate_fn()
            )

    return test_dataloader


if __name__ == '__main__':
    fire.Fire()
