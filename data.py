import os.path as osp
from typing import List, Dict
from pathlib import Path
from collections import Counter
import pickle

import fire
import spacy
import lineflow as lf
import fastText

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


def get(ddir: str, split: str, savedir: str, bsize: int, ft_path: str):
    ddir = Path(ddir)
    savedir = Path(savedir)

    ft_model = fastText.load_model(ft_path)
    swem = SWEM(ft_model)

    quality = lf.TextDataset(str(ddir / ('quality.%s.txt' % split))).map(int)
    sent1 = lf.TextDataset(str(ddir / ('sent1.%s.txt' % split))).map(sent_preprocess(swem))
    sent2 = lf.TextDataset(str(ddir / ('sent2.%s.txt' % split))).map(sent_preprocess(swem))

    ds = lf.zip(quality, sent1, sent2)
    train_dataloader = DataLoader(
            ds.save(savedir / 'swem.train.cache'),
            batch_size=bsize,
            shuffle=True,
            num_workers=4
            )

    return ds


if __name__ == '__main__':
    fire.Fire()
