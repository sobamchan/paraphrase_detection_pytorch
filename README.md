# paraphrase_detection_pytorch


## training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py train --ddir /path/to/data/ --data-cache-dir /path/to/cache/ --savedir /path/to/dump/ --bsize 128 --ft_path /path/to/ft/bin --use_cuda --epoch 10 --lr 0.01 --seed 1111
```

## evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py eval --ddir /path/to/data --model_path ./test/model.pth --use_cuda
```


## reference
`swem.py`: https://github.com/yagays/swem
