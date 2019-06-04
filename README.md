# paraphrase_detection_pytorch


## training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py train --ddir /path/to/data --savedir ./test --bsize 128 --ft_path /path/to/ft/bin --use_cuda --epoch 5 --lr 0.01
```

## evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py eval --ddir /path/to/data --model_path ./test/model.pth --use_cuda
```


## reference
`swem.py`: https://github.com/yagays/swem
