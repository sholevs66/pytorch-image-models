#!/bin/bash
readonly model="$1"
if [ -z ${model:x} ]; then
    echo 'must specify model'
fi
shift
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model $model /fastdata/users/imagenet/ --workers 6 \
--batch-size 256 --drop-path 0.1 --model-ema --model-ema-decay 0.99996 \
--opt adamw --opt-eps 1e-8 --weight-decay 0.05 --lr 5e-4 --warmup-lr 1e-6 --min-lr 1e-5 \
--decay-epochs 30 --warmup-epochs 5 --aa rand-m9-mstd0.5-inc1 --train-interpolation bicubic \
--use-ra-sampler \
--reprob 0.25 --mixup 0.8 --cutmix 1.0 \
$@
