#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.models.layers import Affine
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')

# Model Exponential Moving Average
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuse-bn', action='store_true', default=True,
                    help='fuse batch normalization in supported layers')
parser.add_argument('--no-fuse-bn', action='store_false', dest='fuse_bn',
                    help='do not fuse batch normalization layers')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def replace_layers(model):
    for name, layer in model.named_children():
        if isinstance(layer, timm.models.mlp_mixer.Affine):
            dim = layer.alpha.size(-1)
            new_affine = Affine(dim)
            new_affine.alpha = layer.alpha
            new_affine.beta = layer.beta

            model.add_module(name, new_affine)
            continue

        replace_layers(layer)


def fuse_bn(model):
    for name, layer in model.named_children():
        if hasattr(layer, 'fuse_bn'):
            new_layer = layer.fuse_bn()
            model.add_module(name, new_layer)
            continue

        fuse_bn(layer)


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp)
    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        # FIXME handle model default vs config num_classes more elegantly
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # optionally resume from a checkpoint
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    replace_layers(model)
    if args.fuse_bn:
        fuse_bn(model)

    model.eval()

    input_size = args.input_size or model.default_cfg['input_size']
    torch.onnx.export(model, torch.zeros(1, *input_size),
                      f'{args.model}.onnx', training=torch.onnx.TrainingMode.PRESERVE, opset_version=11)


if __name__ == '__main__':
    main()
