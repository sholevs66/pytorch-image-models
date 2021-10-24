""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict
from contextlib import ExitStack

import torch
from torch.utils.tensorboard.writer import SummaryWriter as TBWriter
try:
    import wandb
except ImportError:
    pass


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


class SummaryWriter():
    def __init__(self, out_dir, filename='summary.csv', log_wandb=False):
        self._filename = os.path.join(out_dir, filename)
        self._log_wandb = log_wandb
        self._exit_stack = ExitStack()
        fileobj = open(self._filename, mode='a')
        self._exit_stack.enter_context(fileobj)
        self._fileobj = fileobj
        self._writer = None
        self._tensorboard_writer = TBWriter(out_dir)
        self._exit_stack.enter_context(self._tensorboard_writer)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._exit_stack.close()

    def update_summary_dict(self, rowd):
        if self._log_wandb:
            wandb.log(rowd)

        if self._writer is None:
            self._writer = csv.DictWriter(
                self._fileobj, fieldnames=list(rowd.keys()))
            self._writer.writeheader()

        self._writer.writerow(rowd)
        self._fileobj.flush()

        epoch = rowd.pop('epoch')
        for k, v in rowd.items():
            self._tensorboard_writer.add_scalar(k, v, global_step=epoch)

    def update_summary(self, epoch, train_metrics, eval_metrics):
        rowd = OrderedDict(epoch=epoch)
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        self.update_summary_dict(rowd)
