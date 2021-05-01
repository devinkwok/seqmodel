"""Replicate pytorch-lightning training loop to eliminate CUDA memory error
"""
import sys
sys.path.append('./src')
import os
import time
from argparse import ArgumentParser
import torch
import numpy as np
import random
from exp.seqbert.model import SeqBERT, SeqBERTLightningModule, Counter, \
                            CheckpointEveryNSteps, bool_to_tokens, main
from exp.seqbert.pretrain import Pretrain
from exp.seqbert.pretrain_esm import PretrainESM


def move_to_device(x, args):
    device = 'cpu'
    if args.gpus > 0:
        device = 'cuda'
    (source, target, mask), (key, coord) = x
    source = source.to(device)
    target = target.to(device)
    mask = mask.to(device)
    key = key
    coord = coord.to(device)
    return (source, target, mask), (key, coord)


def validate(module, val_dl, args, n_batch=None):
    if n_batch is None:
        n_batch = args.limit_val_batches
    module.eval()
    loss = []
    with torch.no_grad():
        for i, x in enumerate(val_dl):
            x = move_to_device(x, args)
            loss.append(module.validation_step(x, i))
            if n_batch > 0 and i >= n_batch:
                break
    module.validation_epoch_end(loss)
    module.train()


def train(module, args):
    optimizer = module.configure_optimizers()
    train_dl = module.train_dataloader()
    val_dl = module.val_dataloader()

    device = 'cpu'
    if args.gpus > 0:
        device = 'cuda'
    module.to(device)

    ckpt_path = os.path.join(args.default_root_dir, 'checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    module.train()
    validate(module, val_dl, args, n_batch=1)  # sanity check
    start_time = time.time()

    total_loss = 0.
    for epoch in range(args.max_epochs):
        for i, x in enumerate(train_dl):
            x = move_to_device(x, args)
            loss = module.training_step(x, i) / args.accumulate_grad_batches
            loss.backward()
            total_loss += loss.item()
            if i % args.accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_value_(module.parameters(), args.gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad()
                elapsed_time = time.time() - start_time
                print('Epoch {} {}it, t={:.2f}, loss={:.2f}'.format(epoch, i, elapsed_time, total_loss))
                total_loss = 0.

            if (i % args.save_checkpoint_freq) == 0:
                outfile = os.path.join(ckpt_path, 'N-Step-Checkpoint_{}_{}.ckpt'.format(epoch, i))
                torch.save(module, outfile)
            
            if args.val_check_interval > 0 and (i % args.val_check_interval) == 0:
                validate(module, val_dl, args)

        validate(module, val_dl, args)


if __name__ == '__main__':
    ModelType = Pretrain
    ModelType = PretrainESM
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--deterministic', default=False, type=bool)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--gradient_clip_val', default=0.5, type=float)
    parser.add_argument('--val_check_interval', default=0, type=int)
    parser.add_argument('--limit_val_batches', default=0, type=int)
    parser.add_argument('--default_root_dir', default='.', type=str)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser = ModelType.add_model_specific_args(parser)
    args = parser.parse_args()
    print('NO PYTORCH_LIGHTNING', vars(args))

    if args.deterministic:
        seed = 0
        print('Setting seed to', seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    module = ModelType(**vars(args))
    try:
        if args.mode == 'train':
            train(module, args)
    except:
        if args.dump_file != '':
            print('Unexpected error, dumping model state to {}'.format(args.dump_file))
            torch.save(module.model, args.dump_file)
        raise
