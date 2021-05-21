#!/bin/bash

python src/exp/seqbert/pretrain.py \
    --n_dims=256 \
    --n_heads=2 \
    --n_layers=2 \
    --n_decode_layers=2 \
    --feedforward_dims=512 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --keep_prop=0.03 \
    --mask_prop=0.1 \
    --random_prop=0.02 \
    --val_keep_prop=0.2 \
    --val_mask_prop=0. \
    --val_random_prop=0.1 \
    --cls_regularization=1. \
    --crop_factor=0.4 \
    --num_workers=4 \
    --batch_size=4 \
    --learning_rate=3e-4 \
    --seq_len=100 \
    --seq_file=data/ref_genome/model_org/chrI.fna \
    --gpus=0 \

    # --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    # --train_intervals=data/ref_genome/grch38-train.bed \
    # --valid_intervals=data/ref_genome/grch38-1M-valid.bed \
    # --default_root_dir='outputs' \
    # --print_progress_freq=100 \
    # --save_checkpoint_freq=100 \
    # --val_check_interval=100 \
    # --limit_val_batches=10 \
    # --kill_param_threshold=100 \
    # --kill_grad_threshold=100 \
    # --gradient_clip_val=0.5 \
    # --gpus=0 \
    # --load_checkpoint_path=./outputs/lightning_logs/version_59124619/checkpoints/N-Step-Checkpoint_0_0.ckpt \

    # --auto_lr_find=True \
    # --accumulate_grad_batches=8 \
    # --seq_file=data/ref_genome/test-2k.fa \
    # --DEBUG_use_random_data=False \
    # --DEBUG_random_repeat_len=2 \
    # --DEBUG_random_n_repeats=100 \
