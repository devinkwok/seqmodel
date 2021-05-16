#!/bin/bash

# python src/exp/seqbert/visualize_repr.py \
#     --n_dims=512 \
#     --n_heads=4 \
#     --n_layers=4 \
#     --n_decode_layers=2 \
#     --feedforward_dims=1024 \
#     --dropout=0.0 \
#     --position_embedding=Sinusoidal \
#     --keep_prop=0.0 \
#     --mask_prop=0.0 \
#     --random_prop=0.0 \
#     --cls_regularization=1. \
#     --crop_factor=0.4 \
#     --num_workers=4 \
#     --batch_size=4 \
#     --learning_rate=3e-4 \
#     --seq_len=1000 \
#     --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
#     --train_intervals=data/ref_genome/grch38-train.bed \
#     --valid_intervals=data/ref_genome/grch38-1M-valid.bed \
#     --default_root_dir='outputs' \
#     --print_progress_freq=100 \
#     --save_checkpoint_freq=100 \
#     --val_check_interval=100 \
#     --limit_val_batches=10 \
#     --kill_param_threshold=100 \
#     --kill_grad_threshold=100 \
#     --dump_file='' \
#     --gradient_clip_val=0.5 \
#     --gpus=0 \
#     --load_checkpoint_path=/home/devin/checkpoints/dump \
#     --load_dump=True \

    # --n_class=9 \
    # --use_pl=True \
    # --load_checkpoint_path=/home/devin/checkpoints/ft8x8deepsea \

python src/exp/seqbert/visualize_repr.py \
    --n_dims=512 \
    --n_heads=8 \
    --n_layers=8 \
    --n_decode_layers=2 \
    --feedforward_dims=1024 \
    --dropout=0.0 \
    --position_embedding=Sinusoidal \
    --keep_prop=0.0 \
    --mask_prop=0.0 \
    --random_prop=0.0 \
    --cls_regularization=1. \
    --crop_factor=0.4 \
    --num_workers=4 \
    --batch_size=4 \
    --learning_rate=3e-4 \
    --seq_len=1000 \
    --seq_file=data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=data/ref_genome/grch38-train.bed \
    --valid_intervals=data/ref_genome/grch38-1M-valid.bed \
    --default_root_dir='outputs' \
    --print_progress_freq=100 \
    --save_checkpoint_freq=100 \
    --val_check_interval=100 \
    --limit_val_batches=10 \
    --kill_param_threshold=100 \
    --kill_grad_threshold=100 \
    --dump_file='' \
    --gradient_clip_val=0.5 \
    --gpus=0 \
    --load_checkpoint_path=/home/devin/checkpoints/ft8x8deepsea \
    --use_pl=True \
    --n_class=919 \

    # --use_esm=True \
    # --load_protein_model=True \
