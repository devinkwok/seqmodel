#!/bin/bash
#SBATCH --job-name=%a-hp-lrb-4x4
#SBATCH --account=def-quanlong          # needed for resource billing if using compute canada
#SBATCH --time=12:00:00                 # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=2               # number of cores
#SBATCH --gres=gpu:p100:1              # type and number of GPU(s) per node
#SBATCH --mem=8000                      # max memory (default unit is MB) per node
#SBATCH --output=%A_%a-hp-lrb-4x4.out       # file name for the output
#SBATCH --error=%A_%a-hp-lrb-4x4.err        # file name for errors
#SBATCH --array=1,11,21                     # number and index of job arrays (from 1)

echo $SLURM_ARRAY_TASK_ID

SRC_DIR=~/proj/seqbert-pretrain        # root dir of src
DATA_DIR=~/data/seqbert-pretrain       # load .tar.gz data
OUT_DIR=~/scratch/hp-lrb-4x4     # save outputs
RUN_DIR=$SLURM_TMPDIR           # save env and tmp data

## set working dir to src root for python imports
cd $SRC_DIR

## load modules
module load nixpkgs/16.09  gcc/7.3.0 cuda/10.1 cudnn/7.6.5 python/3.7.4

## setup virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

## install project dependencies
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.txt
## these dependencies need to be downloaded
pip install pyfaidx pytorch-lightning==1.1.6

## extract all .tar.gz data files
tar xzvf $DATA_DIR/*.tar.gz -C $SLURM_TMPDIR

# hparams
python ./src/exp/seqbert/pretrain.py \
    --hparam_search_idx=$SLURM_ARRAY_TASK_ID \
    --n_dims=512 \
    --n_heads=4 \
    --n_layers=4 \
    --n_decode_layers=2 \
    --feedforward_dims=2048 \
    --batch_size=8 \
    --learning_rate=2e-6 \
    --seq_len=1000 \
    --dropout=0.05 \
    --adam_beta_1=0.9 \
    --adam_beta_2=0.99 \
    --adam_eps=1e-6 \
    --weight_decay=0.01 \
    --keep_prop=0.01 \
    --mask_prop=0.13 \
    --random_prop=0.01 \
    --cls_regularization=0. \
    --num_workers=4 \
    --print_progress_freq=500 \
    --save_checkpoint_freq=5000 \
    --val_check_interval=5000 \
    --limit_val_batches=1000 \
    --seq_file=$SLURM_TMPDIR/data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=$SLURM_TMPDIR/data/ref_genome/grch38-train.bed \
    --valid_intervals=$SLURM_TMPDIR/data/ref_genome/grch38-1M-valid.bed \
    --default_root_dir=$OUT_DIR \
    --gradient_clip_val=0.5 \
    --accumulate_grad_batches=1 \
    --deterministic=True \

## clean up by stopping virtualenv
deactivate
