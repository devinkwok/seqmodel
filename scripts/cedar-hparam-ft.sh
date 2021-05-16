#!/bin/bash
#SBATCH --job-name=hparam-seqbert-ft-8x8
#SBATCH --account=def-quanlong          # needed for resource billing if using compute canada
#SBATCH --time=01:00:00                 # max walltime in D-HH:MM or HH:MM:SS
#SBATCH --cpus-per-task=4               # number of cores
#SBATCH --gres=gpu:v100l:1              # type and number of GPU(s) per node
#SBATCH --mem=8000                      # max memory (default unit is MB) per node
#SBATCH --output=%A_%a-ft-8x8.out      # file name for the output
#SBATCH --error=%A_%a-ft-8x8.err        # file name for errors
                                        # %j gets replaced by the job number

echo $SLURM_ARRAY_TASK_ID

## project name
NAME_DIR=seqbert-ft-deepsea
SRC_DIR=~/proj/$NAME_DIR        # root dir of src
DATA_DIR=~/data/$NAME_DIR       # load .tar.gz data
RUN_DIR=$SLURM_TMPDIR           # save env and tmp data
OUT_DIR=~/scratch/$NAME_DIR     # save outputs

## make output dir if does not exist
mkdir -p $OUT_DIR
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
tar xzvf $DATA_DIR/*.tar.gz -C $RUN_DIR

# hparams
python ./src/exp/seqbert/pretrain.py \
    --hparam_search_idx=$SLURM_ARRAY_TASK_ID \
    --n_dims=512 \
    --n_heads=8 \
    --n_layers=8 \
    --n_decode_layers=2 \
    --feedforward_dims=2048 \
    --position_embedding=Sinusoidal \
    --batch_size=16 \
    --learning_rate=1e-6 \
    --seq_len=1000 \
    --dropout=0.1 \
    --adam_beta_1=0.9 \
    --adam_beta_2=0.99 \
    --adam_eps=1e-6 \
    --weight_decay=0.01 \
    --keep_prop=0.01 \
    --mask_prop=0.13 \
    --random_prop=0.01 \
    --cls_regularization=0. \
    --seq_len_source_multiplier=2. \
    --crop_factor=0.45 \
    --seq_len_sample_freq=0.25 \
    --num_workers=8 \
    --print_progress_freq=500 \
    --save_checkpoint_freq=5000 \
    --val_check_interval=5000 \
    --limit_val_batches=1000 \
    --seq_file=$RUN_DIR/data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa \
    --train_intervals=$RUN_DIR/data/ref_genome/grch38-train.bed \
    --valid_intervals=$RUN_DIR/data/ref_genome/grch38-1M-valid.bed \
    --default_root_dir=$OUT_DIR \
    --gradient_clip_val=0.5 \
    --kill_param_threshold=10000. \
    --kill_grad_threshold=10000. \
    --dump_file=$OUT_DIR/model-dump.pt \
    --accumulate_grad_batches=8 \
    --deterministic=True \
    --use_esm=True \

    # --load_checkpoint_path=$OUT_DIR/lightning_logs/version_55300997/checkpoints/N-Step-Checkpoint_0_170000.ckpt \

## clean up by stopping virtualenv
deactivate
