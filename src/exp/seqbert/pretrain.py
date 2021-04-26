import sys
sys.path.append('./src')
import os.path
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel import INDEX_TO_BASE
from seqmodel.functional.mask import generate_mask, mask_randomize, mask_fill, mask_select
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.iterseq import StridedSequence, bed_from_file, FastaFile
from seqmodel.functional import Compose, bioseq_to_index, permute
from seqmodel.functional.log import prediction_histograms, normalize_histogram, \
                            summarize, correct, accuracy_per_class, accuracy, \
                            summarize_weights_and_grads, tensor_stats_str
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT, SeqBERTLightningModule, Counter, \
                            CheckpointEveryNSteps, bool_to_tokens, main


class PretrainBatchProcessor():

    """Object to hold hyperparameters for forming batches for BERT pretraining,
    from sequence data collected by a `torch.nn.DataLoader`.
    The `collate` function is passed to `collate_fn` parameter in `torch.nn.DataLoader`.
        seq_len: target length of output sequences
        min_len: minimum source length after cropping
        max_len: maximum source length after cropping
        offset_min: minimum index of sequence midpoint when shifted for output
        offset_max: maximum index of sequence midpoint when shifted for output
        mask_prop: proportion of sequence positions (excluding CLS and SEP tokens)
            to assign `exp.seqbert.pretrain.MASK_INDEX` (masked for prediction loss)
        random_prop: proportion of sequence positions (excluding CLS and SEP tokens)
            to assign `exp.seqbert.pretrain.RANDOM_INDEX` (randomized for prediction loss)
        keep_prop: proportion of sequence positions (excluding CLS and SEP tokens)
            to assign `exp.seqbert.pretrain.KEEP_INDEX` (unchanged but included in loss)
    """
    def __init__(self, seq_len,
                min_len, max_len,
                offset_min, offset_max,
                mask_prop, random_prop, keep_prop):
        self.seq_len = seq_len
        self.min_len = min_len
        self.max_len = max_len
        self.offset_min = offset_min
        self.offset_max = offset_max
        self.mask_props = (mask_prop, random_prop, keep_prop)

    """Split batch into two equal sets, shuffle the second half of
    one set of sequences for the NSP (next sequence prediction) pretraining task.
    Sequences are split exactly in the middle.
        batch: index tensor of dimensions (batch, sequence)
        returns: `(classification target, shuffled sequnce tensors)`
            Targets are true/false (whether each sequence is contiguous or shuffled)
            translated to tokens (as defined in `exp.seqbert.TOKENS_BP_IDX`.
            Shuffled sequence tensors are same dimensions as `batch`, except
            a SEP token `TOKENS_BP_IDX['/']`has been added between the first and
            last halves of all sequences so that dimensions are (batch, sequence + 1).
    """
    def split_shuffle(self, batch):
        with torch.no_grad():
            # split into first and last subsequences, permute half of 'last' subsequences
            first, last = torch.split(batch, (batch.size(1) // 2,
                    batch.size(1) - (batch.size(1) // 2)), dim=1)
            is_permuted, last = permute(last, prop=0.5)
            # labels: True (not permuted) or False (permuted) token
            cls_target = bool_to_tokens(torch.logical_not(is_permuted))
            # insert sep token between 'first' and 'last'
            sep = torch.ones([batch.size(0), 1], dtype=batch.dtype) * TOKENS_BP_IDX['/']
            return cls_target, torch.cat([first, sep, last], dim=1)

    """Randomly crops and shifts source sequence (which is longer than needed)
    to match target length. NONE tokens are used, i.e. `TOKENS_BP_IDX['n']`,
    as padding to fill subsequence to target length.
        split_seqs: output sequences from split_shuffle, where the SEP token is 
            exactly at the midpoint of every sequence.
        returns: `(cropped sequences, indexes of SEP token)`
            Cropped sequences are shifted so that their midpoint is between
            `offset_min` and `offset_max`, and cropped to dimensions (batch, `seq_len`).
            Indexes give the position of the SEP token for each sequence in the batch,
            and are of dimension (batch).
    """
    def rand_subseq(self, split_seqs):  # index relative to source midpoint
        batch_size = split_seqs.size(0)
        src_midpoint = split_seqs.size(1) // 2  # assume SEP token at midpoint
        # crop positions of first and last half
        starts = -1 * torch.randint(self.min_len, self.max_len, [batch_size])
        ends = torch.randint(self.min_len, self.max_len, [batch_size]) + 1 # add 1 for SEP token
        # position of SEP token in tgt
        sep_offsets = torch.randint(self.offset_min, self.offset_max, [batch_size])

        # fill target with empty/undefined base 'N'
        target = TOKENS_BP_IDX['n'] * torch.ones([batch_size, self.seq_len],
                                                        dtype=split_seqs.dtype)
        for i, (seq, start, end, offset) in enumerate(zip(split_seqs, starts, ends, sep_offsets)):
            tgt_start = max(1, start + offset)  # position 0 reserved for CLS token
            tgt_end = min(self.seq_len, end + offset)
            src_start = tgt_start - offset + src_midpoint
            src_end = tgt_end - offset + src_midpoint
            target[i, tgt_start:tgt_end] = seq[src_start:src_end]
        return target, sep_offsets

    """Randomly generates masked positions according to `mask_prop`, `random_prop`, `keep_prop`.
    First position of mask is always NO_LOSS_INDEX as the CLS token is assumed to be at this position.
    Applies mask to target sequence to generate source (training) sequence,
    replacing `MASK_INDEX` positions with `TOKENS_BP_IDX['m']` and `RANDOM_INDEX`
    with random different bases (i.e. `A` cannot map to `A`).
    Also replace classification label with CLS token at first position
        target: the sequence to calculate loss against
        returns: (source or training sequence, mask). Source sequence is the training input
            of same dimensions as target.
            Mask is an index tensor of same dimensions recording which positions were modified,
            in order to allow loss to be calculated only on the relevant positions
            (those that are not `NO_LOSS_INDEX`).
    """
    def mask_transform(self, target):
        with torch.no_grad():
            # randomize, mask, or mark for loss calculation some proportion of positions
            mask = generate_mask(target, self.mask_props)
            # omit classification token, separator, and any empty 'N'
            mask[:, 0] = Pretrain.NO_LOSS_INDEX
            omit = torch.logical_or(target == TOKENS_BP_IDX['/'],
                                    target == TOKENS_BP_IDX['n'])
            mask = mask.masked_fill(omit, Pretrain.NO_LOSS_INDEX)
            source = mask_randomize(target, mask == Pretrain.RANDOM_INDEX, 4)  # 4 base pairs
            source = mask_fill(source, mask == Pretrain.MASK_INDEX, TOKENS_BP_IDX['m'])
            # replace the classification target with CLS token '~'
            source[:, 0] = TOKENS_BP_IDX['~']
            # return mask of all positions that will contribute to loss
            return source, mask

    """Combines the above functions to generate batches for BERT pretraining.
        samples: list of objects or tuples from iterating over a `torch.utils.data.Dataset`,
            Length is batch size.
        returns: ((source, target, mask), (key, coord)). Tuple of training inputs and metadata.
            Training inputs are source tensor for input to `model.forward()`, target tensor
            for loss function, and mask for selecting loss positions from
            output and target tensors.
            Metadata is sequence key (i.e. name), which is an array of str of length (batch),
            and start coord (i.e. index) which is an index tensor of dimension (batch).
    """
    def collate(self, samples):
        sequences, metadata = zip(*samples)  # make each datatype a separate list
        key, coord = zip(*metadata)
        # shuffle for next sequence prediction task
        cls_targets, split_seqs = self.split_shuffle(torch.stack(sequences, dim=0))
        target, _ = self.rand_subseq(split_seqs)
        target[:, 0] = cls_targets
        # mask for masked token prediction task
        source, mask = self.mask_transform(target)
        return (source, target, mask), (key, torch.tensor(coord))  # send this to GPU


class Pretrain(SeqBERTLightningModule):

    # for mask
    NO_LOSS_INDEX = 0
    MASK_INDEX = 1
    RANDOM_INDEX = 2
    KEEP_INDEX = 3

    """Module for BERT pretraining on individual bases.
    Based on pytorch-lightning (pl) structure.
        hparams: `ArgumentParser` object containing all hyperparameters
    """
    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.model = SeqBERT(**hparams)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.prev_loss = 10000.

        # get sequence of length seq_len_source_multiplier*seq_len from dataloader
        self.sample_freq = int(self.hparams.seq_len * self.hparams.seq_len_sample_freq)
        self.load_seq_len = int(self.hparams.seq_len_source_multiplier * self.hparams.seq_len)
        min_crop = self.hparams.seq_len
        # if self.hparams.do_crop:
        #     min_crop = int(self.hparams.seq_len * self.hparams.crop_factor)
        max_crop = self.hparams.seq_len + 1
        offset_min = self.hparams.seq_len // 2
        offset_max = self.hparams.seq_len // 2 + 1
        # if self.hparams.do_shift:
        #     offset_min = int(offset_min * self.hparams.shift_factor)
        #     offset_max = self.hparams.seq_len - offset_min + 1

        self.train_batch_processor = PretrainBatchProcessor(self.hparams.seq_len,
                min_crop, max_crop, offset_min, offset_max,
                self.hparams.mask_prop, self.hparams.random_prop, self.hparams.keep_prop)
        if self.hparams.val_mask_prop is None:
            self.hparams.val_mask_prop = self.hparams.mask_prop
        if self.hparams.val_random_prop is None:
            self.hparams.val_random_prop = self.hparams.random_prop
        if self.hparams.val_keep_prop is None:
            self.hparams.val_keep_prop = self.hparams.keep_prop
        self.val_batch_processor = PretrainBatchProcessor(self.hparams.seq_len,
                min_crop, max_crop, offset_min, offset_max, self.hparams.val_mask_prop,
                self.hparams.val_random_prop, self.hparams.val_keep_prop)

        self.count_metric, self.acc_metric, self.pre_metric, self.rec_metric = {}, {}, {}, {}
        for name in ['cls', 'A', 'G', 'C', 'T']:
            self.count_metric[name] = Counter()
            self.acc_metric[name] = pl.metrics.classification.Accuracy()
            self.pre_metric[name] = pl.metrics.classification.Precision()
            self.rec_metric[name] = pl.metrics.classification.Recall()
        for name in ['mask', 'rand', 'keep']:
            self.count_metric[name] = Counter()
            self.acc_metric[name] = pl.metrics.classification.Accuracy()

    """(pl method) Gets Adam optimizer for module parameters.
        return: `torch.optim.Adam` optimizer object.
    """
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    """(pl method) Data loader for training data.
        return: `torch.utils.data.DataLoader`. Draws from RandomRepeatSequence
            if `DEBUG_use_random_data` is set, otherwise draws from `StridedSequence`
            using the sequence file at `hparams.seq_file` and intervals at `hparams.train_intervals`.
            Initial sequence length is `hparams.seq_len_source_multiplier * hparams.seq_len`,
            and then cropped to `hparams.seq_len` by the `PretrainBatchProcessor`.
            collate_fn is set to `PretrainBatchProcessor.collate()`.
    """
    def train_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            train_data = RandomRepeatSequence(self.load_seq_len, n_batch=10000,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(train_data, batch_size=self.hparams.batch_size,
                                shuffle=True, num_workers=self.hparams.num_workers,
                                collate_fn=self.train_batch_processor.collate)
        else:
            intervals = None
            if self.hparams.train_intervals is not None:
                intervals = bed_from_file(self.hparams.train_intervals)
            train_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.load_seq_len, include_intervals=intervals,
                        seq_transform=bioseq_to_index, sequential=False,
                        sample_freq=self.sample_freq)
        return train_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers,
                        collate_fn=self.train_batch_processor.collate)

    """(pl method) Data loader for validation data.
        return: `torch.utils.data.DataLoader`. Draws from RandomRepeatSequence
            if `DEBUG_use_random_data` is set, otherwise draws from `StridedSequence`
            using the sequence file at `hparams.seq_file` and intervals at `hparams.valid_intervals`.
            Initial sequence length is `hparams.seq_len_source_multiplier * hparams.seq_len`,
            and then cropped to `hparams.seq_len` by the `PretrainBatchProcessor`.
            collate_fn is set to `PretrainBatchProcessor.collate()`.
    """
    def val_dataloader(self):
        if self.hparams.DEBUG_use_random_data:
            valid_data = RandomRepeatSequence(self.load_seq_len, n_batch=100,
                                n_repeats=self.hparams.DEBUG_random_n_repeats,
                                repeat_len=self.hparams.DEBUG_random_repeat_len)
            return torch.utils.data.DataLoader(valid_data, batch_size=self.hparams.batch_size,
                                shuffle=False, num_workers=self.hparams.num_workers,
                                collate_fn=self.val_batch_processor.collate)
        else:
            intervals = None
            if self.hparams.valid_intervals is not None:
                intervals = bed_from_file(self.hparams.valid_intervals)
            valid_data = StridedSequence(FastaFile(self.hparams.seq_file),
                        self.load_seq_len, include_intervals=intervals, 
                        seq_transform=bioseq_to_index, sequential=True,
                        sample_freq=self.sample_freq)
        return valid_data.get_data_loader(self.hparams.batch_size, self.hparams.num_workers,
                        collate_fn=self.val_batch_processor.collate)

    """Helper function for BERT forward pass. Gets model output from source sequence,
    then calculates loss on masked output and masked target sequences.
    Calculates separate cls_loss for CLS token (assumed to be index 0 in each sequence),
    and scales cls_loss by `hparams.cls_regularization`. All losses calculated by `nn.CrossEntropyLoss`.
        return: (loss, masked_predict_loss, cls_loss, predicted, latent, source, target, mask, embedded)
            - loss is sum of masked_predict_loss and scaled cls_loss.
            - masked_predict_loss is loss from masked positions.
            - cls_loss is classification loss (first position), not scaled.
            - predicted is model output tensor.
            - latent is model output before decoding layers (typically 2 linear layers).
            - source is input sequence.
            - target is reference sequence for loss.
            - mask is sequence mask generated by PretrainBatchProcessor.
            - embedded is sequence input after applying embedding layer (including positional encoding).
    """
    def masked_forward(self, batch):
        source, target, mask = batch
        empty_positions = (source == TOKENS_BP_IDX['n'])
        predicted, latent, embedded = self.model.forward(source, no_attention=empty_positions)
        masked_predict_loss = self.loss_fn(mask_select(predicted, mask != self.NO_LOSS_INDEX),
                                            mask_select(target, mask != self.NO_LOSS_INDEX))
        # apply classification loss separately
        cls_loss = self.loss_fn(predicted[:,:, 0], target[:, 0])
        loss = masked_predict_loss + self.hparams.cls_regularization * cls_loss
        return loss, masked_predict_loss, cls_loss, predicted, latent, source, target, mask, embedded

    """Helper function for handling metrics using pytorch-lightning.
    Generates accuracy, precision, recall, counts for
    classification, mask positions, random positions, keep (identity) positions, and individual bases.
        predicted: model output tensor.
        source: input sequence.
        target: reference sequence for loss.
        mask: sequence mask generated by PretrainBatchProcessor.
        compute=True: recompute running totals for metrics.
        is_val=False: set to True in validation steps.
    """
    def accuracy_report(self, predicted, source, target, mask, compute=True, is_val=False):
        with torch.no_grad():
            self.log_stats('cls', predicted, target, (source == TOKENS_BP_IDX['~']),
                            prog_bar=True, compute=compute,
                            is_val=is_val, pos_idx=TOKENS_BP_IDX['t'])
            # only report accuracy for mask, rand, keep positions (positive sample is not defined)
            self.log_stats('mask', predicted, target, (mask == self.MASK_INDEX),
                            prog_bar=True, compute=compute, is_val=is_val)
            self.log_stats('rand', predicted, target, (mask == self.RANDOM_INDEX),
                            prog_bar=True, compute=compute, is_val=is_val)
            self.log_stats('keep', predicted, target, (mask == self.KEEP_INDEX),
                            prog_bar=True, compute=compute, is_val=is_val)
            if predicted is None:  # dummy variables so that logical_and works
                target = torch.zeros([1])
                loss_pos = torch.zeros([1])
            else:
                loss_pos = (mask != self.NO_LOSS_INDEX)
            self.log_stats('A', predicted, target, loss_pos, compute=compute,
                            is_val=is_val, pos_idx=TOKENS_BP_IDX['A'])
            self.log_stats('G', predicted, target, loss_pos, compute=compute,
                            is_val=is_val, pos_idx=TOKENS_BP_IDX['G'])
            self.log_stats('C', predicted, target, loss_pos, compute=compute,
                            is_val=is_val, pos_idx=TOKENS_BP_IDX['C'])
            self.log_stats('T', predicted, target, loss_pos, compute=compute,
                            is_val=is_val, pos_idx=TOKENS_BP_IDX['T'])

    """Helper function for handling individual metrics using pytorch-lightning.
        name: name of metric for display in pl logger and console.
        predicted: model output tensor.
        source: input sequence.
        target: reference sequence for loss.
        mask: sequence mask generated by PretrainBatchProcessor.
        prog_bar=False: if True, include in pytorch-lightning progress bar logged to console.
        compute=True: if True, calls `compute()` function of `pl.metrics` classes to summarize
            metrics, and logs metrics to pl logger, optionally prints metrics if `is_val=True`.
            If False, only calls `update()` to update running totals of metrics.
        is_val=False: set to True in validation steps. Adds `val_` prefix to metric name,
            sets `prog_bar=False`, and prints metrics if `compute=True`.
        pos_idx=None: the index value of a positive sample (e.g. base 'A' is 0), if set
            use to calculate precision and recall metrics, otherwise do not calculate.
    """
    def log_stats(self, name, predicted, target, mask,
                prog_bar=False, compute=True, is_val=False, pos_idx=None):
        if predicted is not None:
            pred = mask_select(predicted, mask).detach().cpu()
            tgt = mask_select(target, mask).detach().cpu()
            count = tgt.size(0)
            if pred.nelement() > 0:
                if pos_idx is not None:  # only log precision/recall if positive samples are defined
                    pred = (torch.argmax(pred, dim=1) == pos_idx)
                    tgt = (tgt == pos_idx)
                    self.pre_metric[name].update(pred, tgt)
                    self.rec_metric[name].update(pred, tgt)
                    count = torch.sum(tgt)  # count only positives
                self.acc_metric[name].update(pred, tgt)
            self.count_metric[name].update(count)
        if compute:
            count = self.count_metric[name].compute()
            accuracy = self.acc_metric[name].compute()
            precision, recall = torch.tensor([0]), torch.tensor([0])  # dummy values
            if pos_idx is not None:
                precision = self.pre_metric[name].compute()
                recall = self.rec_metric[name].compute()
            if is_val:
                name = 'val_' + name
                prog_bar = False
                if pos_idx is not None:
                    print('    {:10.10s} n: {:5.0f}  acc: {:1.2f}  pre: {:1.2f}  rec: {:1.2f}'.format(
                            name, count.item(), accuracy.item(), precision.item(), recall.item()))
                else:
                    print('    {:10.10s} n: {:5.0f}  acc: {:1.2f}'.format(
                            name, count.item(), accuracy.item()))
            self.log(name + '_n', count, prog_bar=False)
            self.log(name + '_a', accuracy, prog_bar=prog_bar)
            if pos_idx is not None:
                self.log(name + '_p', precision, prog_bar=False)
                self.log(name + '_r', recall, prog_bar=False)

    """Prints a subsequence and model predictions directly to console.
    This is an older backup method for showing a snapshot of model activity.
    Can be used for debugging.
        loss: sum of masked_predict_loss and scaled cls_loss.
        predicted: model output tensor.
        latent: model output before decoding layers (typically 2 linear layers).
        source: input sequence.
        target: reference sequence for loss.
        mask: sequence mask generated by PretrainBatchProcessor.
        embedded: sequence input after applying embedding layer (including positional encoding).
        seqname: name of sequence from metadata
        coord: start coordinate of sequence from metadata
    """
    def print_progress(self, loss, predicted, latent, source, target, mask, embedded, seqname, coord):
        str_train_sample = summarize(
            mask + len(self.model.tokens),
            source,
            target,
            correct(predicted, target),
            predicted.permute(1, 0, 2),
            index_symbols=self.model.tokens + [' ', '_', '?', '='])  # extra symbols represent masking

        hist = prediction_histograms(predicted.detach().cpu(), target.detach().cpu(), n_bins=5)
        acc = normalize_histogram(hist)
        acc_numbers = accuracy_per_class(hist, threshold_prob=1. / len(self.model.tokens))
        str_acc = summarize(acc, col_labels=INDEX_TO_BASE, normalize_fn=None)
        cls_acc = accuracy(predicted[:,:, 0], target[:, 0])
        print(seqname[0], coord[0], cls_acc, acc_numbers)
        print(str_acc, str_train_sample, sep='\n')

        embedded_vector_lengths = torch.norm(embedded, dim=2)  # vector length along channel dim
        latent_vector_lengths = torch.norm(latent, dim=2)  # vector length along channel dim
        embed_reshape = embedded.reshape(-1, embedded.shape[2])
        latent_reshape = latent.reshape(-1, latent.shape[2])
        embedded_pairwise_dist = F.pairwise_distance(embed_reshape[:, :-1], embed_reshape[:, 1:])
        latent_pairwise_dist = F.pairwise_distance(latent_reshape[:, :-1], latent_reshape[:, 1:])
        print('embedding/latent len/pairwise dist', tensor_stats_str(
            embedded_vector_lengths, embedded_pairwise_dist, latent_vector_lengths, latent_pairwise_dist))

    """(pl method) Runs one training iteration.
        x: input batch from pl, assumed to be tuple from `PretrainBatchProcessor.collate()`.
        batch_idx: iteration number from pl
    """
    def training_step(self, x, batch_idx):
        batch, (seqname, coord) = x
        loss, pred_loss, cls_loss, predicted, latent, source, target, mask, embedded = self.masked_forward(batch)
        self.prev_loss = loss.item()
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord)
        self.log('m_loss', pred_loss.item(), prog_bar=True)
        self.log('c_loss', cls_loss.item(), prog_bar=True)
        self.accuracy_report(predicted, source, target, mask)
        self.log_chr_coord(seqname, coord)
        return loss

    """(pl method) Runs one validation iteration. The only difference from `training_step`
    is logged metrics are named with `val_` prefix, and `accuracy_report` output is suppressed
    until end of validation.
        batch: input batch from pl, assumed to be tuple from `PretrainBatchProcessor.collate()`.
        batch_idx: iteration number from pl
    """
    def validation_step(self, batch, batch_idx):
        x, (seqname, coord) = batch
        loss, pred_loss, cls_loss, predicted, latent, source, target, mask, embedded = self.masked_forward(x)
        if batch_idx % self.hparams.print_progress_freq == 0:
            self.print_progress(loss, predicted, latent, source, target, mask, embedded, seqname, coord)
        self.log('val_m_loss', pred_loss.item())
        self.log('val_c_loss', cls_loss.item())
        self.accuracy_report(predicted, source, target, mask, compute=False, is_val=True)
        self.log_chr_coord(seqname, coord, is_val=True)
        return loss

    """(pl method) Runs at validation end. Currently only used to log and print metrics to console.
        val_step_outputs: (not used) automatically generated by pl from
            outputs of `validation_step`.
    """
    def validation_epoch_end(self, val_step_outputs):
        print('\nValidation complete\n')
        self.accuracy_report(None, None, None, None, compute=True, is_val=True)

    """(pl method) Defines all hyperparameters specific to pretraining.
        List of all hyperparameters:
            --cls_regularization=1.: scale factor for cls_loss.
            --keep_prop=0.05: proportion of KEEP_INDEX positions in mask,
                these are identity positions for masked prediction loss.
            --mask_prop=0.08: proportion of MASK_INDEX positions in mask,
                these are replaced with MASK token for masked prediction task.
            --random_prop=0.02: proportion of RANDOM_INDEX positions in mask,
                these are replaced with random bases for masked prediction loss.
            --val_keep_prop=None: same as keep_prop for validation, if None use keep_prop
            --val_mask_prop=None: same as mask_prop for validation, if None use mask_prop
            --val_random_prop=None: same as random_prop for validation, if None use random_prop
            --seq_file: path to reference sequence FASTA file.
            --train_intervals=None: path to intervals for training, BED file. If None,
                train on all sequences in seq_file.
            --valid_intervals=None: path to intervals for training, BED file. If None,
                validate on all sequences in seq_file.
            --seq_len_source_multiplier=2: sampled sequence length before cropping is
                seq_len_source_multiplier * seq_len, to allow extra length for cropping.
            --crop_factor=0.2: the smallest sequence length output after cropping
                but before padding is crop_factor * seq_len.
            --seq_len_sample_freq=0.5, type=float)  # gives sample_freq in StridedSequence
            # debug task params
            --DEBUG_use_random_data=False, type=bool)
            --DEBUG_random_repeat_len=1, type=int)
            --DEBUG_random_n_repeats=500, type=int)
        parent_parser: `ArgumentParser` object to add hyperparameters to.
    """
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        super_parser = SeqBERTLightningModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser])
        """
        Define parameters that only apply to this model
        """
        # model params
        parser.add_argument('--cls_regularization', default=1., type=float)
        # training params
        parser.add_argument('--keep_prop', default=0.05, type=float)
        parser.add_argument('--mask_prop', default=0.08, type=float)
        parser.add_argument('--random_prop', default=0.02, type=float)
        parser.add_argument('--val_keep_prop', default=None, type=float)  # use training props if None
        parser.add_argument('--val_mask_prop', default=None, type=float)
        parser.add_argument('--val_random_prop', default=None, type=float)
        # data params
        parser.add_argument('--seq_file', default='data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa', type=str)
        parser.add_argument('--train_intervals', default=None, type=str)
        parser.add_argument('--valid_intervals', default=None, type=str)
        parser.add_argument('--seq_len_source_multiplier', default=2., type=float)  # how much length to add when loading
        parser.add_argument('--crop_factor', default=0.2, type=float)  # how much of source sequence to keep when cropping
        parser.add_argument('--seq_len_sample_freq', default=0.5, type=float)  # gives sample_freq in StridedSequence
        # debug task params
        parser.add_argument('--DEBUG_use_random_data', default=False, type=bool)
        parser.add_argument('--DEBUG_random_repeat_len', default=1, type=int)
        parser.add_argument('--DEBUG_random_n_repeats', default=500, type=int)
        return parser

"""Run BERT pretraining using common `main` function to set up pl.
"""
if __name__ == '__main__':
    main(Pretrain, Pretrain)
