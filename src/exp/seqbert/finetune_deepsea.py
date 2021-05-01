import sys
sys.path.append('./src')
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything

from seqmodel.functional import one_hot_to_index
from seqmodel.functional.log import roc_auc
from selene.mat_file_sampler import MatFileSampler
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT, SeqBERTLightningModule, \
            CheckpointEveryNSteps, BinaryPredictTensorMetric, main
from exp.seqbert.pretrain import Pretrain


class MatFileDataset(IterableDataset):

    def __init__(self, sampler, batch_size, target_indexes=None):
        self.sampler = sampler
        self.batch_size = batch_size
        self.target_indexes = target_indexes

    def __iter__(self):
        for _ in range(self.sampler.n_samples):
            while True:
                batch, _ = self.sampler.get_data_and_targets(self.batch_size, n_samples=self.batch_size)
                seq, target = batch[0]
                target = torch.tensor(target, dtype=torch.float)
                if self.target_indexes is not None:
                    target = target[:,self.target_indexes]
                if torch.sum(target) != 0:  # no positives
                    break
            # swap dimensions from (batch, seq, channel) to the usual (batch, channel, seq)
            seq = torch.tensor(seq, dtype=torch.long).permute(0, 2, 1)
            seq = one_hot_to_index(seq)  # embedding works on indices
            cls_tokens = torch.zeros([seq.shape[0], 1], dtype=torch.long) + TOKENS_BP_IDX['~']
            seq = torch.cat([cls_tokens, seq], dim=1)
            yield seq, target  # (batch, seq, channel) and (batch, channel)


class FineTuneDeepSEA(SeqBERTLightningModule):

    def __init__(self, **hparams):
        super().__init__(**hparams)
        self.target = None
        n_class = 919
        if self.hparams.single_target is not None:
            self.target = [self.hparams.single_target]
            n_class = 1
        self.model = SeqBERT(classify_only=True, n_class=n_class, **hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = pl.metrics.Accuracy(threshold=0)
        self.train_roc_auc = pl.metrics.functional.classification.auroc
        self.auc_fn = pl.metrics.functional.classification.auc
        self.val_acc = pl.metrics.Accuracy(threshold=0, compute_on_step=False)
        self.val_pr_curve = pl.metrics.PrecisionRecallCurve(compute_on_step=False, pos_label=1)
        self.val_roc_curve = pl.metrics.classification.ROC(compute_on_step=False, pos_label=1)
        self.test_results = BinaryPredictTensorMetric()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_data = MatFileSampler(self.hparams.train_mat, 'trainxdata', 'traindata',
            sequence_batch_axis=2, sequence_alphabet_axis=1, targets_batch_axis=1)
        return MatFileDataset(train_data, self.hparams.batch_size, target_indexes=self.target)

    def val_dataloader(self):
        valid_data = MatFileSampler(self.hparams.valid_mat, 'validxdata', 'validdata',
            sequence_batch_axis=0, sequence_alphabet_axis=1, targets_batch_axis=0, shuffle=False)
        return MatFileDataset(valid_data, self.hparams.batch_size, target_indexes=self.target)

    def test_dataloader(self):
        test_data = MatFileSampler(self.hparams.test_mat, 'testxdata', 'testdata',
            sequence_batch_axis=0, sequence_alphabet_axis=1, targets_batch_axis=0, shuffle=False)
        return MatFileDataset(test_data, self.hparams.batch_size, target_indexes=self.target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted, target)
        self.log('tr_acc', self.train_acc(predicted.flatten(), target.flatten()), prog_bar=True)
        self.log('tr_roc', self.train_roc_auc(predicted.flatten(), target.flatten()), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, target = batch
        predicted, latent, embedded = self.model.forward(x)
        print(x.shape, predicted.shape, target.shape)
        loss = self.loss_fn(predicted, target)
        self.val_pr_curve(predicted.flatten(), target.flatten())
        self.val_roc_curve(predicted.flatten(), target.flatten())
        return loss

    def validation_epoch_end(self, val_step_outputs):
        self.log('val_acc', self.val_acc.compute())
        precision, recall, _ = self.val_pr_curve.compute()
        self.log('val_pr', self.auc_fn(recall, precision), prog_bar=True)
        fpr, tpr, _ = self.val_roc_curve.compute()
        self.log('val_roc', self.auc_fn(fpr, tpr), prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, target = batch
        predicted, latent, embedded = self.model.forward(x)
        loss = self.loss_fn(predicted, target)
        self.test_results(predicted, target)
        self.log('test_acc', self.train_acc(predicted.flatten(), target.flatten()), prog_bar=True)
        self.log('test_roc', self.train_roc_auc(predicted.flatten(), target.flatten()), prog_bar=True)
        return loss

    def test_epoch_end(self, val_step_outputs):
        scores = self.test_results.compute()
        print('Saving test scores to', self.hparams.test_out_file)
        torch.save(scores, self.hparams.test_out_file)

    @staticmethod
    def add_model_specific_args(parent_parser):
        super_parser = SeqBERTLightningModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[super_parser])
        """
        Define parameters that only apply to this model
        """
        #data params
        parser.add_argument('--train_mat', default='data/deepsea/train.mat', type=str)
        parser.add_argument('--valid_mat', default='data/deepsea/valid.mat', type=str)
        parser.add_argument('--test_mat', default='data/deepsea/test.mat', type=str)
        parser.add_argument('--single_target', default=None, type=int)
        parser.add_argument('--sum_representation', default=False, type=bool)
        return parser


if __name__ == '__main__':
    main(FineTuneDeepSEA, Pretrain)
