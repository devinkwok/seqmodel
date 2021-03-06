import sys
sys.path.append('./src')
import unittest
import torch
import torch.nn as nn
import pandas as pd
import numpy.testing as npt

from exp.seqbert.pretrain import *
from exp.seqbert.model import SeqBERT
from seqmodel.seqdata.iterseq import StridedSequence
from seqmodel.functional.transform import bioseq_to_index


class Test_Pretrain(unittest.TestCase):

    def setUp(self):
        self.seq_len = 5
        self.seqfile = FastaFile('test/data/short.fa')
        intervals = {
            'seqname': ['seq1____', 'seq1____', 'seq2___', 'seq3___'],
            'start': [0, 20, 10, 0],
            'end': [10, 30, 20, 10],
        }
        self.dataset = StridedSequence(self.seqfile, 2 * self.seq_len,
                        sequential=True, include_intervals=intervals,
                        transform=bioseq_to_index)
        self.data = [x[0] for x in self.dataset]
        self.batch = torch.stack(self.data, dim=0)

    def tearDown(self):
        self.seqfile.fasta.close()

    def test_BatchProcessor_split(self):
        batch_processor = PretrainBatchProcessor(self.seq_len, 1, self.seq_len,
                                                2, self.seq_len - 2, 0., 0., 0.)
        cls_target, split_seqs = batch_processor.split_shuffle(self.batch)
        self.assertEqual(self.batch.size(0), split_seqs.size(0))
        self.assertEqual(self.batch.size(1) + 1, split_seqs.size(1))
        npt.assert_array_equal(split_seqs[:, self.seq_len], SeqBERT.TOKENS_BP_IDX['/'])
        for label, seq, shuffled in zip(cls_target, self.batch, split_seqs):
            self.assertTrue(label == SeqBERT.TOKENS_BP_IDX['t'] or \
                            label == SeqBERT.TOKENS_BP_IDX['f'])
            npt.assert_array_equal(seq[:self.seq_len], shuffled[:self.seq_len])
            if label == SeqBERT.TOKENS_BP_IDX['t']:
                npt.assert_array_equal(seq[self.seq_len:], shuffled[self.seq_len+1:])
            else:
                self.assertFalse(torch.all(seq[self.seq_len:] == shuffled[self.seq_len+1:]))

    def test_BatchProcessor_subseq(self):
        batch_processor = PretrainBatchProcessor(self.seq_len, 1, self.seq_len,
                                                2, self.seq_len - 2, 0., 0., 0.)
        _, split_seqs = batch_processor.split_shuffle(self.batch)
        midpoint = split_seqs.size(1) // 2
        cropped_seqs, offsets = batch_processor.rand_subseq(split_seqs)
        self.assertEqual(cropped_seqs.size(0), self.batch.size(0))
        self.assertEqual(cropped_seqs.size(1), self.seq_len)
        for i, j in enumerate(offsets):
            self.assertEqual(cropped_seqs[i, j], SeqBERT.TOKENS_BP_IDX['/'])
            self.assertEqual(cropped_seqs[i, j-1], split_seqs[i, midpoint-1])
            self.assertEqual(cropped_seqs[i, j+1], split_seqs[i, midpoint+1])
        self.assertFalse(torch.any(cropped_seqs[:, (0, 1, -2, -1)] == SeqBERT.TOKENS_BP_IDX['/']))

        tgt_size = split_seqs.size(1) + 4
        src_halfsize = split_seqs.size(1) // 2
        batch_processor = PretrainBatchProcessor(tgt_size,
                                self.seq_len, self.seq_len + 1,
                                tgt_size - 2, tgt_size - 1, 0., 0., 0.)
        cropped_seqs, offsets = batch_processor.rand_subseq(split_seqs)
        npt.assert_array_equal(cropped_seqs[:, 0:-src_halfsize-2], SeqBERT.TOKENS_BP_IDX['n'])
        npt.assert_array_equal(cropped_seqs[:, -src_halfsize-2:-2], self.batch[:, 0:src_halfsize])
        npt.assert_array_equal(cropped_seqs[:, -2], SeqBERT.TOKENS_BP_IDX['/'])
        npt.assert_array_equal(cropped_seqs[:, -1], split_seqs[:, src_halfsize+1])

    def mask_compare(self, source, target, mask):
        bool_mask = (mask == Pretrain.MASK_INDEX)
        inv_mask = torch.logical_not(bool_mask)[:, 1:]
        npt.assert_array_equal(source.masked_select(bool_mask), SeqBERT.TOKENS_BP_IDX['m'])
        npt.assert_array_equal(source[:, 1:].masked_select(inv_mask),
                            target[:, 1:].masked_select(inv_mask))

    def test_BatchProcessor_mask(self):
        batch_processor = PretrainBatchProcessor(self.seq_len, 1, self.seq_len,
                                                2, self.seq_len - 2, 0.5, 0., 0.)
        _, split_seqs = batch_processor.split_shuffle(self.batch)
        target, offsets = batch_processor.rand_subseq(split_seqs)
        source, mask = batch_processor.mask_transform(target)
        npt.assert_array_equal(source[:, 0], SeqBERT.TOKENS_BP_IDX['~'])
        for i, j in enumerate(offsets):
            self.assertEqual(source[i, j], SeqBERT.TOKENS_BP_IDX['/'])
        self.mask_compare(source, target, mask)

    def test_BatchProcessor_collate(self):
        batch_processor = PretrainBatchProcessor(self.seq_len, 1, self.seq_len,
                                                2, self.seq_len - 2, 0.5, 0., 0.)

        batch, metadata = batch_processor.collate(zip(self.data, [('key', 1)] * len(self.data)))
        for tensor in batch:
            self.assertEqual(tensor.size(0), len(self.data))
            self.assertEqual(tensor.size(1), self.seq_len)
        source, target, mask = batch
        self.mask_compare(source, target, mask)
        npt.assert_array_equal(source[:, 0], SeqBERT.TOKENS_BP_IDX['~'])
        for i in range(len(self.data)):
            for j in range(self.seq_len):
                if target[i, j] == SeqBERT.TOKENS_BP_IDX['/']:
                    self.assertEqual(source[i, j], SeqBERT.TOKENS_BP_IDX['/'])
                elif target[i, j] == SeqBERT.TOKENS_BP_IDX['n']:
                    self.assertEqual(source[i, j], SeqBERT.TOKENS_BP_IDX['n'])

        dataloader = self.dataset.get_data_loader(4, 2, collate_fn=batch_processor.collate)


if __name__ == '__main__':
    unittest.main()
