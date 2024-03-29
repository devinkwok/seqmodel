import sys
sys.path.append('./src')
import unittest
import numpy as np
from pyfaidx import Fasta
import numpy.testing as npt

from seqmodel.seqdata.iterseq import *
from seqmodel.functional import bioseq_to_index


class Test_StridedSeq(unittest.TestCase):

    def setUp(self):
        self.seqfile = FastaFile('test/data/short.fa')
        self.ref_fasta = fasta_from_file('test/data/short.fa')
        self.seqs = [x[:len(x)] for x in self.ref_fasta.values()]

    def tearDown(self):
        self.seqfile.fasta.close()
        self.ref_fasta.close()

    def test_StridedSequence(self):
        dataset = StridedSequence(self.seqfile, 3, sequential=True)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 60 - 2 * 3)
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[2][-3:])

        dataset = StridedSequence(self.seqfile, 3, stride=3, start_offset=1)
        data = [x[0] for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][1:4])
        npt.assert_array_equal(data[1], self.seqs[0][4:7])
        npt.assert_array_equal(data[-1], self.seqs[2][-4:-1])

    def test_StridedSequence_intervals(self):
        intervals = {
            'seqname': ['seq1____'],
            'start': [0],
            'end': [30],
        }
        dataset = StridedSequence(self.seqfile, 3, sequential=True, include_intervals=intervals)
        data = [x[0] for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])

        intervals = {
            'seqname': ['seq2___', 'seq2___', 'seq1____', 'seq1____', 'seq1____'],
            'start': [5, 17, 2, 15, 20],
            'end': [9, 18, 5, 16, 30],
        }
        dataset = StridedSequence(self.seqfile, 3, sequential=True, include_intervals=intervals)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 11)
        npt.assert_array_equal(data[0], self.seqs[1][5:8])
        npt.assert_array_equal(data[1], self.seqs[1][6:9])
        npt.assert_array_equal(data[2], self.seqs[0][2:5])
        npt.assert_array_equal(data[3], self.seqs[0][20:23])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])

        intervals = bed_from_file('data/ref_genome/grch38_contig_intervals.bed')
        dataset = StridedSequence(FastaFile('data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa'),
                                100, sequential=True, include_intervals=intervals)
        for i, (_, _) in dataset:
            self.assertFalse(i == 'N')
            break

    def test_StridedSequence_freq(self):
        dataset = StridedSequence(self.seqfile, 10, sequential=True, sample_freq=10)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 6)
        npt.assert_array_equal(data[0], self.seqs[0][:10])
        npt.assert_array_equal(data[-2], self.seqs[1][-10:])

        dataset = StridedSequence(self.seqfile, 1, sequential=True, sample_freq=5)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 12)
        npt.assert_array_equal(data[0], self.seqs[0][:1])
        npt.assert_array_equal(data[1], self.seqs[0][5:6])

        dataset = StridedSequence(self.seqfile, 10, sequential=True, sample_freq=7)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 6)
        npt.assert_array_equal(data[0], self.seqs[0][:10])
        npt.assert_array_equal(data[1], self.seqs[0][7:17])
        npt.assert_array_equal(data[2], self.seqs[0][14:24])
        npt.assert_array_equal(data[3], self.seqs[1][:10])
        npt.assert_array_equal(data[4], self.seqs[1][7:17])

        dataset = StridedSequence(self.seqfile, 10, sequential=True, sample_freq=7, min_len=9)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 7)
        npt.assert_array_equal(data[3], self.seqs[0][21:30])

        dataset = StridedSequence(self.seqfile, 10, sequential=True, sample_freq=7, min_len=3)
        data = [x[0] for x in dataset]
        self.assertEqual(len(data), 9)
        npt.assert_array_equal(data[-1], self.seqs[2][7:10])


if __name__ == '__main__':
    unittest.main()
