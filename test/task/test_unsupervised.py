import sys
sys.path.append('./src')
import unittest
import numpy.testing as npt

from seqmodel.task.unsupervised import *
from seqmodel.seq.transform import one_hot, one_hot_to_index
from seqmodel.task.task import GenericTask, LambdaLoss
from seqmodel.seq.mapseq import RandomRepeatSequence, create_test_batch


class Test_Unsupervised(unittest.TestCase):

    def setUp(self):
        self.x = create_test_batch(39, 17)
        self.identity = lambda x: x

    def test_PredictMaskedToken(self):
        null_task = PredictMaskedToken(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), allow_nan_loss=True)
        generic_task = GenericTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()))
        npt.assert_array_equal(null_task(self.x)[0].detach(),
                        generic_task(self.x)[0].detach())
        npt.assert_array_equal(null_task(self.x)[1].detach(),
                        generic_task(self.x)[1].detach())
        # loss is nan because no positions are included in loss
        npt.assert_array_equal(null_task.loss(self.x)[0].item(), float('nan'))
        # at least one position is inserted to avoid nan loss
        task = PredictMaskedToken(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), allow_nan_loss=False)
        self.assertGreater(task.loss(self.x)[0].item(), 0.)

        mask_value = 3.
        task = PredictMaskedToken(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), mask_prop=1., mask_value=mask_value)
        npt.assert_array_equal(task(self.x)[0], torch.ones(one_hot(self.x).shape)*mask_value)
        npt.assert_array_equal(task.loss(self.x)[0].item(),
                    nn.CrossEntropyLoss()(torch.ones(one_hot(self.x).shape)*mask_value, self.x))

        task = PredictMaskedToken(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), random_prop=1.)
        self.assertGreater(task.loss(self.x)[0].item(), 0.)

        task = PredictMaskedToken(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), keep_prop=1.)
        npt.assert_array_equal(task(self.x)[0], null_task(self.x)[0])
        npt.assert_array_equal(task(self.x)[1], null_task(self.x)[1])
        npt.assert_array_equal(task.loss(self.x)[0].item(), generic_task.loss(self.x)[0].item())


if __name__ == '__main__':
    unittest.main()
