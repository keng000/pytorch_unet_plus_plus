import unittest
from datetime import datetime

import torch
from unet_plus_plus import NestNet


class TestNestNet(unittest.TestCase):
    def test_model_pruning_nest_net(self):
        """
        check if the model pruning get later then the elapse time get greater.
        """
        inputs = torch.rand((3, 1, 128, 128)).cuda()
        unet_plus_plus = NestNet(in_channels=1, n_classes=3).cuda()

        prev_time = None
        for L in range(1, 5):
            start = datetime.now()
            output = unet_plus_plus(inputs, L=L)

            elapse = (datetime.now() - start).total_seconds()
            if prev_time is not None:
                self.assertGreater(elapse, prev_time)

            prev_time = (datetime.now() - start).total_seconds()

        # raise value error if out of the range.
        self.assertRaises(ValueError, unet_plus_plus, inputs, L=5)


if __name__ == '__main__':
    unittest.main()