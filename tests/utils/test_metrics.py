import unittest

import numpy as np
import tensorflow.keras.backend as K

from ISR.utils.metrics import PSNR


class MetricsClassTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_PSNR_sanity(self):
        A = K.ones((10, 10, 3))
        B = K.zeros((10, 10, 3))
        self.assertEqual(K.get_value(PSNR(A, A)), np.inf)
        self.assertEqual(K.get_value(PSNR(A, B)), 0)
