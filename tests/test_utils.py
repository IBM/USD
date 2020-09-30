import unittest
import numpy as np
import torch
from scipy.spatial import distance


class test_utils(unittest.TestCase):

    def test_import(self):
        import utils
        from utils import ddict

    def test_ddict(self):
        from utils import ddict

        # Test basics
        dd = ddict(a=1.0, b=[1,2])
        self.assertTrue(dd.a == 1.0)
        self.assertTrue(dd.b == [1,2])
        self.assertTrue(dd.a == dd['a'])
        self.assertTrue(dd.b == dd['b'])

        dd.c = 'c'
        self.assertTrue(dd.c == 'c')
        dd['c'] = 'efg'
        self.assertTrue(dd.c == 'efg')


if __name__ == '__main__':
    unittest.main()
