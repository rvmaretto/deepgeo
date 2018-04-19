import unittest
from nose.tools import *
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"../../src"))


class TestSampleGenerator(unittest.TestCase):
    @raises(TypeError, ValueError)
    def test_c(self):
        self.assertTrue('c' == 'c')
        raise TypeError("This test passes")
