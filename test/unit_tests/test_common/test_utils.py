from nose.tools import *
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
import deepgeo.common.utils as utils


class Test_check_dict_parameters():
    def setup(self):
        self.params = {'a': 1, 'b': 2, 'c': 3}

    def test_check_dict_parameters_no_exception(self):
        mandatory = ['a', 'b']
        default = {'d': 4}
        params = utils.check_dict_parameters(self.params, mandatory, default)

        assert_equal(params['a'], 1)
        assert_equal(params['b'], 2)
        assert_equal(params['c'], 3)
        assert_equal(params['d'], 4)

    def test_check_dict_parameters_no_default(self):
        mandatory = ['a', 'b']
        params = utils.check_dict_parameters(self.params, mandatory)

        assert_equal(params['a'], 1)
        assert_equal(params['b'], 2)
        assert_equal(params['c'], 3)

    def test_check_dict_parameters_no_mandatory(self):
        default = {'d': 4}
        params = utils.check_dict_parameters(self.params, default=default)

        assert_equal(params['a'], 1)
        assert_equal(params['b'], 2)
        assert_equal(params['c'], 3)
        assert_equal(params['d'], 4)

    def test_check_dict_parameters_exception(self):
        mandatory = ['a', 'e']
        default = {'d': 4}
        assert_raises(AttributeError, utils.check_dict_parameters, self.params, mandatory, default)
