import nose.tools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
import deepgeo.common.filesystem as fs


def test_directoryCreationAndDeletion():
    pathDir = os.path.join(os.path.dirname(__file__), './aaa')
    fs.mkdir(pathDir)
    nose.tools.assert_true(os.path.exists(pathDir))
    fs.delete_dir(pathDir)
    nose.tools.assert_false(os.path.exists(pathDir))
