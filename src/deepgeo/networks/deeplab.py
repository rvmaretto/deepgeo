import sys
from os import path
import tensorflow as tf
import numpy as np
from importlib import reload

sys.path.insert(0, path.join(path.dirname(__file__),"../"))
import networks.layers as layers
import networks.loss_functions as lossf
reload(layers)
reload(lossf)

def deeplab_description(features, labels, params, mode, config):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL

    hyper_params = params