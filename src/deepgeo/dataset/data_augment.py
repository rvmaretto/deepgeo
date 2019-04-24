import math
import sys
import numpy as np
import tensorflow as tf
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), '../'))
import common.utils as utils


# Methods based on https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#f8ea
def rotate_images(images, angles, data_type=np.float32):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    if not isinstance(angles, list):
        angles = [angles]
        
    if not isinstance(images, list) and not type(images) is np.ndarray:
        images = [images]
    
    tf.reset_default_graph()

    if type(images is np.ndarray):
        tf_shape = images.shape
    else:
        tf_shape = (None,) + images[0].shape
    # print(images[0].shape)
    # print(tf_shape)

    rotated_imgs = []

    # TODO: Links bellow can help to distribute on the GPUs:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    # https://www.tensorflow.org/guide/using_gpu
    # http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
    for device in utils.get_available_gpus():
        with tf.device(device):
            img = tf.placeholder(data_type, shape=tf_shape)
            radian = tf.placeholder(tf.float32, shape=len(images))
            tf_img = tf.contrib.image.rotate(img, radian)

            # with tf.Session(config=config) as sess:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for angle in angles:
            radian_angle = angle * math.pi / 180
            radian_list = [radian_angle] * len(images)
            rot_img = sess.run(tf_img, feed_dict={img: images,
                                                  radian: radian_list})
            rotated_imgs.extend(rot_img)

        sess.close()
    return np.array(rotated_imgs, dtype=data_type)


def flip_images(images, data_type=np.float32):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    if not isinstance(images, list) and not type(images) is np.ndarray:
        images = [images]

    flipped_imgs = []
    tf.reset_default_graph()
    
    # with tf.device('/cpu:0'): #TODO: REmove this. Try to distribute on the GPUs according to the available mem
    tf_img = tf.placeholder(data_type, shape=images[0].shape)
    tf_img1 = tf.image.flip_left_right(tf_img)
    tf_img2 = tf.image.flip_up_down(tf_img)
    tf_img3 = tf.image.transpose_image(tf_img)

    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img in images:
            flipped = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={tf_img: img})
            flipped_imgs.extend(flipped)

        sess.close()

    flipped_imgs = np.array(flipped_imgs, dtype=data_type)
    return flipped_imgs