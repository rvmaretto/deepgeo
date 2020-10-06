import tensorflow as tf
import math
import numpy as np
import tensorflow_addons as tfa


# Methods based on https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#f8ea
def rotate_images(images, angles, data_type=np.float32):
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True

    if not isinstance(angles, list):
        angles = [angles]
        
    if not isinstance(images, list) and not type(images) is np.ndarray:
        images = [images]
    
    tf.compat.v1.reset_default_graph()

    if type(images is np.ndarray):
        tf_shape = images.shape
    else:
        tf_shape = (None,) + images[0].shape
    # print(images[0].shape)
    # print(tf_shape)


    # TODO: Links bellow can help to distribute on the GPUs:
    # https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    # https://www.tensorflow.org/guide/using_gpu
    # http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
    with tf.device('/cpu:0'):  # TODO: Remove this. Try to distribute on the GPUs according to the available mem
        img = tf.compat.v1.placeholder(data_type, shape=tf_shape)
        radian = tf.compat.v1.placeholder(tf.float32, shape=len(images))
        tf_img = tfa.image.rotate(img, radian)

        # with tf.Session(config=config) as sess:
        with tf.compat.v1.Session() as sess:
            rotated_imgs = []
            sess.run(tf.compat.v1.global_variables_initializer())
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
    tf.compat.v1.reset_default_graph()
    
    # with tf.device('/cpu:0'): #TODO: REmove this. Try to distribute on the GPUs according to the available mem
    tf_img = tf.compat.v1.placeholder(data_type, shape=images[0].shape)
    tf_img1 = tf.image.flip_left_right(tf_img)
    tf_img2 = tf.image.flip_up_down(tf_img)
    tf_img3 = tf.image.transpose(tf_img)

    # with tf.Session(config=config) as sess:
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for img in images:
            flipped = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={tf_img: img})
            flipped_imgs.extend(flipped)

        sess.close()

    flipped_imgs = np.array(flipped_imgs, dtype=data_type)
    return flipped_imgs