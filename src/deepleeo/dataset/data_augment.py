import tensorflow as tf
import math
import numpy as np

# Methods based on https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9#f8ea
def rotate_images(images, angles, data_type=np.float32):
    if(not isinstance(angles, list)):
        angles = [angles]
        
    if(not isinstance(images, list)):
        images = [images]
    
    tf.reset_default_graph()
    
    tf_shape = (None,) + images[0].shape
    
    img = tf.placeholder(data_type, shape = tf_shape)
    radian = tf.placeholder(tf.float32, shape = (len(images)))
    tf_img = tf.contrib.image.rotate(img, radian)
    
    with tf.Session() as sess:
        rotated_imgs = []
        sess.run(tf.global_variables_initializer())
        for angle in angles:
            radian_angle = angle * math.pi / 180
            radian_list = [radian_angle] * len(images)
            rot_img = sess.run(tf_img, feed_dict = {img: images, radian: radian_list})
            rot_img = np.array(rot_img, dtype=data_type)
            rotated_imgs.extend(rot_img)
            
        sess.close()
        return rotated_imgs

def flip_images(images, data_type=np.float32):
    if(not isinstance(images, list)):
        angles = [images]
        
    flipped_imgs = []
    tf.reset_default_graph()
    
    tf_img = tf.placeholder(data_type, shape = images[0].shape)
    tf_img1 = tf.image.flip_left_right(tf_img)
    tf_img2 = tf.image.flip_up_down(tf_img)
    tf_img3 = tf.image.transpose_image(tf_img)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img in images:
            flipped = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {tf_img: img})
            flipped_imgs.extend(flipped)
            
        sess.close()
    
    flipped_imgs = np.array(flipped_imgs, dtype = data_type)
    return flipped_imgs