import os
import sys
# import tensorflow as tf
import tensorflow.keras as keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
# import networks.layers_new as layers


def unet(params):
    input_size = params['shape']#(params["patch_size"], params["patch_size"], params['bands'])
    inputs = keras.Input(input_size)

    if params['fusion'] == "early":
        fusion = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(inputs)
    else:
        fusion = inputs

    params['padding'] = 'same'

    ## ENCODER
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(fusion)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(pool1)
    conv2 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(pool2)
    conv3 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(pool3)
    conv4 = keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv4)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(pool4)
    conv5 = keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv5)
    drop5 = keras.layers.Dropout(0.5)(conv5)

    ## DECODER
    up6 = keras.layers.UpSampling2D(size=(2, 2))(drop5)
    if params['padding'] == 'valid':
        drop4 = keras.layers.Cropping2D(cropping=(int((drop4.shape[1] - up6.shape[1]) / 2), int((drop4.shape[1] - up6.shape[1]) / 2)))(drop4)
    merge6 = keras.layers.concatenate([drop4, up6], axis=-1)
    conv6 = keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(merge6)
    conv6 = keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv6)

    up7 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
    if params['padding'] == 'valid':
        conv3 = keras.layers.Cropping2D(cropping=(int((conv3.shape[1] - up7.shape[1]) / 2), int((conv3.shape[1] - up7.shape[1]) / 2)))(conv3)
    merge7 = keras.layers.concatenate([conv3, up7], axis=-1)
    conv7 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(merge7)
    conv7 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv7)

    up8 = keras.layers.UpSampling2D(size=(2, 2))(conv7)
    if params['padding'] == 'valid':
        conv2 = keras.layers.Cropping2D(cropping=(int((conv2.shape[1] - up8.shape[1]) / 2), int((conv2.shape[1] - up8.shape[1]) / 2)))(conv2)
    merge8 = keras.layers.concatenate([conv2, up8], axis=-1)
    conv8 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(merge8)
    conv8 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv8)

    up9 = keras.layers.UpSampling2D(size=(2, 2))(conv8)
    if params['padding'] == 'valid':
        conv1 = keras.layers.Cropping2D(cropping=(int((conv1.shape[1] - up9.shape[1]) / 2), int((conv1.shape[1] - up9.shape[1]) / 2)))(conv1)
    merge9 = keras.layers.concatenate([conv1, up9], axis=-1)
    conv9 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(merge9)
    conv9 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding=params['padding'], kernel_initializer='he_normal')(conv9)

    conv10 = keras.layers.Conv2D(params["num_classes"], 1, activation='softmax')(conv9)

    model = keras.Model(inputs=inputs, outputs=conv10)

    print(model.summary())
    print("-----------")

    if 'pretrained_weights' in params:
        model.load_weights(params['pretrained_weights'])

    return model
