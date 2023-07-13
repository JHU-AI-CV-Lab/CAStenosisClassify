import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
from matplotlib.pyplot import subplots, show
# io related
from skimage.io import imread
import os
from glob import glob
import tensorflow as tf
from keras import backend as K
import keras
from keras.applications.inception_v3 import preprocess_input
import numpy as np

def tf_image_loader(out_size,
                    horizontal_flip=True,
                    vertical_flip=False,
                    random_brightness=True,
                    random_contrast=True,
                    random_saturation=True,
                    random_hue=True,
                    color_mode='rgb',
                    preproc_func=preprocess_input,
                    on_batch=False):
    def _func(X):
        with tf.name_scope('image_augmentation'):
            with tf.name_scope('input'):
                X = tf.image.decode_png(tf.read_file(X), channels=3 if color_mode == 'rgb' else 0)
                X = tf.image.resize_images(X, out_size)
            with tf.name_scope('augmentation'):
                if horizontal_flip:
                    X = tf.image.random_flip_left_right(X)
                if vertical_flip:
                    X = tf.image.random_flip_up_down(X)
                if random_brightness:
                    X = tf.image.random_brightness(X, max_delta=0.1)
                if random_saturation:
                    X = tf.image.random_saturation(X, lower=0.75, upper=1.5)
                if random_hue:
                    X = tf.image.random_hue(X, max_delta=0.15)
                if random_contrast:
                    X = tf.image.random_contrast(X, lower=0.75, upper=1.5)
                return preproc_func(X)

    if on_batch:
        # we are meant to use it on a batch
        def _batch_func(X, y):
            return tf.map_fn(_func, X), y

        return _batch_func
    else:
        # we apply it to everything
        def _all_func(X, y):
            return _func(X), y

        return _all_func


def tf_augmentor(out_size,
                 intermediate_size=(640, 640),
                 intermediate_trans='scale',
                 batch_size=16,
                 horizontal_flip=True,
                 vertical_flip=False,
                 random_brightness=True,
                 random_contrast=True,
                 random_saturation=True,
                 random_hue=True,
                 color_mode='rgb',
                 preproc_func=preprocess_input,
                 min_crop_percent=0.001,
                 max_crop_percent=0.005,
                 crop_probability=0.5,
                 rotation_range=10):
    load_ops = tf_image_loader(out_size=intermediate_size,
                               horizontal_flip=horizontal_flip,
                               vertical_flip=vertical_flip,
                               random_brightness=random_brightness,
                               random_contrast=random_contrast,
                               random_saturation=random_saturation,
                               random_hue=random_hue,
                               color_mode=color_mode,
                               preproc_func=preproc_func,
                               on_batch=False)

    def batch_ops(X, y):
        batch_size = tf.shape(X)[0]
        with tf.name_scope('transformation'):
            # code borrowed from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
            # The list of affine transformations that our image will go under.
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if rotation_range > 0:
                angle_rad = rotation_range / 180 * np.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms += [tf.contrib.image.angles_to_projective_transforms(angles, intermediate_size[0],
                                                                                intermediate_size[1])]

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], min_crop_percent, max_crop_percent)
                left = tf.random_uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
                top = tf.random_uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
                crop_transform = tf.stack([
                    crop_pct,
                    tf.zeros([batch_size]), top,
                    tf.zeros([batch_size]), crop_pct, left,
                    tf.zeros([batch_size]),
                    tf.zeros([batch_size])
                ], 1)
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
            if len(transforms) > 0:
                X = tf.contrib.image.transform(X,
                                               tf.contrib.image.compose_transforms(*transforms),
                                               interpolation='BILINEAR')  # or 'NEAREST'
            if intermediate_trans == 'scale':
                X = tf.image.resize_images(X, out_size)
            elif intermediate_trans == 'crop':
                X = tf.image.resize_image_with_crop_or_pad(X, out_size[0], out_size[1])
            else:
                raise ValueError('Invalid Operation {}'.format(intermediate_trans))
            return X, y

    def _create_pipeline(in_ds):
        batch_ds = in_ds.map(load_ops, num_parallel_calls=None).batch(batch_size)#New
        return batch_ds.map(batch_ops)

    return _create_pipeline

def flow_from_dataframe(idg,
                        in_df,
                        path_col,
                        y_col,
                        batch_size,
                        shuffle=True,
                        color_mode='rgb'):
    files_ds = tf.data.Dataset.from_tensor_slices((in_df[path_col].values,
                                                   np.stack(in_df[y_col].values, 0)))
    in_len = in_df[path_col].values.shape[0]
    while True:
        if shuffle:
            files_ds = files_ds.shuffle(in_len)  # shuffle the whole dataset

        next_batch = idg(files_ds).repeat().make_one_shot_iterator().get_next()
        for i in range(max(in_len // batch_size, 1)):
            # NOTE: if we loop here it is 'thread-safe-ish' if we loop on the outside it is completely unsafe
            yield K.get_session().run(next_batch)


from keras.metrics import top_k_categorical_accuracy
def top_2_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

def inceptionV3_retina_model(input_x, input_y):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda
    from keras.models import Model
    in_lay = Input(input_x.shape[1:])
    base_pretrained_model = PTModel(input_shape=input_x.shape[1:], include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(in_lay)
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(input_y.shape[-1], activation='softmax')(dr_steps)
    retina_model = Model(inputs=[in_lay], outputs=[out_layer])

    from keras import optimizers
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0., amsgrad=False)  # learning rate 1/10 lower

    retina_model.compile(optimizer=adam, loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
    retina_model.summary()
    return retina_model


def inceptionV3_coronary_model(input_x, input_y, learning_rate=0.0001):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda

    base_pretrained_model = PTModel(input_shape=input_x.shape[1:], include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(input_y.shape[-1], activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)

    from keras import optimizers
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0., amsgrad=False)  # learning rate 1/10 lower

    coronary_model.compile(optimizer=adam, loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
    #coronary_model.summary()
    return coronary_model




def inceptionV3_coronary_model_notop(input_x, input_y, learning_rate=0.0001, weights=None):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda

    base_pretrained_model = PTModel(input_shape=input_x.shape[1:], include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(input_y.shape[-1], activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)
    if weights != None:
        coronary_model.load_weights(weights)
    out_layer = coronary_model.layers[324].output
    pretrained_model = keras.models.Model(coronary_model.layers[0].input, outputs=out_layer)

    return pretrained_model


def inceptionV3_coronary_model_fc2(input_shape, output_classes, learning_rate=0.0001, weights=None):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda

    base_pretrained_model = PTModel(input_shape=input_shape, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(output_classes, activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)
    if weights != None:
        coronary_model.load_weights(weights)
    out_layer = coronary_model.layers[323].output
    pretrained_model = keras.models.Model(coronary_model.layers[0].input, outputs=out_layer)

    return pretrained_model


def inceptionV3_coronary_model_gap(input_shape, output_classes, learning_rate=0.0001, weights=None):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda

    base_pretrained_model = PTModel(input_shape=input_shape, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(output_classes, activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)
    if weights != None:
        coronary_model.load_weights(weights)
    out_layer = coronary_model.layers[321].output
    pretrained_model = keras.models.Model(coronary_model.layers[0].input, outputs=out_layer)

    return pretrained_model

def VGG_coronary_model(input_x, input_y):
    from keras.applications.vgg16 import VGG16 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda

    base_pretrained_model = PTModel(input_shape=input_x.shape[1:], include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(input_y.shape[-1], activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)

    from keras import optimizers
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0., amsgrad=False)  # learning rate 1/10 lower

    coronary_model.compile(optimizer=adam, loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
    coronary_model.summary()
    return coronary_model

def inceptionV3_model_org(input_size, class_number):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda
    from keras.models import Model
    in_lay = Input(input_size)
    base_pretrained_model = PTModel(input_shape=input_size, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model(in_lay)
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    x = GlobalAveragePooling2D()(bn_features)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    out_layer = Dense(class_number, activation='softmax')(x)
    coronary_model = Model(inputs=[in_lay], outputs=[out_layer])

    from keras import optimizers
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0., amsgrad=False)  # learning rate 1/10 lower

    coronary_model.compile(optimizer=adam, loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
    coronary_model.summary()
    return coronary_model


def inceptionV3_coronary_model_lca(input_x, input_y_shape, learning_rate=0.0001):
    from keras.applications.inception_v3 import InceptionV3 as PTModel
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, \
        LocallyConnected2D, Lambda
    #input_tensor = Input(shape=input_x.shape[1:])
    base_pretrained_model = PTModel(input_tensor=input_x, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = True  # allin: trainable=True
    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_features = base_pretrained_model.layers[-1].output
    from keras.layers import BatchNormalization
    bn_features = BatchNormalization()(pt_features)
    # here we do an attention mechanism to turn pixels in the GAP on an off
    attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = Conv2D(1,
                        kernel_size=(1, 1),
                        padding='valid',
                        activation='sigmoid')(attn_layer)
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                   activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0] / x[1])([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
    out_layer = Dense(input_y_shape, activation='softmax')(dr_steps)
    coronary_model = keras.models.Model(base_pretrained_model.layers[0].input, outputs=out_layer)

    from keras import optimizers
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=0., amsgrad=False)  # learning rate 1/10 lower

    coronary_model.compile(optimizer=adam, loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])
    #coronary_model.summary()
    return coronary_model


if __name__=="__main__":
    img_input = np.zeros((512, 512, 3))
    img_input = np.expand_dims(img_input, axis=0)
    pred_output = np.zeros((1, 1, 3))
    coronary_model = inceptionV3_coronary_model(img_input, pred_output)
    coronary_model.summary()

    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
