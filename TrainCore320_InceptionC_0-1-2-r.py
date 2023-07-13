import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
from matplotlib.pyplot import subplots, show
# io related
from skimage.io import imread
import os, sys
from glob import glob
from Inception_Models import inceptionV3_coronary_model, tf_image_loader, tf_augmentor, flow_from_dataframe
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, KFold
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
import numpy as np

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    k_fold_idx = sys.argv[2]
    TRAINING_PATH = sys.argv[3] #'/home/ccong3/data/Core320_train_0-2-3_with_r_20190701/R/'
    LOG_NAME = sys.argv[4]
    lr = 0.0001
    isReduceLR = False
    useWarm = False
    isWarm = False
    num_epoches = 200
    num_category = 4
    H5_PATH = './hdf5/'
    LOG_PATH = './model_log/'

    weight_best_path = os.path.join(H5_PATH, "coronary_weightsC." + LOG_NAME + ".0_1_2_r.best." + k_fold_idx + ".hdf5")
    weight_warm_path = os.path.join(H5_PATH, "coronary_weightsC." + LOG_NAME + ".0_1_2_r.warm." + k_fold_idx + ".hdf5")
    weight_final_path = os.path.join(H5_PATH, "full_coronary_modelC." + LOG_NAME + ".0_1_2_r." + k_fold_idx + ".hdf5")

    IMG_SIZE = (512, 512)  # slightly smaller than vgg16 normally expects
    batch_size = 8
    base_image_dir = TRAINING_PATH

    #Snapshot
    print("==============Evaluate Snapshot==============\n"
          "Dataset path: %s\n"
          "Fold index: %s\n"
          "Image size: %d %d\n"
          "lr= %f\n"
          "Reduce lr: %s\n"
          "Use warm: %s, warm path=%s\n"
          "Is warm: %s, save path=%s\n"
          "==============End==============\n"
          % (TRAINING_PATH, k_fold_idx, IMG_SIZE[0], IMG_SIZE[1], lr, isReduceLR, useWarm, weight_warm_path, isWarm,
             weight_final_path)
          )

    angio_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
    angio_df['PatientId'] = angio_df['image'].map(lambda x: x.split('_')[0])
    angio_df['VideoId'] = angio_df['image'].map(lambda x: x.split('_')[0] + '_' + x.split('_')[1])  # New
    angio_df['path'] = angio_df['image'].map(lambda x: os.path.join(base_image_dir, 'jpg',
                                                                      '{}'.format(x)))
    angio_df['exists'] = angio_df['path'].map(os.path.exists)
    print(angio_df['exists'].sum(), 'images found of', angio_df.shape[0], 'total')
    angio_df['level_cat'] = angio_df['level'].map(lambda x: to_categorical(x, num_category))

    angio_df.dropna(inplace=True)
    angio_df = angio_df[angio_df['exists']]
    print(angio_df.sample(20))
    angio_df[['level']].hist(figsize=(10, num_category))
    #show()

    a_df = angio_df[['PatientId', 'VideoId', 'level']].drop_duplicates()  # New
    a_df = a_df[(True ^ a_df['level'].isin([3]))]
    kf = GroupKFold(n_splits=4)
    _fold_idx = int(k_fold_idx)
    _idx = 0
    for train, valid in kf.split(a_df['VideoId'], a_df['level'], a_df['PatientId']):
        print("%s %s" % (train, valid))
        _idx = _idx + 1
        if _idx >= _fold_idx:
            break

    train_ids = a_df['PatientId'].values[train]
    valid_ids = a_df['PatientId'].values[valid]
    print('Validation patient number: ' + str(pd.Series(valid_ids).drop_duplicates().size) + '; Validation patient ID: ')
    print(pd.Series(valid_ids).drop_duplicates())
    print('Are validation and training sets overlap?')
    print(pd.Series(valid_ids).isin(train_ids))
    raw_train_df = angio_df[angio_df['PatientId'].isin(train_ids)]  # New
    valid_df = angio_df[angio_df['PatientId'].isin(valid_ids)]  # New
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    valid_df = valid_df[(True ^ valid_df['level'].isin([3]))]# Remove redundant frames in validation set
    print('Remove redundant frames in validation set: %d validation images left'%(valid_df.shape[0]))
    train_df = raw_train_df.groupby(['level']).apply(
        lambda x: x.sample(int(raw_train_df[(True ^ raw_train_df['level'].isin([3]))].shape[0] / 3),
                           replace=True)).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
    train_df[['level']].hist(figsize=(10, 4))
    #show()
    print(train_df.sample(20))

    core_idg = tf_augmentor(out_size=IMG_SIZE,
                            color_mode='rgb',
                            vertical_flip=True,
                            crop_probability=0.0,  # crop doesn't work yet
                            batch_size=batch_size)
    valid_idg = tf_augmentor(out_size=IMG_SIZE, color_mode='rgb',
                             crop_probability=0.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             random_brightness=False,
                             random_contrast=False,
                             random_saturation=False,
                             random_hue=False,
                             rotation_range=0,
                             batch_size=batch_size)

    train_gen = flow_from_dataframe(core_idg, train_df, path_col='path',
                                    y_col='level_cat', batch_size=batch_size)

    valid_gen = flow_from_dataframe(valid_idg, valid_df, path_col='path',
                                    y_col='level_cat', batch_size=batch_size)  # we can use much larger batches for evaluation

    # t_x, t_y = next(valid_gen)
    # fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
    # for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    #    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    #    c_ax.set_title('Severity {}'.format(np.argmax(c_y, -1)))
    #    c_ax.axis('off')

    # show()

    t_x, t_y = next(train_gen)
    # fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
    # for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    #    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    #    c_ax.set_title('Severity {}'.format(np.argmax(c_y, -1)))
    #    c_ax.axis('off')
    coronary_model = inceptionV3_coronary_model(t_x, t_y, learning_rate=float(lr))

    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

    checkpoint = ModelCheckpoint(weight_best_path, monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only = True)

    min_lr = min(float(lr)/10.0, 0.00001)
    if isReduceLR:
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.8, patience=20, verbose=1, mode='max', epsilon=0.0001, cooldown=5, min_lr=min_lr)

    early = EarlyStopping(monitor="val_categorical_accuracy",
                          mode="max",
                          patience=100) # probably needs to be more patient
    tensorboard = TensorBoard(log_dir=os.path.join(LOG_PATH, 'core320', 'InceptionC', LOG_NAME, '0-2-3-r_' + k_fold_idx), histogram_freq=0)
    if isReduceLR:
        callbacks_list = [checkpoint, early, reduceLROnPlat, tensorboard]
    else:
        callbacks_list = [checkpoint, early, tensorboard]

    if useWarm and os.path.exists(weight_warm_path):
        coronary_model.load_weights(weight_warm_path)
        print('Using warm, successfully load weight file: %s' % (weight_warm_path))
    elif useWarm and not os.path.exists(weight_warm_path):
        print('Using warm, but cannot find weight file: %s' % (weight_warm_path))

    coronary_model.fit_generator(train_gen,
                                 steps_per_epoch = train_df.shape[0]//batch_size,
                                 validation_data = valid_gen,
                                 validation_steps = valid_df.shape[0]//batch_size,
                                 epochs = num_epoches,
                                 callbacks = callbacks_list,
                                 workers = 0,  # tf-generators are not thread-safe
                                 use_multiprocessing=False,
                                 max_queue_size = 0
                                 )

    # load the best version of the model
    coronary_model.load_weights(weight_best_path)

    coronary_model.save(weight_final_path)