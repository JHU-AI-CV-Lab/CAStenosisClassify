from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, Activation, TimeDistributed, Concatenate, Masking
import numpy as np
import glob, os, sys
from Inception_Models import inceptionV3_coronary_model

from keras.models import Model
from keras.layers import Input, merge, Conv2D, Conv1D, BatchNormalization, Dropout, Embedding, Bidirectional
from keras.models import Model
import pandas as pd
from losses import f1_kaggle_loss, f1_kaggle
from Inception_Models import inceptionV3_coronary_model_fc2, inceptionV3_coronary_model_gap, inceptionV3_coronary_model_notop


def bring_data_from_directory():
    datagen = ImageDataGenerator(rescale=1. / 255)
    traindf = pd.read_csv(LABEL_PATH, dtype = str)
    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=TRAIN_PATH,
        x_col="image",
        y_col="label",
        subset="training",
        batch_size=time_phase,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=image_size)

    return train_generator, traindf

def extract_features(incept_model, data_generator, label_df):
    series_num = label_df['series'].nunique()
    pretrain_data = []
    y_lable = None
    batch = 0
    for x, y in data_generator:
        if batch > int(data_generator.n / data_generator.batch_size):
            break
        print("predict on inceptionV3_coronary_model_fc2. batch:{}, total:{}"
              .format(batch, int(data_generator.n / data_generator.batch_size)))
        batch += 1
        if len(pretrain_data) == 0:
            pretrain_data = incept_model.predict_on_batch(x)
            y_lable = y
            print(y)
        else:
            pretrain_data = np.append(pretrain_data, incept_model.predict_on_batch(x), axis=0)
            y_lable = np.append(y_lable, y, axis=0)

    feature_num = 0
    if pretrain_data.ndim == 2:
        feature_num = pretrain_data.shape[-1]
    elif pretrain_data.ndim == 3:
        feature_num = pretrain_data.shape[-1]*pretrain_data.shape[-2]
    elif pretrain_data.ndim == 4:
        feature_num = pretrain_data.shape[-1]*pretrain_data.shape[-2]*pretrain_data.shape[-3]

    all_data = np.zeros((series_num, time_phase, feature_num))
    all_labels = np.zeros((series_num, time_phase, 2))
    all_labels[..., 0] = np.ones((series_num, time_phase))
    series_frame_counts = label_df['series'].value_counts()
    prev_series_name = 'none' #用这个来判断一个新的series
    _idx_frame = 0
    _idx_series = -1 #start from -1
    for index, row in label_df.iterrows():
        series_name = row['series']

        if series_name != prev_series_name:
            if prev_series_name != 'none':
                print('In {} there are {} frames, actually {} frames'.
                      format(prev_series_name, series_frame_counts[prev_series_name], _idx_frame))
            _idx_frame = 0
            _idx_series = _idx_series+1
            prev_series_name = series_name
        all_data[_idx_series, _idx_frame] = np.reshape(pretrain_data[index], feature_num)
        all_labels[_idx_series, _idx_frame] = y_lable[index]
        _idx_frame = _idx_frame+1

    return all_data, all_labels

def train_model_region_b(input_shape):

    inp1 = Input(shape=input_shape)

    bi_lstm = Bidirectional(LSTM(256, return_sequences=True))(inp1)
    bt1 = TimeDistributed(BatchNormalization())(bi_lstm)
    conv1 = TimeDistributed(Dense(256, activation='relu'))(inp1)
    bt2 = TimeDistributed(BatchNormalization())(conv1)
    cc1 = Concatenate()([bt1, bt2])
    drop1 = Dropout(0.2)(cc1)
    conv2 = TimeDistributed(Dense(512, activation='relu'))(drop1)
    bt3 = TimeDistributed(BatchNormalization())(conv2)
    conv3 = TimeDistributed(Dense(256, activation='relu'))(bt3)
    bt4 = TimeDistributed(BatchNormalization())(conv3)
    drop2 = Dropout(0.2)(bt4)
    conv4 = TimeDistributed(Dense(1))(drop2)
    ac1 = Activation('sigmoid')(conv4)

    model = Model(inputs=inp1, outputs=ac1)
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    from keras import optimizers
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                           epsilon=None, decay=1e-6, amsgrad=False)

    model.compile(optimizer=adam, loss=f1_kaggle_loss, metrics=['accuracy', f1_kaggle])
    model.summary()

    return model

def evaluate_acc(test_Y, pred_Y):
    pred_Y_cat = np.round(np.reshape(pred_Y, pred_Y.shape[0]*pred_Y.shape[1]))
    test_Y_cat = np.reshape(test_Y, test_Y.shape[0]*test_Y.shape[1])
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    report = classification_report(test_Y_cat, pred_Y_cat)
    acc_score = accuracy_score(test_Y_cat, pred_Y_cat)
    f_score = f1_score(test_Y_cat, pred_Y_cat)

    print('Accuracy on Test Data: %2.4f' % (acc_score))
    print(report)
    return report, acc_score, f_score

if __name__ == '__main__':

    k_fold_idx = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

    series_batch_size = 16
    time_phase = 64
    image_size = (512, 512)
    BASE_PATH = './Core320_test_candidate_region_interpolate/'
    NPY_PATH = './npy'
    H5_PATH = './hdf5'
    LOG_PATH = './model_log'

    LABEL_PATH = os.path.join(BASE_PATH, 'trainLabels.csv')
    RESULT_PATH = os.path.join(BASE_PATH, 'resultLabels.csv')
    TRAIN_PATH = os.path.join(BASE_PATH, 'jpg')
    WEIGHTS_INCEPT_PATH = os.path.join(H5_PATH, 'full_coronary_model.hdf5')
    WEIGHTS_LSTM_PATH = os.path.join(H5_PATH, 'coronary_Incept+LSTM.hdf5')
    if not os.path.exists(NPY_PATH):
        os.mkdir(NPY_PATH)
    if not os.path.exists(H5_PATH):
        os.mkdir(H5_PATH)
    input_shape = (image_size[0], image_size[1], 3)
    output_classes = 2
    incept_model = inceptionV3_coronary_model_fc2(input_shape, output_classes, weights=WEIGHTS_INCEPT_PATH)
    data_generator, label_df = bring_data_from_directory()
    all_data, all_labels = extract_features(incept_model, data_generator, label_df)
    #for binary:
    all_labels = np.expand_dims(all_labels[...,1], axis=-1)
    input_lstm_shape = (time_phase, incept_model.output.shape[-1].value)
    #for cross validation
    n_folds = 4
    _total = all_data.shape[0]
    split = int(_total/n_folds)
    if k_fold_idx == '0':
        validation_data = all_data
        validation_labels = all_labels
    else:
        _fold_idx = int(k_fold_idx) - 1
        for idx_fold in range(n_folds):
            if idx_fold == _fold_idx:
                print("Running Fold", idx_fold + 1, "/", n_folds)
                validation_data = all_data[split * idx_fold:split * (idx_fold + 1)]
                validation_labels = all_labels[split * idx_fold:split * (idx_fold + 1)]

    model = None
    model = train_model_region_b(input_lstm_shape)
    model.load_weights(WEIGHTS_LSTM_PATH)
    valid_pred = model.predict(validation_data, batch_size=series_batch_size, verbose=True)
    report, acc_score, f_score = evaluate_acc(validation_labels, valid_pred)
    print('Accuracy on Test Data: %2.4f' % (acc_score))
    print('F1 score on Test Data: %2.4f' % (f_score))
    print(report)



