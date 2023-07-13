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
from sklearn.model_selection import train_test_split, StratifiedKFold

def bring_data_from_directory(fold_idx):
    datagen = ImageDataGenerator(rescale=1. / 255)
    df = pd.read_csv(LABEL_PATH, dtype = str)
    rr_df = df[['patient', 'label']].drop_duplicates()  # New
    kf = StratifiedKFold(n_splits=4)
    train_valid_ids = kf.split(rr_df['patient'], rr_df['label'])
    _idx = 0
    if fold_idx == 0:
        valid_df = rr_df
    else:
        for train, valid in kf.split(rr_df['patient'], rr_df['label']):
            _idx = _idx + 1
            if _idx >= fold_idx:
                break

        train_ids = rr_df['patient'].as_matrix()[train]
        valid_ids = rr_df['patient'].as_matrix()[valid]
        print("Train patients: %s" % train_ids)
        print("Validate patients: %s" % valid_ids)
        train_df = df[df['patient'].isin(train_ids)]
        valid_df = df[df['patient'].isin(valid_ids)]

    train_generator = datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=TRAIN_PATH,
        x_col="image",
        y_col="label",
        batch_size=time_phase,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=image_size)

    return train_generator, valid_df

def train_model_region_b(train_data):

    inp1 = Input(shape=(train_data.shape[1], train_data.shape[2]))

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

    time_phase = 64
    image_size = (512, 512)

    BASE_PATH = './data/Core320_train_candidate_region_interpolate/'
    NPY_PATH = './npy/'
    H5_PATH = './hdf5/'
    LOG_PATH = './model_log/'
    LABEL_PATH = os.path.join(BASE_PATH, 'trainLabels.csv')
    RESULT_PATH = os.path.join(BASE_PATH, 'resultLabels.csv')
    TRAIN_PATH = os.path.join(BASE_PATH, 'jpg')
    WEIGHTS_INCEPT_PATH = os.path.join(H5_PATH, 'full_coronary_model.hdf5')
    if not os.path.exists(NPY_PATH):
        os.mkdir(NPY_PATH)
    if not os.path.exists(H5_PATH):
        os.mkdir(H5_PATH)
    _fold_idx = int(k_fold_idx)
    data_generator, label_df = bring_data_from_directory(_fold_idx)
    validation_labels = None
    batch = 0
    for x, y in data_generator:
        if batch >= int(data_generator.n / data_generator.batch_size):
            break
        print("batch:{}, total:{}"
              .format(batch, int(data_generator.n / data_generator.batch_size)))

        if batch == 0:
            validation_labels = y
        else:
            validation_labels = np.append(validation_labels, y, axis=0)
        batch += 1
    t_x, t_y = next(data_generator)
    model = None
    model = inceptionV3_coronary_model(t_x, t_y)
    model.load_weights(WEIGHTS_INCEPT_PATH)
    report = model.evaluate_generator(generator = data_generator,
                                      steps = label_df.shape[0]//time_phase,
                                      verbose = True)
    #report, acc_score, f_score = evaluate_acc(validation_labels, valid_pred)
    #print('Accuracy on Test Data: %2.4f' % (acc_score))
    #print('F1 score on Test Data: %2.4f' % (f_score))
    print(report)
    valid_pred = model.predict_generator(generator = data_generator,
                                      steps = label_df.shape[0]//time_phase,
                                      verbose = True)

    report, acc_score, f_score = evaluate_acc(validation_labels, valid_pred)
    print('Accuracy on Test Data: %2.4f' % (acc_score))
    print('F1 score on Test Data: %2.4f' % (f_score))

