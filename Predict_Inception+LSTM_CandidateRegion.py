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
import glob, os, sys, re
import shutil
from keras.preprocessing import image
from keras.layers import Input, merge, Conv2D, Conv1D, BatchNormalization, Dropout, Embedding, Bidirectional
from keras.models import Model
import pandas as pd
from losses import f1_kaggle_loss, f1_kaggle
from Inception_Models import inceptionV3_coronary_model_fc2, inceptionV3_coronary_model_gap, inceptionV3_coronary_model_notop


def bring_data_from_directory(start_idx, end_idx, input_data):
    out_patient_list = []
    out_image_list = []
    out_series_list = []
    out_label_list = []
    out_frame_list = []
    case_list = input_data.case
    image_list = input_data.image
    series_num = end_idx - start_idx + 1
    input_img_array = np.zeros((series_num, time_phase, input_img_shape[0], input_img_shape[1], input_img_shape[2]))

    for idx in range(input_data.shape[0]):
        if idx < start_idx or idx > end_idx:
            continue
        case_name = case_list[idx].lower()
        series_name = image_list[idx]

        jpg_path = os.path.join(BASE_PATH, case_name, series_name)
        _file_name = os.listdir(jpg_path)[-1]
        match = re.search(CASE_MATCH_STR, _file_name)
        if match == None:
            print('Missing series: ' + series_name)
            continue

        video_index = match.group(1)
        frame_number = int(match.group(2))
        for _index_frame in range(0, time_phase):
            # copy jpg image by frame index
            _org_frame = int(float(_index_frame) / float(time_phase) * frame_number)
            jpg_index_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_org_frame + 1))
            jpg_file = os.path.join(jpg_path, jpg_index_name)
            src_jpg = jpg_file
            img = image.load_img(src_jpg, target_size=image_size)
            x = image.img_to_array(img)
            x = x/255.0
            input_img_array[idx-start_idx, _index_frame] = x
            if os.path.exists(src_jpg):
                out_patient_list.append(case_name)
                out_image_list.append(jpg_file)
                out_series_list.append(series_name)
                out_frame_list.append(frame_number)
                out_label_list.append(0)
            else:
                # skip and continue
                print('Missing file: ' + jpg_index_name)
                continue


    patient_column = pd.Series(out_patient_list, name='patient')
    image_column = pd.Series(out_image_list, name='image')
    label_column = pd.Series(out_label_list, name='label')
    series_column = pd.Series(out_series_list, name='series')
    frame_column = pd.Series(out_frame_list, name='frames')
    label_s = pd.concat([patient_column, series_column, image_column, label_column, frame_column], axis=1)
    input_df = pd.DataFrame(label_s)
    debug_df_path = os.path.join(DEBUG_DF_PATH, str(start_idx) + '-' + str(end_idx) + '.csv' )
    input_df.to_csv(debug_df_path, index=True, sep=',')

    return input_img_array, input_df

def extract_features(incept_model, img_array, label_df):
    series_num = img_array.shape[0]
    batch_num = img_array.shape[1]
    pretrain_data = []
    y_lable = None
    for idx in range(series_num):
        if len(pretrain_data) == 0:
            pretrain_data = incept_model.predict(x=img_array[idx], batch_size=batch_num)
            #y_lable = y
        else:
            pretrain_data = np.append(pretrain_data, incept_model.predict(x=img_array[idx], batch_size=batch_num), axis=0)
            #y_lable = np.append(y_lable, y, axis=0)

    feature_num = 0
    if pretrain_data.ndim == 2:
        feature_num = pretrain_data.shape[-1]
    elif pretrain_data.ndim == 3:
        feature_num = pretrain_data.shape[-1]*pretrain_data.shape[-2]
    elif pretrain_data.ndim == 4:
        feature_num = pretrain_data.shape[-1]*pretrain_data.shape[-2]*pretrain_data.shape[-3]

    all_data = np.zeros((series_num, time_phase, feature_num))
    all_labels = label_df['label'].as_matrix().reshape(series_num, time_phase, 1)
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
        #all_labels[_idx_series, _idx_frame] = y_lable[index]
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
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    candidate_number = 10
    series_batch_size = 16
    time_phase = 64
    image_size = (512, 512)
    BASE_PATH = './core320_sample_jpg'
    DEBUG_DF_PATH = './core320_sample_jpg'
    CANDIDATE_IN_PATH = './core320_sample_candidate.csv'
    CANDIDATE_OUT_PATH = './core320_sample_candidate_out.csv'
    NPY_PATH = './npy'
    H5_PATH = './hdf5'
    LOG_PATH = './model_log'

    WEIGHTS_INCEPT_PATH = os.path.join(H5_PATH, 'full_coronary_model.hdf5')
    WEIGHTS_LSTM_PATH = os.path.join(H5_PATH, 'coronary_Incept+LSTM.hdf5')
    CASE_MATCH_STR = r'IMG-(\d{4})-(\d{5}).*'
    if not os.path.exists(NPY_PATH):
        os.mkdir(NPY_PATH)
    if not os.path.exists(H5_PATH):
        os.mkdir(H5_PATH)
    out_case_list = []
    out_image_list = []
    out_region_start_list = []
    out_region_end_list = []
    input_img_shape = (image_size[0], image_size[1], 3)
    class_number = 2
    incept_model = inceptionV3_coronary_model_fc2(input_img_shape, class_number, weights=WEIGHTS_INCEPT_PATH)
    input_lstm_shape = (time_phase, incept_model.output.shape[-1].value)
    lstm_model = train_model_region_b(input_lstm_shape)
    lstm_model.load_weights(WEIGHTS_LSTM_PATH)
    input_data = pd.read_csv(CANDIDATE_IN_PATH)
    input_data_number = input_data.shape[0]
    print('Find %d cases, processing with batch size: %d' % (input_data_number, series_batch_size))
    total_batch = int(input_data_number/series_batch_size) + 1
    for idx_batch in range(total_batch):
        start_idx = idx_batch * series_batch_size
        end_idx = min((idx_batch + 1) * series_batch_size, input_data_number) - 1
        print('Processing %d ~ %d' % (start_idx, end_idx))

        input_img_array, label_df = bring_data_from_directory(start_idx, end_idx, input_data)
        all_data, all_labels = extract_features(incept_model, input_img_array, label_df)

        _total = all_data.shape[0]
        pred_region = lstm_model.predict(all_data, batch_size=series_batch_size, verbose=True)
        for idx_case in range(end_idx - start_idx + 1):
            case_name = label_df.patient[idx_case*time_phase]
            image_name = label_df.series[idx_case*time_phase]
            frame_number = int(label_df.frames[idx_case*time_phase])
            start_candidate_frame = -1
            end_candidate_frame = time_phase
            for idx_phase in range(time_phase):
                pred_region_frame = pred_region[idx_case][idx_phase][0]
                if pred_region_frame < 0.5: #Not in candidate region
                    if end_candidate_frame >= time_phase:
                        start_candidate_frame = idx_phase + 1
                if pred_region_frame >= 0.5: #In candidate region
                    if start_candidate_frame >= 0:
                        end_candidate_frame = idx_phase

            out_case_list.append(case_name)
            out_image_list.append(image_name)
            out_region_start_list.append(int(start_candidate_frame/time_phase*frame_number))
            out_region_end_list.append(int(end_candidate_frame / time_phase * frame_number))

    patient_column = pd.Series(out_case_list, name='case')
    image_column = pd.Series(out_image_list, name='image')
    region_start_column = pd.Series(out_region_start_list, name='candidate_start')
    region_end_column = pd.Series(out_region_end_list, name='candidate_end')
    label_s = pd.concat([patient_column, image_column, region_start_column, region_end_column], axis=1)
    save = pd.DataFrame(label_s)
    save.to_csv(CANDIDATE_OUT_PATH, index=True, sep=',')
