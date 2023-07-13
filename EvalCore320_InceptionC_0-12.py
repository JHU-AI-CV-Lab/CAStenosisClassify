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
    EVALUATE_PATH = sys.argv[3] #'/home/ccong3/data/Core320_train_0-2-3_with_r_20190701/R/'
    weight_path = sys.argv[4]

    IMG_SIZE = (512, 512)  # slightly smaller than vgg16 normally expects
    base_image_dir = EVALUATE_PATH
    batch_size = 8
    category_number = 2
    #Snapshot
    print("==============Evaluate Snapshot==============\n"
          "Dataset path: %s\n"
          "Fold index: %s\n"
          "Image size: %d %d\n"
          "Use weight, path=%s\n"
          "==============End==============\n"
          % (EVALUATE_PATH, k_fold_idx, IMG_SIZE[0], IMG_SIZE[1], weight_path)
          )

    angio_df = pd.read_csv(os.path.join(base_image_dir, 'trainLabels.csv'))
    angio_df['PatientId'] = angio_df['image'].map(lambda x: x.split('_')[0])
    angio_df['VideoId'] = angio_df['image'].map(lambda x: x.split('_')[0] + '_' + x.split('_')[1])  # New
    angio_df['path'] = angio_df['image'].map(lambda x: os.path.join(base_image_dir, 'jpg',
                                                                      '{}'.format(x)))
    angio_df['exists'] = angio_df['path'].map(os.path.exists)
    print(angio_df['exists'].sum(), 'images found of', angio_df.shape[0], 'total')
    angio_df['level_cat'] = angio_df['level'].map(lambda x: to_categorical(x, 1 + category_number))

    angio_df.dropna(inplace=True)
    angio_df = angio_df[angio_df['exists']]
    print(angio_df.sample(20))
    angio_df[['level']].hist(figsize=(10, 2))
    #show()

    a_df = angio_df[['PatientId', 'VideoId', 'level']].drop_duplicates()  # New
    kf = GroupKFold(n_splits=4)
    # train_valid_ids = kf.split(a_df['VideoId'], a_df['level'], a_df['PatientId'])
    _fold_idx = int(k_fold_idx)
    _idx = 0
    if _fold_idx <= 0:
        train_ids = None
        valid_ids = a_df['PatientId']
        valid_df = angio_df[angio_df['PatientId'].isin(valid_ids)]  # New
    else:
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

    valid_gen = flow_from_dataframe(valid_idg, valid_df, path_col='path',
                                    y_col='level_cat', batch_size=batch_size, shuffle=False)  # we can use much larger batches for evaluation

    t_x, t_y = next(valid_gen)
    # fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
    # for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    #    c_ax.imshow(np.clip(c_x*127+127, 0, 255).astype(np.uint8))
    #    c_ax.set_title('Severity {}'.format(np.argmax(c_y, -1)))
    #    c_ax.axis('off')
    coronary_model = inceptionV3_coronary_model(t_x, t_y)

# load the best version of the model
    coronary_model.load_weights(weight_path)

    ##### create one fixed dataset for evaluating
    from tqdm import tqdm_notebook

    # fresh valid gen
    valid_gen = flow_from_dataframe(valid_idg, valid_df,
                                    path_col='path',
                                    y_col='level_cat', batch_size=batch_size, shuffle=False)
    vbatch_count = (valid_df.shape[0] // batch_size - 1)
    out_size = vbatch_count * batch_size
    test_X = np.zeros((out_size,) + t_x.shape[1:], dtype=np.float32)
    test_Y = np.zeros((out_size,) + t_y.shape[1:], dtype=np.float32)
    for i, (c_x, c_y) in zip(range(vbatch_count), valid_gen):
        j = i * batch_size
        test_X[j:(j + c_x.shape[0])] = c_x
        test_Y[j:(j + c_x.shape[0])] = c_y

    from sklearn.metrics import accuracy_score, classification_report, f1_score, cohen_kappa_score
    from metrics import quadratic_kappa

    pred_Y = coronary_model.predict(test_X, batch_size=8, verbose=True)

    pred_Y_cat = np.argmax(pred_Y, -1)
    test_Y_cat = np.argmax(test_Y, -1)
    report = classification_report(test_Y_cat, pred_Y_cat)
    acc_score = accuracy_score(test_Y_cat, pred_Y_cat)
    f_score = 0.0
    eval_result_dir = os.path.join(EVALUATE_PATH, 'evaluation_' + k_fold_idx)
    if not os.path.exists(eval_result_dir):
        os.mkdir(eval_result_dir)
    eval_acc_result_txt = os.path.join(eval_result_dir, 'eval_acc.txt')
    # q_kappa = quadratic_kappa(test_Y, np.round(pred_Y))
    c_kappa = cohen_kappa_score(test_Y_cat, pred_Y_cat)
    print('Accuracy on Test Data: %2.4f' % (acc_score))
    print(report)
    with open(eval_acc_result_txt, 'w') as f:
        print('Accuracy on Test Data: %2.4f' % (acc_score), file=f)
        print('F1 score on Test Data: %2.4f' % (f_score), file=f)
        print('Kappa on Test Data: %2.4f' % (c_kappa), file=f)
        print(report, file=f)  # Python 3.x

    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score

    np.savetxt(os.path.join(eval_result_dir, 'test_Y.csv'), test_Y, delimiter=",")
    np.savetxt(os.path.join(eval_result_dir, 'pred_Y.csv'), pred_Y, delimiter=",")
    sick_vec = test_Y_cat > 0
    sick_score = np.sum(pred_Y[:, 1:], 1)
    fpr, tpr, _ = roc_curve(sick_vec, sick_score)
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    np.savetxt(os.path.join(eval_result_dir, 'fpr.csv'), fpr, delimiter=",")
    np.savetxt(os.path.join(eval_result_dir, 'tpr.csv'), tpr, delimiter=",")

    ax1.plot(fpr, tpr, 'b.-', label='Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
    ax1.plot(fpr, fpr, 'g-', label='Random Guessing')
    ax1.legend()
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    fig.savefig(os.path.join(eval_result_dir, 'roc_curve.png'))


    #Display wrong cases
    wrong_idx = []
    for idx in range(pred_Y_cat.shape[0]):
        _pred = pred_Y_cat[idx]
        _true = test_Y_cat[idx]
        if _pred != _true:
            wrong_idx.append(idx)
    error_result_dir = os.path.join(eval_result_dir, 'error_pred')
    if not os.path.exists(error_result_dir):
        os.mkdir(error_result_dir)
    for idx in wrong_idx:
        fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
        ax1.imshow(np.clip(test_X[idx] * 127 + 127, 0, 255).astype(np.uint8), cmap='bone')
        ax1.set_title('Actual Severity: {}\n{}'.format(test_Y_cat[idx], '\n'.join(
            ['Predicted %02d (%04.1f%%): %s' % (k, 100 * v, '*' * int(10 * v))
             for k, v in sorted(enumerate(pred_Y[idx]), key=lambda x: -1 * x[1])])), loc='left')
        ax1.axis('off')
        png_name = '{}.png'.format(valid_df.image.base[0][idx%valid_df.image.shape[0]])
        plt.savefig(os.path.join(error_result_dir, png_name), dpi=150)
        plt.clf()