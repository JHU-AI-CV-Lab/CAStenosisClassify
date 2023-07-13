
import os, sys, re
from shutil import copyfile
import pandas as pd
import shutil
import math
import glob


BASE_PATH = './Core320_train_candidate_total_cat/'
OUT_PATH = './Core320_train_candidate_total/'
OUT_LBL_PATH = os.path.join(OUT_PATH, 'trainLabels.csv')
OUT_JPG_PATH = os.path.join(OUT_PATH, 'jpg')
class_list = []
image_list = []
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(OUT_JPG_PATH):
    os.makedirs(OUT_JPG_PATH)
classes = os.listdir(BASE_PATH)
for class_name in classes:
    class_path = os.path.join(BASE_PATH, class_name)
    files = os.listdir(class_path)
    for file_name in files:
        if os.path.splitext(file_name)[1] != '.jpg':
            continue
        if class_name == '0':
            class_list.append(0)
        elif class_name == '1':
            class_list.append(1)

        image_list.append(file_name)
        src_jpg = os.path.join(class_path, file_name)
        dst_jpg = OUT_JPG_PATH
        shutil.copy2(src_jpg, dst_jpg)

class_column = pd.Series(class_list, name='level')
image_column = pd.Series(image_list, name='image')
label_s = pd.concat([image_column, class_column], axis=1)
save = pd.DataFrame(label_s)
save.to_csv(OUT_LBL_PATH, index=True, sep=',')