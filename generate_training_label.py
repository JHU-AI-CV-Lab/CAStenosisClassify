
import os, sys, re
from shutil import copyfile
import pandas as pd
import shutil
import math
import glob


BASE_PATH = './Core320_evaluate_20190701/'
OUT_PATH = './Core320_eval_0-12_20191022/'
CORONARY_TYPE = 'R'
IS_COMBINE_ANGLE_VIEW = True
IS_WITH_REDUNDANCY = False
base_path_full = os.path.join(BASE_PATH, CORONARY_TYPE)
out_path_full = os.path.join(OUT_PATH, CORONARY_TYPE)

class_list = []
image_list = []
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(out_path_full):
    os.makedirs(out_path_full)

angle_views = os.listdir(base_path_full)
for angle_view_name in angle_views:
    angle_view_path = os.path.join(base_path_full, angle_view_name)
    if IS_COMBINE_ANGLE_VIEW:
        out_label_path = os.path.join(out_path_full, 'trainLabels.csv')
        out_image_path = os.path.join(out_path_full, 'jpg')
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
    else:
        class_list = []
        image_list = []
        out_label_path = os.path.join(out_path_full, angle_view_name, 'trainLabels.csv')
        out_image_path = os.path.join(out_path_full, angle_view_name)
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
        out_image_path = os.path.join(out_path_full, angle_view_name, 'jpg')
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
    classes = os.listdir(angle_view_path)
    for class_name in classes:
        class_path = os.path.join(angle_view_path, class_name)
        files = os.listdir(class_path)
        for file_name in files:
            if os.path.splitext(file_name)[1] != '.jpg':
                continue
            if class_name == 'Normal':
                class_list.append(0)
            elif class_name == 'Mild':
                class_list.append(1)
            elif class_name == 'Stenosis':
                class_list.append(1)
            elif class_name == 'TotalOcclude':
                class_list.append(1)
            elif class_name == 'redundancy_late' and IS_WITH_REDUNDANCY:
                class_list.append(2)
            elif class_name == 'redundancy_early' and IS_WITH_REDUNDANCY:
                class_list.append(2)
            elif not IS_WITH_REDUNDANCY:
                continue
            else:
                continue

            image_list.append(file_name)
            src_jpg = os.path.join(class_path, file_name)
            dst_jpg = out_image_path
            shutil.copy2(src_jpg, dst_jpg)
    if not IS_COMBINE_ANGLE_VIEW:
        class_column = pd.Series(class_list, name='level')
        image_column = pd.Series(image_list, name='image')
        label_s = pd.concat([image_column, class_column], axis=1)
        save = pd.DataFrame(label_s)
        save.to_csv(out_label_path, index=True, sep=',')
if IS_COMBINE_ANGLE_VIEW:
    class_column = pd.Series(class_list, name='level')
    image_column = pd.Series(image_list, name='image')
    label_s = pd.concat([image_column, class_column], axis=1)
    save = pd.DataFrame(label_s)
    save.to_csv(out_label_path, index=True, sep=',')