
import os, sys, re
from shutil import copyfile
import pandas as pd
import shutil
import math
import glob


IN_JPG_PATH = './core320_sample_jpg/'
BASE_PATH = './RAO_CRA/'
OUT_PATH = './RAO_CRA/'
OUT_PRE_PATH = os.path.join(OUT_PATH, 'pre')
OUT_POST_PATH = os.path.join(OUT_PATH, 'post')
BASE_JPG_PATH = os.path.join(BASE_PATH, 'jpg')
case_list = []
patient_list = []
INTERVAL_FRAME = 5
MARGIN_PRE = 10
MARGIN_POST = 10
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(OUT_PRE_PATH):
    os.makedirs(OUT_PRE_PATH)
if not os.path.exists(OUT_POST_PATH):
    os.makedirs(OUT_POST_PATH)

#create maximal index and minimal index dict for cases
case_max_dict = {}
case_min_dict = {}
case_num_dict = {}
images = os.listdir(BASE_JPG_PATH)
for image_name in images:
    image_split = image_name.split('_')
    patient_name = image_split[0]
    case_name = image_split[1]
    image_idx = int(image_split[-1].split('-')[-1].split('.')[0])
    if not case_max_dict.get(case_name):
        case_max_dict[case_name] = image_idx
        case_min_dict[case_name] = image_idx
        case_num_dict[case_name] = 1
        patient_list.append(patient_name)
        case_list.append(case_name)
    else:
        if case_max_dict[case_name] < image_idx:
            case_max_dict[case_name] = image_idx
        if case_min_dict[case_name] > image_idx:
            case_min_dict[case_name] = image_idx
        case_num_dict[case_name] = case_num_dict[case_name] + 1

#create dummy images for training
for _idx_p in range(len(patient_list)):
    patient_name = patient_list[_idx_p]
    case_name = case_list[_idx_p]
    idx_max = case_max_dict[case_name] + MARGIN_POST
    idx_min = case_min_dict[case_name] - MARGIN_PRE
    num_max_frame = case_num_dict[case_name]
    num_frame = 0
    for _idx_f in range(1, idx_min, INTERVAL_FRAME):
        if num_frame >= num_max_frame/2:
            break
        jpg_path = os.path.join(IN_JPG_PATH, patient_name, case_name)
        jpg_index_name = "IMG-*-{:0>5d}.jpg".format(int(_idx_f))

        for jpg_file in glob.glob(os.path.join(jpg_path, jpg_index_name)):
            src_jpg = jpg_file
            jpg_file_name = os.path.basename(jpg_file)
            dst_jpg = os.path.join(OUT_PRE_PATH, patient_name + '_' + case_name + '_' + jpg_file_name)
            shutil.copy2(src_jpg, dst_jpg)
            num_frame = num_frame + 1

    num_frame = 0
    for _idx_f in range(256, idx_max, -INTERVAL_FRAME):
        if num_frame >= num_max_frame/2:
            break
        jpg_path = os.path.join(IN_JPG_PATH, patient_name, case_name)
        jpg_index_name = "IMG-*-{:0>5d}.jpg".format(int(_idx_f))

        for jpg_file in glob.glob(os.path.join(jpg_path, jpg_index_name)):
            src_jpg = jpg_file
            jpg_file_name = os.path.basename(jpg_file)
            dst_jpg = os.path.join(OUT_POST_PATH, patient_name + '_' + case_name + '_' + jpg_file_name)
            shutil.copy2(src_jpg, dst_jpg)
            num_frame = num_frame + 1