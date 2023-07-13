
import os, sys, cv2, re
from shutil import copyfile
import pydicom
import pandas as pd
import shutil
import math
import glob
import cv2
import numpy as np

JPG_PATH = './core320_sample_jpg/'
STENOSIS_IN_PATH = './core320_sample_candidate_20190419.csv'
OUT_PATH = './Core320_train_candidate_total/'
OUT_LABEL_PATH = './Core320_train_candidate_total/trainLabels.csv'
OUT_JPG_PATH = os.path.join(OUT_PATH, 'jpg')

CLASSIFY_ARTERY_NAME = ['LCA', 'RCA']
TYPICAL_L_ANGLE = ['LAO_CRA', 'LAO_CAU', 'RAO_CRA', 'RAO_CAU', 'OTHER']
TYPICAL_R_ANGLE = ['LAO_CRA', 'RAO_CRA', 'LAO', 'RAO', 'OTHER']
CASE_MATCH_STR = r'IMG-(\d{4})-(\d{5}).*'
SAMPLE_FRAME = 5
MARGIN_PRE = 10
MARGIN_POST = 10
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
if not os.path.exists(OUT_JPG_PATH):
    os.makedirs(OUT_JPG_PATH)
image_size = (512, 512)
stenosis_data = pd.read_csv(STENOSIS_IN_PATH)

stenosis_dict = {}
stenosis_desc_dict = {}
phase_dict = {}
case_list = stenosis_data.case
image_list = stenosis_data.image
coronary_list = stenosis_data.coronary_artery
stenosis_list = stenosis_data.stenosis_level
angle_view_list = stenosis_data.angle_view_name
candidate_list = stenosis_data.candidate_frame
candidate_2_list = stenosis_data.candidate_2
candidate_3_list = stenosis_data.candidate_3
candidate_4_list = stenosis_data.candidate_4
candidate_5_list = stenosis_data.candidate_5
candidate_6_list = stenosis_data.candidate_6
candidate_7_list = stenosis_data.candidate_7
candidate_8_list = stenosis_data.candidate_8
candidate_9_list = stenosis_data.candidate_9
candidate_10_list = stenosis_data.candidate_10
candidate_11_list = stenosis_data.candidate_11
candidate_12_list = stenosis_data.candidate_12
candidate_13_list = stenosis_data.candidate_13
candidate_14_list = stenosis_data.candidate_14
candidate_15_list = stenosis_data.candidate_15
candidate_16_list = stenosis_data.candidate_16
candidate_17_list = stenosis_data.candidate_17
candidate_18_list = stenosis_data.candidate_18
candidate_19_list = stenosis_data.candidate_19
candidate_20_list = stenosis_data.candidate_20

out_patient_list = []
out_image_list = []
out_series_list = []
out_label_list = []


for idx in range(stenosis_data.shape[0]):
    candidate_frame_array = []
    case_name = case_list[idx].lower()
    series_name = image_list[idx]
    coronary_name = coronary_list[idx]
    angle_view_name = angle_view_list[idx]
    stenosis_level = stenosis_list[idx]
    candidate_frame_array.append(candidate_list[idx])
    candidate_frame_array.append(candidate_2_list[idx])
    candidate_frame_array.append(candidate_3_list[idx])
    candidate_frame_array.append(candidate_4_list[idx])
    candidate_frame_array.append(candidate_5_list[idx])
    candidate_frame_array.append(candidate_6_list[idx])
    candidate_frame_array.append(candidate_7_list[idx])
    candidate_frame_array.append(candidate_8_list[idx])
    candidate_frame_array.append(candidate_9_list[idx])
    candidate_frame_array.append(candidate_10_list[idx])
    candidate_frame_array.append(candidate_11_list[idx])
    candidate_frame_array.append(candidate_12_list[idx])
    candidate_frame_array.append(candidate_13_list[idx])
    candidate_frame_array.append(candidate_14_list[idx])
    candidate_frame_array.append(candidate_15_list[idx])
    candidate_frame_array.append(candidate_16_list[idx])
    candidate_frame_array.append(candidate_17_list[idx])
    candidate_frame_array.append(candidate_18_list[idx])
    candidate_frame_array.append(candidate_19_list[idx])
    candidate_frame_array.append(candidate_20_list[idx])
    region_frame_min = int(min(candidate_frame_array))
    region_frame_max = int(max(candidate_frame_array))
    if True:#coronary_name == 'R':
        jpg_path = os.path.join(JPG_PATH, case_name, series_name)
        _file_name = os.listdir(jpg_path)[-1]
        match = re.search(CASE_MATCH_STR, _file_name)
        if match == None:
            print(_file_name)
            continue

        video_index = match.group(1)
        frame_number = int(match.group(2))
        pre_frame_range = int(region_frame_min - MARGIN_PRE)
        pre_interval = int(pre_frame_range/SAMPLE_FRAME)+1
        if pre_interval > 0:
            for _index_frame in range(0, pre_frame_range, pre_interval):
                # copy jpg image by frame index
                _org_frame = _index_frame
                jpg_index_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_org_frame + 1))
                jpg_file = os.path.join(jpg_path, jpg_index_name)
                src_jpg = jpg_file
                jpg_file_name = os.path.basename(jpg_file)
                dst_file_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_index_frame + 1))
                dst_jpg = os.path.join(OUT_JPG_PATH, case_name + '_' + series_name + '_' + dst_file_name)
                dst_jpg_2 = os.path.join(OUT_PATH, '0', case_name + '_' + series_name + '_' + dst_file_name)
                if os.path.exists(src_jpg):
                    out_patient_list.append(case_name)
                    out_image_list.append(case_name + '_' + series_name + '_' + dst_file_name)
                    out_series_list.append(series_name)
                    shutil.copy2(src_jpg, dst_jpg)
                    shutil.copy2(src_jpg, dst_jpg_2)
                    out_label_list.append(0)
                else:
                    # skip and continue
                    print('miss: ' + src_jpg)
                    continue

        candidate_frame_range = int(region_frame_max - region_frame_min + 1)
        candidate_interval = int(candidate_frame_range / SAMPLE_FRAME)+1
        if candidate_interval > 0:
            for _index_frame in range(region_frame_min, region_frame_max+1, candidate_interval):
                # copy jpg image by frame index
                _org_frame = _index_frame
                jpg_index_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_org_frame + 1))
                jpg_file = os.path.join(jpg_path, jpg_index_name)
                src_jpg = jpg_file
                jpg_file_name = os.path.basename(jpg_file)
                dst_file_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_index_frame + 1))
                dst_jpg = os.path.join(OUT_JPG_PATH, case_name + '_' + series_name + '_' + dst_file_name)
                dst_jpg_2 = os.path.join(OUT_PATH, '1', case_name + '_' + series_name + '_' + dst_file_name)
                if os.path.exists(src_jpg):
                    out_patient_list.append(case_name)
                    out_image_list.append(case_name + '_' + series_name + '_' + dst_file_name)
                    out_series_list.append(series_name)
                    shutil.copy2(src_jpg, dst_jpg)
                    shutil.copy2(src_jpg, dst_jpg_2)
                    out_label_list.append(1)
                else:
                    # skip and continue
                    print('miss: ' + src_jpg)
                    continue

        post_frame_range = int(frame_number - region_frame_max - MARGIN_POST)
        post_interval = int(post_frame_range/SAMPLE_FRAME)+1
        if post_interval > 0:
            for _index_frame in range(region_frame_max + MARGIN_POST, frame_number, post_interval):
                # copy jpg image by frame index
                _org_frame = _index_frame
                jpg_index_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_org_frame + 1))
                jpg_file = os.path.join(jpg_path, jpg_index_name)
                src_jpg = jpg_file
                jpg_file_name = os.path.basename(jpg_file)
                dst_file_name = "IMG-{}-{:0>5d}.jpg".format(video_index, int(_index_frame + 1))
                dst_jpg = os.path.join(OUT_JPG_PATH, case_name + '_' + series_name + '_' + dst_file_name)
                dst_jpg_2 = os.path.join(OUT_PATH, '0', case_name + '_' + series_name + '_' + dst_file_name)
                if os.path.exists(src_jpg):
                    out_patient_list.append(case_name)
                    out_image_list.append(case_name + '_' + series_name + '_' + dst_file_name)
                    out_series_list.append(series_name)
                    shutil.copy2(src_jpg, dst_jpg)
                    shutil.copy2(src_jpg, dst_jpg_2)
                    out_label_list.append(0)
                else:
                    # skip and continue
                    print('miss: ' + src_jpg)
                    continue

patient_column = pd.Series(out_patient_list, name='patient')
image_column = pd.Series(out_image_list, name='image')
label_column = pd.Series(out_label_list, name='label')
series_column = pd.Series(out_series_list, name='series')
label_s = pd.concat([patient_column, series_column, image_column, label_column], axis=1)
save = pd.DataFrame(label_s)
save.to_csv(OUT_LABEL_PATH, index=True, sep=',')
