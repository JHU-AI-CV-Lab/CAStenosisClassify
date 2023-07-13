# CAStenosisClassify
This is the source code of paper: <a href="https://www.frontiersin.org/articles/10.3389/fcvm.2023.944135/full" target="_blank"> Deep learning-based end-to-end automated stenosis classification and localization on catheter coronary angiography </a>
## Introduction
CAStenosisClassify is a deep learning-based end-to-end project for artery stenosis classification in coronary angiography (CAG).
The workflow was employed as follows: 1) Candidate frame selection from CAG videograms with CNN+ Long Short Term Memory (LSTM) network, 2) Stenosis classification with Inception-v3 using 2 or 3 categories (<25%, >25%, and/or total occlusion) with and without redundancy training, and 3) Stenosis localization with two methods of class activation map (CAM) and anchor-based feature pyramid network (FPN).

## Getting Started
### Environment Setup
See castenosis_classify.yaml.

## Usage
### Candidate Frame Selection
Run TrainCore320_Inception_TOTAL_012-r.py to train an InceptionV3 model for candidate frame classification.
Run TrainCore320_Inception+LSTM_Region.py to train an LSTM for candidate region selection.

### Stenosis Classification
Run TrainCore320_Inception_TOTAL_012-r.py to train an InceptionV3 model for candidate frame classification.
Run TrainCore320_Inception+LSTM_Region.py to train an LSTM for candidate region selection.


