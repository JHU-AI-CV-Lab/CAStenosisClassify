# CAStenosisClassify
This is the source code of paper: <a href="[https://join.slack.com/t/chatglm/shared_invite/zt-1y7pqoloy-9b1g6T6JjA8J0KxvUjbwJw](https://www.frontiersin.org/articles/10.3389/fcvm.2023.944135/full)" target="_blank"> Deep learning-based end-to-end automated stenosis classification and localization on catheter coronary angiography </a>
## Introduction
CAStenosisClassify is a deep learning-based end-to-end project for artery stenosis classification in coronary angiography (CAG).
The workflow was employed as follows: 1) Candidate frame selection from CAG videograms with CNN+ Long Short Term Memory (LSTM) network, 2) Stenosis classification with Inception-v3 using 2 or 3 categories (<25%, >25%, and/or total occlusion) with and without redundancy training, and 3) Stenosis localization with two methods of class activation map (CAM) and anchor-based feature pyramid network (FPN).

## Getting Started

### Hardware Requirements


### Environment Setup

### Usage
