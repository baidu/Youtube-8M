# Temporal Modeling Approaches for Large-scale Youtube-8M Video Understanding
______________________________________________________________________________

By Fu Li, Chuang Gan, Xiao Liu, Yunlong Bian, Xiang Long, Yandong Li, Zhichao Li, Jie Zhou, Shilei Wen (Baidu IDL & Tsinghua University)

## Table of Contents
0. Introduction
1. Usage
2. Results
3. Citation

## Introduction
This repository contains the data providers and model configurations of three temporal modeling approaches (fast-forward sequence model, two stream sequence model and temporal residual neural networks) described in the paper "Temporal Modeling Approaches for Large-scale Youtube-8M Video Understanding" (xxx). 
These model configurations are those used in the Google Cloud & YouTube-8M Video Understanding Challenge (https://www.kaggle.com/c/youtube8m/leaderboard).

## Usage
Dependencies of PaddlePaddle 0.9.0 (https://github.com/PaddlePaddle/Paddle) and Python 2.7.

### Model Training:
```
cfg=your_config_file
paddle_trainer \
    --config=$cfg \
    --save_dir=./models \
    --trainer_count=4 \
    --log_period=20 \
    --num_passes=100 \
    --use_gpu=true \
    --test_period=0 \
    --show_parameter_stats_period=100
```

### Model Testing:
```
cfg=your_config_file
paddle_trainer \
    --config=$cfg \
    --use_gpu=true \
    --gpu_id=0 \
    --trainer_count=1 \
    --job=test \
    --init_model_path=pass-00000 \
    --predict_output_dir=output \
    --log_period=20 
```

## Results

Model | GAP@20
---------- | ----------
Temporal CNN | 0.80889
Two-stream LSTM | 0.82172
Two-stream GRU | 0.82366
Fast-forward LSTM | 0.81885
Fast-forward GRU | 0.81970
Fast-forward LSTM (depth7) | 0.82750

## Citation

