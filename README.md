# PaddlePaddle Networks for Youtube-8M Video Understanding Challenge

This repo describes the data providers and networks with PaddlePaddle, licensed under the Apache License 2.0.

## Dependencies

PaddlePaddle 0.9.0 (https://github.com/PaddlePaddle/Paddle)
Python 2.7

## Model Training

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

## Model Testing

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


