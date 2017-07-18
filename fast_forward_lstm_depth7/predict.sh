config=$1
model=$2
gpu_id=$3
test_file=$4
paddle_trainer  \
  --config=$config \
  --use_gpu=true \
  --gpu_id=$gpu_id \
  --trainer_count=1 \
  --job=test \
  --init_model_path=$model \
  --config_args=is_predict=1,test_file=$test_file \
  --predict_output_dir=output_""$test_file \
  --log_period=10 \
