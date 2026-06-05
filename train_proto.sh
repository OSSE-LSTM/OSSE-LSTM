#!/bin/bash
# bash scripts/train_proto.sh 0 2 os_cnn ./exp/train.py 2

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Input illegal number of parameters " $#
  echo "Need 4-5 parameters for the GPUs, n_samples, arch, pythonfile, [full]"
  exit 1 
fi
gpus=$1
lr=0.001
weight_decay=1e-5
num_support_tr=$5
num_query_tr=$2
num_support_val=$5
num_query_val=$2
arch=$3
pythonfile=$4
full_mode=${6:-""}  

extra_args=""
if [ "$full_mode" == "full" ]; then
    extra_args="--full_mode"
fi

CUDA_VISIBLE_DEVICES=${gpus} python ${pythonfile} \
    --dataset_root C:/Users/ADMIN/Desktop/OSSE-LSTM/Splitted_datasets \
    --log_dir ./logs/${arch}/ \
    --log_interval 20 \
    --test_interval 5 \
    --epochs 1000 \
    --iterations 100 \
    --lr ${lr} \
    --lr_step 25 \
    --lr_gamma 0.7 \
    --weight_decay ${weight_decay} \
    --num_support_tr ${num_support_tr} \
    --num_query_tr ${num_query_tr} \
    --num_support_val ${num_support_val} \
    --num_query_val ${num_query_val} \
    --arch ${arch} \
    ${extra_args}