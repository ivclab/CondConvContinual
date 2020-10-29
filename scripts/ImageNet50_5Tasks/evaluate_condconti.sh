#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {BACKBONE} {FINAL_TASK_IDX}"
    exit 0;
fi


GPU_ID=$1
BACKBONE=$2
FINAL_TASK_IDX=$3


CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_continual_without_boundary.py \
    ImageNet50_5Tasks_CondConti -1 dummy $BACKBONE $FINAL_TASK_IDX
