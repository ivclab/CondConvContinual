#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {START_TASK_IDX} {BACKBONE}"
    exit 0;
fi


DATASETS=(
    'dummy'
    'A10'
    'A20'
    'A30'
    'A40'
    'A50'
)


GPU_ID=$1
START_TASK_IDX=$2
BACKBONE=$3


for TASK_IDX in `seq $START_TASK_IDX 5`;
do
    DATASET=${DATASETS[$TASK_IDX]}
    echo "Task Index: ${TASK_IDX}, DATASET: ${DATASET}"

    if [ $TASK_IDX -lt $START_TASK_IDX ]; then
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_continual.py \
        ImageNet50_5Tasks_CondConti $TASK_IDX $DATASET $BACKBONE
done
