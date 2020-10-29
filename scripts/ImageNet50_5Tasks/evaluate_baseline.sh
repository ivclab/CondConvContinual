#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {START_IDX} {END_IDX} {BACKBONE}"
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
START_IDX=$2
END_IDX=$3
BACKBONE=$4


for INDEX in `seq $START_IDX $END_IDX`;
do
    DATASET=${DATASETS[$INDEX]}
    echo "DATASET: ${DATASET}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_individual.py \
        ImageNet50_5Tasks_Individual $DATASET baseline $BACKBONE
done
