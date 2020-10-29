#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {START_TASK_IDX} {BACKBONE}"
    exit 0;
fi


DATASETS=(
    'dummy'
    'aquatic_mammals'
    'fish'
    'flowers'
    'food_containers'
    'fruit_and_vegetables'
    'household_electrical_devices'
    'household_furniture'
    'insects'
    'large_carnivores'
    'large_man-made_outdoor_things'
    'large_natural_outdoor_scenes'
    'large_omnivores_and_herbivores'
    'medium_mammals'
    'non-insect_invertebrates'
    'people'
    'reptiles'
    'small_mammals'
    'trees'
    'vehicles_1'
    'vehicles_2'
)


GPU_ID=$1
START_TASK_IDX=$2
BACKBONE=$3


for TASK_IDX in `seq $START_TASK_IDX 20`;
do
    DATASET=${DATASETS[$TASK_IDX]}
    echo "Task Index: ${TASK_IDX}, DATASET: ${DATASET}"

    if [ $TASK_IDX -lt $START_TASK_IDX ]; then
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_continual.py \
        CIFAR100_20Tasks_CondConti $TASK_IDX $DATASET $BACKBONE
done
