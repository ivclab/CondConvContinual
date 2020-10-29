#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {BACKBONE} {FINAL_TASK_IDX}"
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
BACKBONE=$2
FINAL_TASK_IDX=$3


for TASK_IDX in `seq 1 $FINAL_TASK_IDX`;
do
    DATASET=${DATASETS[$TASK_IDX]}
    echo "Task Index: ${TASK_IDX}, DATASET: ${DATASET}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_continual_with_boundary.py \
        CIFAR100_20Tasks_CondConti $TASK_IDX $DATASET $BACKBONE $FINAL_TASK_IDX
done
