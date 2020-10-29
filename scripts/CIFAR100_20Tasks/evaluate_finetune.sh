#!/bin/bash


if [ $# -eq 0 ];
then
    echo "ARGUMENTS: {GPU_ID} {START_IDX} {END_IDX} {BACKBONE}"
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
START_IDX=$2
END_IDX=$3
BACKBONE=$4


if [ $START_IDX -eq 1 ]; then
    echo "Start Index cannot be 1!"
    exit
fi


for INDEX in `seq $START_IDX $END_IDX`;
do
    DATASET=${DATASETS[$INDEX]}
    echo "DATASET: ${DATASET}"

    for PREV_INDEX in `seq 1 $(($INDEX - 1))`;
    do
        PREV_DATASET=${DATASETS[$PREV_INDEX]}
        PRETRAIN=CHECKPOINTS/Individual/CIFAR100_20Tasks/$BACKBONE/$PREV_DATASET/baseline
        echo "PRETRAIN: ${PRETRAIN}"

        CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_individual.py \
            CIFAR100_20Tasks_Individual $DATASET finetune $BACKBONE $PRETRAIN _${PREV_DATASET}
    done
done
