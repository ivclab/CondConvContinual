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


for INDEX in `seq $START_IDX $END_IDX`;
do
    DATASET=${DATASETS[$INDEX]}
    echo "DATASET: ${DATASET}"

    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_individual.py \
        CIFAR100_20Tasks_Individual $DATASET baseline $BACKBONE
done

