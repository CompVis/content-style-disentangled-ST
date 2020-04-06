#!/usr/bin/env bash

#declare -a CKPTS=(650000 750000)
declare -a CKPTS=(650000 675000 700000 725000 750000 775000 800000 825000 850000 875000 900000)

DEV_NMBR=0
SIZE=768
PATH_DATA=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/coco_images_for_person_det

for ((i=0;i<${#CKPTS[@]};++i))
do
    MODEL_NAME=Project3_5_5_8_ckpt${CKPTS[i]}_sz${SIZE}

    PATH1_INPUT=${PATH_DATA}/input/positive
    PATH1_OUTPUT=${PATH_DATA}/output/${MODEL_NAME}/positive/
    PATH2_INPUT=${PATH_DATA}/input/negative
    PATH2_OUTPUT=${PATH_DATA}/output/${MODEL_NAME}/negative/

    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                          --batch_size=1 \
                                          --continue_train=True \
                                          --phase=test \
                                          --image_size=${SIZE} \
                                          --artist_slug=vincent-van-gogh  \
                                          --window_size=256 \
                                          --ii_dir=${PATH1_INPUT} \
                                          --original_size_inference=0 \
                                          --save_dir=${PATH1_OUTPUT} \
                                          --ckpt_nmbr=${CKPTS[i]}


    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                          --batch_size=1 \
                                          --continue_train=True \
                                          --phase=test \
                                          --image_size=${SIZE} \
                                          --artist_slug=vincent-van-gogh  \
                                          --window_size=256 \
                                          --ii_dir=${PATH2_INPUT} \
                                          --original_size_inference=0 \
                                          --save_dir=${PATH2_OUTPUT} \
                                          --ckpt_nmbr=${CKPTS[i]}

done
