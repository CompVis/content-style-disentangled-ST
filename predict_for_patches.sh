#!/usr/bin/env bash
# INFERENCE_ON_PATCHES

declare -a CKPTS=(25000 50000 75000
                          100000 200000 300000 400000 500000 600000)

DEV_NMBR=1
for ((i=0;i<${#CKPTS[@]};++i))
do

    MODEL_NAME=Project3_5_5_8_ckpt${CKPTS[i]}

    PATH1_INPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/input/coco_full/
    PATH1_OUTPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/output/${MODEL_NAME}/coco_full/
    PATH2_INPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/input/coco_positive/
    PATH2_OUTPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/output/${MODEL_NAME}/coco_positive/
    PATH3_INPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/input/coco_negative/
    PATH3_OUTPUT=/export/home/dkotoven/workspace/StyleTransferProject3/final_results/patches/output/${MODEL_NAME}/coco_negative/


    CUDA_VISIBLE_DEVICES=$DEV_NMBR$ python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                          --batch_size=1 \
                                          --continue_train=True \
                                          --phase=test \
                                          --image_size=768 \
                                          --artist_slug=vincent-van-gogh  \
                                          --window_size=256 \
                                          --ii_dir=${PATH1_INPUT} \
                                          --original_size_inference=True \
                                          --save_dir=${PATH1_OUTPUT} \
                                          --ckpt_nmbr=${CKPTS[i]}


    CUDA_VISIBLE_DEVICES=$DEV_NMBR$ python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                          --batch_size=1 \
                                          --continue_train=True \
                                          --phase=test \
                                          --image_size=768 \
                                          --artist_slug=vincent-van-gogh  \
                                          --window_size=256 \
                                          --ii_dir=${PATH2_INPUT} \
                                          --original_size_inference=True \
                                          --save_dir=${PATH2_OUTPUT} \
                                          --ckpt_nmbr=${CKPTS[i]}

    CUDA_VISIBLE_DEVICES=$DEV_NMBR$ python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                          --batch_size=1 \
                                          --continue_train=True \
                                          --phase=test \
                                          --image_size=768 \
                                          --artist_slug=vincent-van-gogh  \
                                          --window_size=256 \
                                          --ii_dir=${PATH3_INPUT} \
                                          --original_size_inference=True \
                                          --save_dir=${PATH3_OUTPUT} \
                                          --ckpt_nmbr=${CKPTS[i]}

done
