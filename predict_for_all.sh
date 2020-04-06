#!/usr/bin/env bash

INPUT_DIR=/export/home/dkotoven/workspace/Places2_dataset/for_eccv_paper
#INPUT_DIR=/export/home/asanakoy/workspace/gan_style/data/mscoco_val_200
DEV_NMBR=1
IMG_SIZE=1280
WINDOW_SIZE=256
# Final checkpoints so far:

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_cezanne_dlw20_clw1000_ilw200_batch8 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=paul-cezanne \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=500000

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_gauguin_dlw20_clw1000_ilw200_batch8 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=paul-gauguin \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=560000

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_kirchner_dlw20_clw1000_ilw200_batch8 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=ernst-ludwig-kirchner \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=540000

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_monet_dlw20_clw1000_ilw400_batch4 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=claude-monet \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=550000

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_picasso_dlw20_clw1000_ilw200_batch8 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=pablo-picasso \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=540000

CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py \
                                      --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8_sz768 \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=$IMG_SIZE \
                                      --artist_slug=vincent-van-gogh \
                                      --ii_dir=${INPUT_DIR} \
                                      --window_size=$WINDOW_SIZE \
                                      --ckpt_nmbr=750000

CUDA_VISIBLE_DEVICES=0 python main.py \
                                      --model_name=model4_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw2_style_dim2_cosine_loss \
                                      --continue_train=True \
                                      --phase=test \
                                      --image_size=1280 \
                                      --artists_list=vincent-van-gogh,paul-cezanne \
                                      --ii_dir=/export/home/dkotoven/workspace/Places2_dataset/for_eccv_paper \
                                      --window_size=384 \
                                      --ckpt_nmbr=950000 \
                                      --style_dim=2




# For model 1
CUDA_VISIBLE_DEVICES=2 python main.py --model_name=model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss \
                                      --image_size=1280 \
                                      --window_size=384 \
                                      --continue_train=True \
                                      --phase=test \
                                      --artists_list=vincent-van-gogh,paul-cezanne \
                                      --ii_dir=/export/home/dkotoven/workspace/Places2_dataset/for_eccv_paper \
                                      --style_dim=16 \
                                      --ckpt_nmbr=850000


CUDA_VISIBLE_DEVICES=2 python main.py --model_name=model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss \
                                      --image_size=1280 \
                                      --window_size=384 \
                                      --continue_train=True \
                                      --phase=test_from_patches \
                                      --artists_list=vincent-van-gogh,paul-cezanne \
                                      --ii_dir=/export/home/dkotoven/workspace/Places2_dataset/for_eccv_paper \
                                      --style_dim=16 \
                                      --ckpt_nmbr=850000

# Extracting style features:
CUDA_VISIBLE_DEVICES=0 python main.py --model_name=model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss \
                                      --image_size=1280 \
                                      --window_size=384 \
                                      --continue_train=True \
                                      --phase=extract_style_feats \
                                      --artists_list=vincent-van-gogh,paul-cezanne \
                                      --style_dim=16 \
                                      --ckpt_nmbr=850000