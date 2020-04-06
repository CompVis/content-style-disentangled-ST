#!/usr/bin/env bash

# Final models so far
CUDA_VISIBLE_DEVICES=4 python main.py --model_name=model_van-gogh_dlw20_clw1000_ilw200_batch8 \
                                      --batch_size=8 \
                                      --continue_train=True \
                                      --phase=train \
                                      --image_size=768 \
                                      --artist_slug=vincent-van-gogh  \
                                      --lr=0.0002 \
                                      --dsr=0.8 \
                                      --window_size=256 \
                                      --clw=1000 --cplw=0 --dlw=20 --flw=1 --ilw=200

