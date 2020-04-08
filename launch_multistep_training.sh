#!/usr/bin/env bash


DEV_NMBR=0
MODEL_NAME=model_vangogh_cezanne

declare -a IMAGE_SIZES=(128 256 384 512 640 768)
declare -a BATCH_SIZES=(16 8 4 2 1 1)
declare -a WINDOW_SIZES=(128 128 192 256 320 384)
declare -a TOTAL_STEPS=(50000 75000 100000 125000 150000 1000000)
declare -a MARGINS=(0.2 0.3 0.4 0.5 0.5 0.5)

for ((i=0;i<${#IMAGE_SIZES[@]};++i))
do
    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d. Margin %f\n" \
    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}" "${MARGINS[i]}"

    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
        --ptcd=/export/home_old/dkotoven/workspace/Places2_dataset/data_large \
        --ptad=./data/art_images/cezanne_vangogh/ \
        --batch_size=${BATCH_SIZES[i]} \
        --image_size=${IMAGE_SIZES[i]} \
        --window_size=${WINDOW_SIZES[i]} \
        --total_steps=${TOTAL_STEPS[i]} \
        --continue_train=True \
        --phase=train \
        --lr=0.0002 \
        --dsr=0.5 \
        --dlw=20 --cclw=10 --cslw=20 --ilw=400 --cfplw=2 --sfplw=50 --cplw=0  --tvlw=0 \
        --style_dim=16 \
        --margin=${MARGINS[i]} \



done


