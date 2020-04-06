#!/usr/bin/env bash


#DEV_NMBR=0
#MODEL_NAME=model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss
#
#
#declare -a IMAGE_SIZES=(256 384 512 640 768)
#declare -a BATCH_SIZES=(8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000)
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
#declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
#declare -a MARGINS=(0.2 0.3 0.4 0.5 0.5 0.5)
#
#for ((i=5;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d. Margin %f\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}" "${MARGINS[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=16 \
#                                          --margin=${MARGINS[i]}
#
#done




#DEV_NMBR=3
#MODEL_NAME=model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss_from_Project6_1
#
#
#declare -a IMAGE_SIZES=(256 384 512 640 768)
#declare -a BATCH_SIZES=(8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000)
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
#declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
#
#for ((i=5;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=16 \
#                                          --ckpt_nmbr=400000
#
#done
#

DEV_NMBR=1
MODEL_NAME=model2_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim64_cosine_loss_margin0.5


declare -a IMAGE_SIZES=(256 384 512 640 768)
declare -a BATCH_SIZES=(8 4 2 1 1)
declare -a WINDOW_SIZES=(128 192 256 320 384)
declare -a TOTAL_STEPS=(200 400 600 800 1000)


declare -a IMAGE_SIZES=(128 256 384 512 640 768)
declare -a BATCH_SIZES=(16 8 4 2 1 1)
declare -a WINDOW_SIZES=(128 128 192 256 320 384)
declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
declare -a MARGINS=(0.2 0.3 0.4 0.5 0.5 0.5)

for ((i=0;i<${#IMAGE_SIZES[@]};++i))
do
    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d. Margin %f\n" \
    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}" "${MARGINS[i]}"

    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
                                          --batch_size=${BATCH_SIZES[i]} \
                                          --image_size=${IMAGE_SIZES[i]} \
                                          --window_size=${WINDOW_SIZES[i]} \
                                          --total_steps=${TOTAL_STEPS[i]} \
                                          --continue_train=True \
                                          --phase=train \
                                          --artists_list=vincent-van-gogh,paul-cezanne \
                                          --lr=0.0002 \
                                          --dsr=0.8 \
                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
                                          --style_dim=64 \
                                          --margin=${MARGINS[i]}

done


#DEV_NMBR=2
#MODEL_NAME=model2_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim2_cosine_loss
#
#
#declare -a IMAGE_SIZES=(256 384 512 640 768)
#declare -a BATCH_SIZES=(8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000)
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
#declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
#
#for ((i=0;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=2
#
#done
#


#DEV_NMBR=1
#MODEL_NAME=model3_many_cclw10_cslw20_cfplw2_sfplw20_style_dim16
#
#
#declare -a IMAGE_SIZES=(256 384 512 640 768)
#declare -a BATCH_SIZES=(8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000)
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
#declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
#
#for ((i=5;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne,ernst-ludwig-kirchner,claude-monet,edvard-munch,paul-gauguin,berthe-morisot,wassily-kandinsky,el-greco \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=16
#
#done


#DEV_NMBR=1
#MODEL_NAME=model4_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim16
#
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a BATCH_SIZES=(16 6 3 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(100000 200000 300000 350000 400000 1000000)
#
#for ((i=3;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne,ernst-ludwig-kirchner,claude-monet,edvard-munch,paul-gauguin,berthe-morisot,wassily-kandinsky,el-greco,samuel-peploe \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=200 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=16
#
#done


#DEV_NMBR=4
#MODEL_NAME=model5_many2_cclw10_cslw20_cfplw2_sfplw20_style_dim16_dsr0.5
#
#
#declare -a IMAGE_SIZES=(256 384 512 640 768)
#declare -a BATCH_SIZES=(8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000)
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(200 400 600 800 1000 1200 1400)
#declare -a TOTAL_STEPS=(100000 150000 200000 250000 300000 1000000)
#
#for ((i=0;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne,ernst-ludwig-kirchner,claude-monet,edvard-munch,paul-gauguin,berthe-morisot,wassily-kandinsky,el-greco,samuel-peploe \
#                                          --lr=0.0002 \
#                                          --dsr=0.5 \
#                                          --dlw=20 --cclw=10 --cslw=20 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=16
#
#done


#DEV_NMBR=3
#MODEL_NAME=model6_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim64
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a BATCH_SIZES=(16 6 3 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(100000 200000 300000 350000 400000 1000000)
#
#for ((i=1;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne,ernst-ludwig-kirchner,claude-monet,edvard-munch,paul-gauguin,berthe-morisot,wassily-kandinsky,el-greco,samuel-peploe \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=200 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=64
#
#done
#



#DEV_NMBR=2
#MODEL_NAME=model7_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim4096_no_normality
#
#
#declare -a IMAGE_SIZES=(128 256 384 512 640 768)
#declare -a BATCH_SIZES=(16 8 4 2 1 1)
#declare -a BATCH_SIZES=(16 6 3 2 1 1)
#declare -a WINDOW_SIZES=(128 128 192 256 320 384)
#declare -a TOTAL_STEPS=(100000 200000 300000 350000 400000 1000000)
#
#for ((i=0;i<${#IMAGE_SIZES[@]};++i))
#do
#    printf "Train model %s on patches of size %d with %d patches in batch for %d steps. Window size is %d\n" \
#    "$MODEL_NAME" "${IMAGE_SIZES[i]}" "${BATCH_SIZES[i]}" "${TOTAL_STEPS[i]}" "${WINDOW_SIZES[i]}"
#
#    CUDA_VISIBLE_DEVICES=$DEV_NMBR python main.py --model_name=${MODEL_NAME} \
#                                          --batch_size=${BATCH_SIZES[i]} \
#                                          --image_size=${IMAGE_SIZES[i]} \
#                                          --window_size=${WINDOW_SIZES[i]} \
#                                          --total_steps=${TOTAL_STEPS[i]} \
#                                          --continue_train=True \
#                                          --phase=train \
#                                          --artists_list=vincent-van-gogh,paul-cezanne,ernst-ludwig-kirchner,claude-monet,edvard-munch,paul-gauguin,berthe-morisot,wassily-kandinsky,el-greco,samuel-peploe \
#                                          --lr=0.0002 \
#                                          --dsr=0.8 \
#                                          --dlw=20 --cclw=10 --cslw=200 --ilw=200 --cfplw=2 --sfplw=20 --cplw=0  --tvlw=0 \
#                                          --style_dim=4096
#
#done
#


