#!/usr/bin/env bash

tensorboard \
--logdir=\
Project4_model1:./Project4/models/model1_van-gogh_cezanne/logs/,\
Project4_model2:./Project4/models/model1_van-gogh_cezanne_cslw2/logs,\
Project4_model3:./Project4/models/model3_van-gogh_cezanne_cslw200/logs/,\
Project4_model4:./Project4/models/model4_van-gogh_cezanne_correct_disc_acc/logs,\
Project4_model5:./Project4/models/model5_van-gogh_cezanne_correct_cclw1000_cslw1/logs/,\
Project4_model6:./Project4/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model6:./Project5/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs \
--port=1235

tensorboard \
--logdir=\
Project4_model6:./Project4/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model6:./Project5/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model7:./Project5/models/model7_van-gogh_cezanne_correct_cclw20_cslw20_style_dim2/logs/,\
Project6_model1:./Project6/models/model1_van-gogh_cezanne_correct_cclw20_cslw20_cfplw20_sfplw20_style_dim2/logs/,\
Project6_model2:./Project6/models/model2_van-gogh_cezanne_correct_cclw20_cslw20_cfplw2_sfplw2_style_dim2/logs/,\
Project6_model3:./Project6/models/model3_van-gogh_cezanne_correct_cclw20_cslw50_cfplw2_sfplw2_style_dim2_cosine_loss/logs/,\
Project6_model4:./Project6/models/model4_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw2_style_dim2_cosine_loss/logs/ \
--port=1235

tensorboard \
--logdir=\
Project4_model6:./Project4/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model6:./Project5/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model7:./Project5/models/model7_van-gogh_cezanne_correct_cclw20_cslw20_style_dim2/logs/,\
Project6_model4:./Project6/models/model4_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw2_style_dim2_cosine_loss/logs/,\
Project6_1_model1:./Project6_1/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss/logs/,\
Project6_1_model2:./Project6_1/models/model2_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim2_cosine_loss/logs/,\
Project6_1_model3:./Project6_1/models/model3_many_cclw10_cslw20_cfplw2_sfplw20_style_dim16/logs/ \
--port=1235

tensorboard \
--logdir=\
Project4_model6:./Project4/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model6:./Project5/models/model6_van-gogh_cezanne_correct_cclw20_cslw20/logs,\
Project5_model7:./Project5/models/model7_van-gogh_cezanne_correct_cclw20_cslw20_style_dim2/logs/,\
Project6_model4:./Project6/models/model4_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw2_style_dim2_cosine_loss/logs/,\
Project6_1_model1:./Project6_1/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss/logs/,\
Project6_1_model2:./Project6_1/models/model2_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim2_cosine_loss/logs/,\
Project6_1_model3:./Project6_1/models/model3_many_cclw10_cslw20_cfplw2_sfplw20_style_dim16/logs/,\
Project6_1_model4:./Project6_1/models/model4_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim16/logs/,\
Project6_1_model6:./Project6_1/models/model6_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim64/logs/ \
--port=1235



tensorboard \
--logdir=\
Project6_model4:./Project6/models/model4_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw2_style_dim2_cosine_loss/logs/,\
Project6_1_model1:./Project6_1/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss/logs/,\
Project6_1_model4:./Project6_1/models/model4_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim16/logs/,\
Project6_1_model6:./Project6_1/models/model6_many2_cclw10_cslw200_cfplw2_sfplw20_style_dim64/logs/,\
Project6_1_4_model1:./Project6_1_4/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss/logs/,\
Project6_1_4_model1_from_Project6_1:./Project6_1_4/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss_from_Project6_1/logs/,\
Project6_1_4_3_model1:./Project6_1_4_3/models/model1_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss/logs/,\
Project6_1_4_3_1_model2:./Project6_1_4_3_1/models/model2_van-gogh_cezanne_correct_cclw10_cslw20_cfplw2_sfplw20_style_dim16_cosine_loss_margin0.5/logs/ \
--port=1235