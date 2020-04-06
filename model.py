from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import pickle
from tqdm import tqdm
import multiprocessing
import pandas as pd
import yaml
import pickle
import random

from module import *
from utils import *
import ops
import prepare_dataset
import img_augm
import datetime
import pprint

import tensorflow.contrib.distributions

from scipy.ndimage.filters import gaussian_filter
from itertools import product


eps = 1e-6

class Artgan(object):
    def __init__(self, sess, args):
        self.model_name = args.model_name
        self.root_dir = './models'
        self.checkpoint_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint')
        self.checkpoint_long_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint_long')
        self.sample_dir = os.path.join(self.root_dir, self.model_name, 'sample')
        self.inference_dir = os.path.join(self.root_dir, self.model_name, 'inference')
        self.logs_dir = os.path.join(self.root_dir, self.model_name, 'logs')

        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        self.loss = sce_criterion

        self.initial_step = 0

        with open("config.yaml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        OPTIONS = namedtuple('OPTIONS',
                             'batch_size image_size window_size style_dim margin \
                              total_steps save_freq lr \
                              discr_success_rate \
                              continue_train \
                              gf_dim df_dim cf_dim \
                              blurring_kernel_size \
                              generator_steps discriminator_steps \
                              output_c_dim \
                              is_training \
                              artists_list \
                              artists_number \
                              techniques_list \
                              techniques_number \
                              path_to_wikiart \
                              path_to_art_dataset \
                              path_to_content_dataset \
                              loss_weights')

        self.options = OPTIONS._make((args.batch_size, args.image_size, args.window_size, args.style_dim, args.margin,
                                      args.total_steps, args.save_freq, args.lr,
                                      args.discr_success_rate,
                                      args.continue_train,
                                      args.ngf, args.ndf, args.ncf,
                                      args.blurring_kernel_size,
                                      args.generator_steps, args.discriminator_steps,
                                      args.output_nc,
                                      args.phase == 'train',
                                      args.artists_list,
                                      len(args.artists_list),
                                      cfg['techniques_list'],
                                      len(cfg['techniques_list']),
                                      cfg['path_to_wikiart'],
                                      cfg['path_to_art_dataset'],
                                      cfg['path_to_content_dataset'],
                                      {'discr': args.discr_loss_weight,
                                       'clsf_cont': args.clsf_cont_loss_weight,
                                       'clsf_style': args.clsf_style_loss_weight,
                                       'image': args.image_loss_weight,
                                       'fp_content': args.cont_fp_loss_weight,
                                       'fp_style': args.style_fp_loss_weight,
                                       'content_preservation': args.content_preservation_loss_weight,
                                       'tv': args.tv_loss_weight}
                                      ))

        # Create all the folders for saving the model
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(os.path.join(self.root_dir, self.model_name)):
            os.makedirs(os.path.join(self.root_dir, self.model_name))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.checkpoint_long_dir):
            os.makedirs(self.checkpoint_long_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.inference_dir):
            os.makedirs(self.inference_dir)

        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=2)#,var_list=[x for x in tf.trainable_variables() if 'oving' not in x.name])
        self.saver_long = tf.train.Saver(max_to_keep=None)#,var_list=[x for x in tf.trainable_variables() if 'oving' not in x.name])
        # Create or append log file.
        with open(os.path.join(self.root_dir, self.model_name, 'model.log'), 'a') as f:
            print("Current time:", datetime.datetime.now(), file=f)
            pprint.PrettyPrinter(indent=4, stream=f).pprint(dict(self.options._asdict()))
            print('\n'*5, file=f)

    def _build_graph(self):
        self.__build_graph_generator(self.options.is_training)

        if self.options.is_training:
            self.__build_graph_discriminator()
            self.__build_graph_losses()
            self.__build_graph_optimizers()
            self.__build_graph_summaries()



    def __build_graph_generator(self, is_training):

        if not is_training:
            with tf.name_scope('placeholder'):
                self.input_photo = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None, None, 3],
                                                  name='photo')
                self.input_painting = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, None, None, 3],
                                                     name='painting')
                self.input_painting_label = tf.placeholder(dtype=tf.float32,
                                                           shape=[self.batch_size, len(self.options.artists_list)],
                                                           name='painting_label')

            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                kernels=self.options.gf_dim,
                                                options=self.options,
                                                is_training=self.options.is_training,
                                                reuse=False)
            self.input_painting_features_style = encoder_style(image=self.input_painting,
                                                kernels=self.options.gf_dim,
                                                options=self.options,
                                                is_training=self.options.is_training,
                                                reuse=False)

            # Decode obtained features together with labels.
            self.output_photo = decoder(features=self.input_photo_features,
                                        style=self.input_painting_features_style,
                                        kernels=self.options.gf_dim,
                                        options=self.options,
                                        is_training=self.options.is_training,
                                        reuse=False)
        else:
            # ==================== Define placeholders. ===================== #
            with tf.name_scope('placeholder'):
                self.input_painting = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, None, None, 3],
                                                     name='painting')
                self.input_painting_negative = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, None, None, 3],
                                                              name='painting_negative')
                self.input_painting_label = tf.placeholder(dtype=tf.float32,
                                                           shape=[self.batch_size, len(self.options.artists_list)],
                                                           name='painting_label')
                self.input_photo = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None, None, 3],
                                                  name='photo')
                self.input_photo_negative = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None, None, 3],
                                                  name='photo_negative')
                self.input_photo_label = tf.placeholder(dtype=tf.float32,
                                                        shape=[self.batch_size,
                                                               len(prepare_dataset.PlacesDataset.categories_names)]
                                                        )
                self.lr = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

            # ===================== Wire the graph. ========================= #
            # Encode input images.
            self.input_photo_features = encoder(image=self.input_photo,
                                                kernels=self.options.gf_dim,
                                                options=self.options,
                                                is_training=self.options.is_training,
                                                reuse=False)
            self.input_photo_negative_features = encoder(image=self.input_photo_negative,
                                                kernels=self.options.gf_dim,
                                                options=self.options,
                                                is_training=self.options.is_training,
                                                reuse=True)

            self.input_painting_features_style = encoder_style(image=self.input_painting,
                                                         kernels=self.options.gf_dim,
                                                         options=self.options,
                                                         is_training=self.options.is_training,
                                                         reuse=False)
            self.input_painting_negative_features_style = encoder_style(image=self.input_painting_negative,
                                                               kernels=self.options.gf_dim,
                                                               options=self.options,
                                                               is_training=self.options.is_training,
                                                               reuse=True)
            # Decode obtained features together with labels.
            self.output_photo = decoder(features=self.input_photo_features,
                                        style=self.input_painting_features_style,
                                        kernels=self.options.gf_dim,
                                        options=self.options,
                                        is_training=self.options.is_training,
                                        reuse=False)
            self.output_photo_negative = decoder(features=self.input_photo_negative_features,
                                              style=self.input_painting_features_style,
                                              kernels=self.options.gf_dim,
                                              options=self.options,
                                              is_training=self.options.is_training,
                                              reuse=True)
            # To switch to previous version - uncomment lines below.
            # print("self.input_painting_features_style:", self.input_painting_features_style)
            # features_style_rolled = tf.concat(
            #     [tf.slice(self.input_painting_features_style, begin=[1, 0], size=[-1, -1]),
            #      tf.slice(self.input_painting_features_style, begin=[0, 0], size=[1, -1])],
            #     axis=0,
            #     name="features_style_rolled"
            # )

            self.output_photo_shift_style = decoder(features=self.input_photo_features,
                                                    # style=ops.roll(input=self.input_painting_features_style,
                                                    #                     shift=1,
                                                    #                     axis=0),
                                                    style=self.input_painting_negative_features_style,
                                                    kernels=self.options.gf_dim,
                                                    options=self.options,
                                                    is_training=self.options.is_training,
                                                    reuse=True)

            # Get features of output images. Need them to compute feature loss.
            self.output_photo_features = encoder(image=self.output_photo,
                                                 kernels=self.options.gf_dim,
                                                 options=self.options,
                                                 is_training=self.options.is_training,
                                                 reuse=True)


            self.output_photo_features_style = encoder_style(image=self.output_photo,
                                                               kernels=self.options.gf_dim,
                                                               options=self.options,
                                                               is_training=self.options.is_training,
                                                               reuse=True)

            self.output_photo_negative_features_style = encoder_style(image=self.output_photo_negative,
                                                                   kernels=self.options.gf_dim,
                                                                   options=self.options,
                                                                   is_training=self.options.is_training,
                                                                   reuse=True)

            self.output_photo_shift_style_features_style = encoder_style(image=self.output_photo_shift_style,
                                                                         kernels=self.options.gf_dim,
                                                                         options=self.options,
                                                                         is_training=self.options.is_training,
                                                                         reuse=True)

    def __build_graph_discriminator(self):
        self.input_painting_discr_predictions = discriminator(image=self.input_painting,
                                                              kernels=self.options.df_dim,
                                                              options=self.options,
                                                              is_training=self.options.is_training,
                                                              num_clsf_clss=len(
                                                                  prepare_dataset.PlacesDataset.categories_names),
                                                              reuse=False)
        self.input_photo_discr_predictions = discriminator(image=self.input_photo,
                                                           kernels=self.options.df_dim,
                                                           options=self.options,
                                                           is_training=self.options.is_training,
                                                           num_clsf_clss=len(
                                                               prepare_dataset.PlacesDataset.categories_names),
                                                           reuse=True)
        self.output_photo_discr_predictions = discriminator(image=self.output_photo,
                                                            kernels=self.options.df_dim,
                                                            options=self.options,
                                                            is_training=self.options.is_training,
                                                            num_clsf_clss=len(
                                                                prepare_dataset.PlacesDataset.categories_names),
                                                            reuse=True)

    def __build_graph_losses(self):
        # ========================================================================== #
        # ===================== Final losses that we optimize. ===================== #
        # ========================================================================== #

        with tf.name_scope('losses'):
            # Image loss.

            self.img_loss = mse_criterion(blur_img(input_tensor=self.input_photo,
                                                   kernel_size=self.options.blurring_kernel_size),
                                          blur_img(input_tensor=self.output_photo,
                                                   kernel_size=self.options.blurring_kernel_size))

            # Fixpoint loss.
            # self.fixpoint_loss_photo = abs_criterion(self.output_photo_features, self.input_photo_features)
            # self.fixpoint_loss_painting = abs_criterion(self.output_photo_features_style, self.input_painting_features_style)
            self.fixpoint_loss_photo = tf.reduce_mean(cosine_loss(
                a=tf.reshape(self.output_photo_features, shape=(self.batch_size, -1)),
                b=tf.reshape(self.input_photo_features, shape=(self.batch_size, -1))))
            self.fixpoint_loss_painting = \
                0.5 * tf.reduce_mean(triplet_loss(
                    anchor=tf.reshape(self.input_painting_features_style, shape=(self.batch_size, -1)),
                    positive=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                    negative=tf.reshape(self.output_photo_shift_style_features_style, shape=(self.batch_size, -1)))) + \
                0.5 * tf.reduce_mean(triplet_loss(
                    anchor=tf.reshape(self.input_painting_negative_features_style, shape=(self.batch_size, -1)),
                    positive=tf.reshape(self.output_photo_shift_style_features_style, shape=(self.batch_size, -1)),
                    negative=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                ))

            #     + tf.reduce_mean(triplet_loss(
            #     anchor=tf.reshape(self.input_painting_negative_features_style, shape=(self.batch_size, -1)),
            #     positive=tf.reshape(self.output_photo_shift_style_features_style, shape=(self.batch_size, -1)),
            #     negative=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
            # ))

            # Disentanglement loss
            self.disentanglement_loss = tf.reduce_mean(triplet_loss(
                anchor=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                positive=tf.reshape(self.output_photo_negative_features_style, shape=(self.batch_size, -1)),
                negative=tf.reshape(self.input_painting_features_style, shape=(self.batch_size, -1)),
                margin=0.))

            # Style feature normality.
            input_painting_features_style_mean, input_painting_features_style_cov = ops.tf_mean_cov(self.input_painting_features_style)
            self.features_style_normality_loss = tf.contrib.distributions.kl_divergence(
                tf.contrib.distributions.MultivariateNormalFullCovariance(input_painting_features_style_mean,
                                                                          input_painting_features_style_cov + \
                                                                          tf.eye(self.options.style_dim)*eps
                                                                          ),
                tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros_like(input_painting_features_style_mean)))
            # self.features_style_normality_loss = tf.reduce_mean(self.features_style_normality_loss)
            self.features_style_normality_loss = tf.constant(0.)

            # TV-loss
            self.output_photo_tv_loss = tf.reduce_sum(tf.image.total_variation(images=self.output_photo)) / tf.cast(
                tf.reduce_prod(tf.shape(self.output_photo)), tf.float32)
            self.tv_loss = self.output_photo_tv_loss

            # Classifier loss.
            self.output_photo_clsf_cont_loss = sce_criterion(logits=self.output_photo_discr_predictions['content_pred'],
                                                             labels=self.input_photo_label)
            self.input_photo_clsf_cont_loss = sce_criterion(logits=self.input_photo_discr_predictions['content_pred'],
                                                            labels=self.input_photo_label)
            self.clsf_loss = tf.reduce_mean([self.input_photo_clsf_cont_loss, self.output_photo_clsf_cont_loss])

            self.input_photo_clsf_cont_acc = get_clsf_acc(
                in_=self.input_photo_discr_predictions['content_pred'],
                labels_=self.input_photo_label)
            self.output_photo_clsf_cont_acc = get_clsf_acc(
                in_=self.output_photo_discr_predictions['content_pred'],
                labels_=self.input_photo_label)
            self.clsf_acc = tf.reduce_mean([self.input_photo_clsf_cont_acc, self.output_photo_clsf_cont_acc])

            # Style classifier loss.
            self.output_photo_clsf_style_loss = sce_criterion(
                logits=self.output_photo_discr_predictions['style_pred'],
                labels=add_spatial_dim(input_tensor=self.input_painting_label,
                                       dims_list=[1, 2],
                                       resol_list=[self.options.image_size // 2, self.options.image_size // 2]))
            self.input_painting_clsf_style_loss = sce_criterion(
                logits=self.input_painting_discr_predictions['style_pred'],
                labels=add_spatial_dim(input_tensor=self.input_painting_label,
                                       dims_list=[1, 2],
                                       resol_list=[self.options.image_size // 2, self.options.image_size // 2]))

            self.output_photo_clsf_style_acc = get_clsf_acc(
                in_=self.output_photo_discr_predictions['style_pred'],
                labels_=add_spatial_dim(input_tensor=self.input_painting_label,
                                        dims_list=[1, 2],
                                        resol_list=[self.options.image_size // 2, self.options.image_size // 2]))
            self.input_painting_clsf_style_acc = get_clsf_acc(
                in_=self.input_painting_discr_predictions['style_pred'],
                labels_=add_spatial_dim(input_tensor=self.input_painting_label,
                                        dims_list=[1, 2],
                                        resol_list=[self.options.image_size // 2, self.options.image_size // 2]))


            # Content preservation loss.
            self.content_preservation_loss = mse_criterion(self.input_photo_discr_predictions["scale6"],
                                                           self.output_photo_discr_predictions["scale6"])

            # Discriminator.
            # Have to predict ones only for original paintings, otherwise predict zero.
            self.input_painting_discr_loss = sce_criterion(
                logits=self.input_painting_discr_predictions["discr_pred"],
                labels=tf.ones_like(self.input_painting_discr_predictions["discr_pred"]))

            # self.input_photo_discr_loss = sce_criterion(
            #     logits=self.input_photo_discr_predictions["discr_pred"],
            #     labels=tf.zeros_like(self.input_photo_discr_predictions["discr_pred"]))

            self.output_photo_discr_loss = sce_criterion(
                logits=self.output_photo_discr_predictions["discr_pred"],
                labels=tf.zeros_like(self.output_photo_discr_predictions["discr_pred"]))

            self.discr_loss = tf.reduce_mean([self.input_painting_discr_loss,
                                              self.output_photo_discr_loss])


            # Compute discriminator accuracies.
            self.input_painting_discr_acc = tf.reduce_mean(
                input_tensor=tf.cast(x=self.input_painting_discr_predictions["discr_pred"] > \
                                       tf.zeros_like(self.input_painting_discr_predictions["discr_pred"]),
                                     dtype=tf.float32)
            )

            self.output_photo_discr_acc = tf.reduce_mean(
                input_tensor=tf.cast(x=self.output_photo_discr_predictions["discr_pred"] < \
                                       tf.zeros_like(self.output_photo_discr_predictions["discr_pred"]),
                                     dtype=tf.float32)
            )

            self.discr_acc = tf.reduce_mean([self.input_painting_discr_acc, self.output_photo_discr_acc])

            # Generator.
            # Predicts ones for both output images.
            self.output_photo_gener_loss = sce_criterion(
                logits=self.output_photo_discr_predictions["discr_pred"],
                labels=tf.ones_like(self.output_photo_discr_predictions["discr_pred"]))

            self.gener_loss = self.output_photo_gener_loss+1e-6

            # Compute generator accuracies.
            self.output_photo_gener_acc = tf.reduce_mean(
                input_tensor=tf.cast(x=self.output_photo_discr_predictions["discr_pred"] > \
                                       tf.zeros_like(self.output_photo_discr_predictions["discr_pred"]),
                                     dtype=tf.float32)
            )

            self.gener_acc = self.output_photo_gener_acc+1e-6

    def __build_graph_optimizers(self):
        # ============================================================= #
        # ================== Define optimization steps. =============== #
        # ============================================================= #
        with tf.name_scope('optimizers'):
            t_vars = tf.trainable_variables()
            self.discr_vars = [var for var in t_vars if 'discriminator' in var.name]
            self.encoder_vars = [var for var in t_vars if 'encoder' in var.name]
            self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]

            # Discriminator and generator steps.
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_optim_step = tf.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.loss_weights['discr'] * self.discr_loss +
                         self.options.loss_weights['clsf_cont'] * self.input_photo_clsf_cont_loss +
                         self.options.loss_weights['clsf_style'] * self.input_painting_clsf_style_loss +
                         self.options.loss_weights['content_preservation'] * self.content_preservation_loss,
                    var_list=[self.discr_vars])
                self.g_optim_step = tf.train.AdamOptimizer(self.lr).minimize(
                    loss=self.options.loss_weights['discr'] * self.gener_loss +
                         self.options.loss_weights['image'] * self.img_loss +
                         self.options.loss_weights['fp_content'] * self.fixpoint_loss_photo +
                         self.options.loss_weights['fp_style'] * self.fixpoint_loss_painting +
                         self.options.loss_weights['fp_style'] * self.disentanglement_loss +
                         self.options.loss_weights['fp_style'] * self.features_style_normality_loss +
                         self.options.loss_weights['tv'] * self.tv_loss +
                         self.options.loss_weights['clsf_cont'] * self.output_photo_clsf_cont_loss +
                         self.options.loss_weights['clsf_style'] * self.output_photo_clsf_style_loss +
                         self.options.loss_weights['content_preservation'] * self.content_preservation_loss,
                    var_list=[self.encoder_vars + self.decoder_vars])

    def __build_graph_summaries(self):
        # ============================================================= #
        # ============= Write statistics to tensorboard. ============== #
        # ============================================================= #

        # Discriminator loss summary writing.
        discriminator_summary_vals = {
            # Losses
            "discriminator/loss/input_painting_discr": self.input_painting_discr_loss,
            "discriminator/loss/output_photo_discr": self.output_photo_discr_loss,
            "discriminator/loss/discr": self.discr_loss,
            "discriminator/loss/input_photo_clsf_cont": self.input_photo_clsf_cont_loss,
            "discriminator/loss/input_painting_clsf_style": self.input_painting_clsf_style_loss,

            # Accuracies
            "discriminator/acc/input_painting_discr": self.input_painting_discr_acc,
            "discriminator/acc/output_photo_discr": self.output_photo_discr_acc,
            "discriminator/acc/discr_loss": self.discr_acc,
            "discriminator/acc/input_photo_clsf_cont": self.input_photo_clsf_cont_acc,
            "discriminator/acc/input_painting_clsf_style": self.input_painting_clsf_style_acc
        }

        # Generator loss summary writing.
        generator_summary_vals = {
            "discriminator/loss/output_photo_gener": self.output_photo_gener_loss,
            "discriminator/loss/gener": self.gener_loss,
            "image/loss/": self.img_loss,
            "fixpoint/loss/content": self.fixpoint_loss_photo,
            "fixpoint/loss/style": self.fixpoint_loss_painting,
            "feature_normality/loss": self.features_style_normality_loss,
            "total_variation/loss": self.tv_loss,
            "disentanglement/loss": self.disentanglement_loss,
            "discriminator/loss/output_photo_clsf_cont": self.output_photo_clsf_cont_loss,
            "discriminator/loss/output_photo_clsf_style": self.output_photo_clsf_style_loss,
            "discriminator/loss/content_preservation": self.content_preservation_loss,
            # Accuracies
            "discriminator/acc/output_photo_gener": self.output_photo_gener_acc,
            "discriminator/acc/gener": self.gener_acc,
            "discriminator/acc/output_photo_clsf_cont": self.output_photo_clsf_cont_acc,
            "discriminator/acc/output_photo_clsf_style": self.output_photo_clsf_style_acc,
        }

        ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)

        with tf.control_dependencies(discriminator_summary_vals.values()+generator_summary_vals.values()):
            self.summary_accumulation = tf.group(ema.apply(discriminator_summary_vals.values() +
                                                           generator_summary_vals.values()))

        moving_average = [ema.average(x) for x in discriminator_summary_vals.values() + generator_summary_vals.values()]

        self.summary_writing = tf.summary.merge([
            tf.summary.scalar(name, val) for name, val in zip(discriminator_summary_vals.keys() +
                                                              generator_summary_vals.keys(),
                                                              moving_average)])


        # TODO
        # Compute fractions of each loss contributed to each of the losses.

        # self.summary_losses_fractions = tf.summary.merge([
        #     # For discriminator first:
        #     tf.summary.scalar("losses_fractions/discriminator_step/discr_loss",
        #                       self.options.loss_weights['discr'] * self.discr_loss / discriminator_step_loss),
        #     tf.summary.scalar("losses_fractions/discriminator_step/input_photo_clsf_loss",
        #                       self.options.loss_weights['clsf'] * self.input_photo_clsf_loss / discriminator_step_loss),
        #     tf.summary.scalar("losses_fractions/discriminator_step/input_painting_clsf_style_loss",
        #                       self.options.loss_weights['clsf'] * self.input_painting_clsf_style_loss / discriminator_step_loss),
        #     tf.summary.scalar("losses_fractions/discriminator_step/content_preservation_loss",
        #                       self.options.loss_weights['content_preservation'] * self.content_preservation_loss / discriminator_step_loss),
        #     # Now for generator:
        #     tf.summary.scalar("losses_fractions/generator_step/discr_loss",
        #                       self.options.loss_weights['discr'] * self.gener_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/img_loss",
        #                       self.options.loss_weights['image'] * self.img_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/fixpoint_loss_photo",
        #                       self.options.loss_weights['feature'] * self.fixpoint_loss_photo / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/fixpoint_loss_painting",
        #                       self.options.loss_weights['feature_style'] * self.fixpoint_loss_painting / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/features_style_normality_loss",
        #                       self.options.loss_weights['feature_style'] * self.features_style_normality_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/tv_loss",
        #                       self.options.loss_weights['tv'] * self.tv_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/output_photo_clsf_loss",
        #                       self.options.loss_weights['clsf'] * self.output_photo_clsf_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/output_photo_clsf_style_loss",
        #                       self.options.loss_weights['clsf'] * self.output_photo_clsf_style_loss / generator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step/content_preservation_loss",
        #                       self.options.loss_weights['content_preservation'] * self.content_preservation_loss / generator_step_loss),
        #     # Now discriminator to generator loss ratio
        #     tf.summary.scalar("losses_fractions/discriminator_step_loss",
        #                       discriminator_step_loss),
        #     tf.summary.scalar("losses_fractions/generator_step_loss",
        #                       discriminator_step_loss),
        #     tf.summary.scalar("losses_fractions/discriminator_step_loss_to_generator_step_loss ratio",
        #                       discriminator_step_loss / generator_step_loss),
        # ])

        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

    def train(self, args, ckpt_nmbr=None):
        # Initialize datasets and augmentor.
        content_dataset_places = prepare_dataset.PlacesDataset(path_to_dataset=self.options.path_to_content_dataset)
        print("self.options.artists_list:", self.options.artists_list)
        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset,
                                                 artists_list=self.options.artists_list)
        augmentor = img_augm.Augmentor(crop_size=[self.options.image_size, self.options.image_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.10,
                                       saturation_augm_shift=0.10, saturation_augm_scale=0.10,
                                       value_augm_shift=0.10, value_augm_scale=0.10,
                                       affine_trnsfm_prb=1.0, affine_trnsfm_range=0.1)

        # Initialize queue workers for both datasets.
        q_art = multiprocessing.Queue(maxsize=10 * 4)
        q_art_negative = multiprocessing.Queue(maxsize=10 * 4)
        q_content = multiprocessing.Queue(maxsize=10 * 4)
        q_content_negative = multiprocessing.Queue(maxsize=10 * 4)
        jobs = []
        for i in range(5):
            p = multiprocessing.Process(target=content_dataset_places.initialize_batch_worker,
                                        args=(q_content, augmentor, self.batch_size, i))
            p.start()
            jobs.append(p)

            p = multiprocessing.Process(target=content_dataset_places.initialize_batch_worker,
                                        args=(q_content_negative, augmentor, self.batch_size, i+5))
            p.start()
            jobs.append(p)

            p = multiprocessing.Process(target=art_dataset.initialize_batch_worker,
                                        args=(q_art, augmentor, self.batch_size, i))
            p.start()
            jobs.append(p)

            p = multiprocessing.Process(target=art_dataset.initialize_batch_worker,
                                        args=(q_art_negative, augmentor, self.batch_size, i+5))
            p.start()
            jobs.append(p)

        print("Queues are initialized. Processes are started.")
        time.sleep(3)

        # Now initialize the graph
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Graph is initialized.")
        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        if self.options.continue_train:
            self.load(self.checkpoint_dir, ckpt_nmbr)

        # Initial discriminator success rate.
        win_rate = self.options.discr_success_rate
        discr_success = self.options.discr_success_rate
        discr_success = 0.5
        alpha = 0.01

        for step in tqdm(range(self.initial_step, self.options.total_steps+1),
                         initial=self.initial_step,
                         total=self.options.total_steps):

            # Get batch from the queue with batches q, if the last is non-empty.
            while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty() :
                pass
            batch_art = q_art.get()
            batch_art_negative = q_art_negative.get()
            batch_content = q_content.get()
            batch_content_negative = q_content_negative.get()

            # Save batch to classify together with its label
            # scipy.misc.imsave(os.path.join(self.root_dir, self.model_name, 'sample', 'step_%d_%s_%d.jpg' %
            #                                (step, batch_content['label_text'][0].replace('/','_'), int(np.argmax(batch_content['label_onehot'][0])))),
            #                   batch_content['image'][0])
            # print("batch_art['artist_slug_onehot']:", batch_art['artist_slug_onehot'])
            if discr_success >= win_rate:
                # Train generator
                # print("Train generator")
                _, _, gener_acc_ = self.sess.run(
                    fetches=[self.g_optim_step, self.summary_accumulation, self.gener_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_painting_negative: normalize_arr_of_imgs(batch_art_negative['image']),
                        self.input_painting_label: batch_art['artist_slug_onehot'],

                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.input_photo_negative: normalize_arr_of_imgs(batch_content_negative['image']),
                        self.input_photo_label: batch_content['label_onehot'],
                        self.lr: self.options.lr
                    })
                discr_success = discr_success * (1. - alpha) + alpha * (1. - gener_acc_)
            else:
                # Train discriminator.
                _, _, discr_acc_ = self.sess.run(
                    fetches=[self.d_optim_step, self.summary_accumulation, self.discr_acc],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_painting_negative: normalize_arr_of_imgs(batch_art_negative['image']),
                        self.input_painting_label: batch_art['artist_slug_onehot'],
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.input_photo_negative: normalize_arr_of_imgs(batch_content_negative['image']),
                        self.input_photo_label: batch_content['label_onehot'],
                        self.lr: self.options.lr
                    })


                discr_success = discr_success * (1. - alpha) + alpha * discr_acc_


            if step % self.options.save_freq == 0 and step > self.initial_step:
                self.save(step)

            if step % 25000 == 0 and step > self.initial_step:
                self.save(step, is_long=True)

            if step == 0:
                validate(path_to_input_images_folder='/export/home/dkotoven/workspace/Places2_dataset/validation',
                         save_dir=os.path.join(self.root_dir, self.model_name, 'holdout_validation'),
                         save_prefix='validation_images',
                         num_images=100,
                         images_per_group=25,
                         sizes=[768],
                         block_save_size=2048,
                         process_routine=(lambda x: x))

            if (step % 200 == 0) and (step > 0):
                summary = self.sess.run(self.summary_writing)
                self.writer.add_summary(summary, step)
                summary = self.sess.run(self.summary_writing)
                self.writer.add_summary(summary, step)

                # Get batch from the queue with batches q, if the last is non-empty.
                while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty():
                    pass
                batch_art = q_art.get()
                batch_art_negative = q_art_negative.get()
                batch_content = q_content.get()
                batch_content_negative = q_content_negative.get()

                output_paintings_, output_photos_, output_photos_shift_style_, output_photo_negative_ = self.sess.run(
                    fetches=[self.input_painting, self.output_photo, self.output_photo_shift_style, self.output_photo_negative],
                    feed_dict={
                        self.input_painting: normalize_arr_of_imgs(batch_art['image']),
                        self.input_painting_negative: normalize_arr_of_imgs(batch_art_negative['image']),
                        self.input_painting_label: batch_art['artist_slug_onehot'],
                        self.input_photo: normalize_arr_of_imgs(batch_content['image']),
                        self.input_photo_negative: normalize_arr_of_imgs(batch_content_negative['image']),
                        self.lr: self.options.lr
                    })

                save_batch(input_painting_batch=np.concatenate([batch_art['image'],
                                                                batch_art_negative['image']],
                                                               axis=2),
                           input_photo_batch=np.concatenate([batch_content['image'],
                                                             batch_content_negative['image']],
                                                            axis=2),
                           output_painting_batch=denormalize_arr_of_imgs(output_paintings_),
                           output_photo_batch=denormalize_arr_of_imgs(np.concatenate([output_photos_,
                                                                                      output_photos_shift_style_,
                                                                                      output_photo_negative_],
                                                                                     axis=2)
                                                                      ),
                           filepath='%s/step_%d.jpg' % (self.sample_dir, step))

            # Validate on out of train batches for comparison.
            # Disable validation temporarily
            if (step % self.options.save_freq == 0) and False:
                def inference_pipeline(img):
                    img = np.expand_dims(enhance_image(img), axis=0)
                    img_ = self.sess.run(
                        fetches=self.output_photo,
                        feed_dict={
                            self.input_photo: normalize_arr_of_imgs(img),
                            self.input_painting_label: batch_art['artist_slug_onehot']}
                    )
                    return denormalize_arr_of_imgs(img_[0])
                validate(path_to_input_images_folder='/export/home/dkotoven/workspace/Places2_dataset/validation',
                         save_dir=os.path.join(self.root_dir, self.model_name, 'holdout_validation'),
                         save_prefix='step%d' % step,
                         num_images=100,
                         images_per_group=25,
                         sizes=[768, 1280],
                         block_save_size=2048,
                         process_routine=inference_pipeline)


        print("Training is finished. Terminate jobs.")
        for p in jobs:
            p.terminate()
            p.join()
        print("Done.")

    def inference(self, args, path_to_folder, to_save_dir=None, resize_to_original=True,
                  original_size_inference=False,
                  ckpt_nmbr=None):

        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset,
                                                 artists_list=self.options.artists_list)
        augmentor = img_augm.Augmentor(crop_size=[256, 256],#[self.options.image_size, self.options.image_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.10,
                                       saturation_augm_shift=0.10, saturation_augm_scale=0.10,
                                       value_augm_shift=0.10, value_augm_scale=0.10,
                                       affine_trnsfm_prb=1.0, affine_trnsfm_range=0.1)
        # TODO: try to use augmentor defined above

        art_images = []
        art_labels = []
        while len(art_images) < 2 * len(self.options.artists_list):
            art_batch = art_dataset.get_batch(augmentor=augmentor, batch_size=8)
            for img, artist_slug in zip(art_batch['image'], art_batch['artist_slug']):
                if len(art_images) >= 2 * len(self.options.artists_list):
                    break
                else:
                    if len([x for x in art_labels if x == artist_slug]) < 2:
                        art_images.append(img)
                        art_labels.append(artist_slug)

        print("Initialized the art dataset and got the batch of different artworks.")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d_ws%d' %
                                       (self.initial_step, self.image_size, self.options.window_size)
                                       )
        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for img_idx, img_path in enumerate(tqdm(names)):
            img = scipy.misc.imread(img_path, mode='RGB')

            # New version
            # img = scipy.misc.imresize(img, size=2.)
            # img = img[:self.image_size, :self.image_size]

            # Prev version
            img_shape = img.shape[:2]

            # Resize the smallest side of the image to the self.image_size
            if not original_size_inference:
                alpha = float(self.image_size) / float(min(img_shape))
            else:
                # to change
                alpha = max(float(self.options.window_size) / float(min(img_shape)), 1.0)
            #img = scipy.misc.imresize(img, size=alpha)
            img = scipy.misc.imresize(img, size=[int(min(img.shape[0] * alpha, self.image_size * 2.5)),
                                                 int(min(img.shape[1] * alpha, self.image_size * 2.5))])

            img = enhance_image(img)
            img = np.expand_dims(img, axis=0)
            img_name = os.path.basename(img_path)

            # for pair_idx, (img_art1, img_art2) in enumerate(product(art_batch["image"], art_batch["image"])):

            for img_art1_idx, img_art2_idx in product(range(len(art_images)), range(len(art_images))):
                #
                # print("New iteration.")
                # print("art_images.index(img_art1):", art_images.index(img_art1))
                # print("art_images.index(img_art2):", art_images.index(img_art2))

                if img_art1_idx < img_art2_idx:
                    to_save_dir_ = os.path.join(to_save_dir,
                                                "img_art_indices=(%d,%d)" % (img_art1_idx, img_art2_idx))
                    if not os.path.exists(to_save_dir_):
                        os.makedirs(to_save_dir_)

                    img_art1 = art_images[img_art1_idx]
                    img_art2 = art_images[img_art2_idx]

                    img_art1_style_vector = self.sess.run(self.input_painting_features_style,
                                                          feed_dict={
                                                              self.input_painting: normalize_arr_of_imgs(
                                                                  np.expand_dims(img_art1, axis=0))
                                                          })
                    img_art2_style_vector = self.sess.run(self.input_painting_features_style,
                                                          feed_dict={
                                                              self.input_painting: normalize_arr_of_imgs(
                                                                  np.expand_dims(img_art2, axis=0))
                                                          })

                    # scipy.misc.imsave(
                    #     os.path.join(to_save_dir, "img_art1_idx=%d_" % img_art1_idx + img_name[:-4] + 'art1_style_vec=%s.jpg' % str(img_art1_style_vector)),
                    #                   img_art1)
                    # scipy.misc.imsave(
                    #     os.path.join(to_save_dir, "img_art2_idx=%d_" % img_art2_idx + img_name[:-4] + 'art2_style_vec=%s.jpg' % str(img_art2_style_vector)),
                    #     img_art2)

                    scipy.misc.imsave(os.path.join(to_save_dir_,
                                                   img_name[:-4] + '_art1.jpg'),
                                      img_art1)
                    scipy.misc.imsave(os.path.join(to_save_dir_,
                                                   img_name[:-4] + '_art2.jpg'),
                                      img_art2)
                    for alpha in np.arange(0, 1.1, 0.1):
                        style_vector = alpha*img_art1_style_vector + (1.-alpha)*img_art2_style_vector
                        # print("alpha=%.2f, style_vector=%s" % (alpha, str(style_vector)))

                        img_stylized = self.sess.run(
                            self.output_photo,
                            feed_dict={
                                self.input_photo: normalize_arr_of_imgs(img),
                                self.input_painting_features_style: style_vector,
                            })

                        img_stylized = img_stylized[0]
                        # img = denormalize_arr_of_imgs(normalize_arr(img))
                        img_stylized = denormalize_arr_of_imgs(img_stylized)
                        if resize_to_original:
                            img_stylized = scipy.misc.imresize(img_stylized, size=img_shape)
                        else:
                            pass

                        # scipy.misc.imsave(os.path.join(to_save_dir, img_name[:-4] + "_stylized_" +
                        #                                args.artist_slug + '.jpg'), img)
                        # print("Image saved under path %s." % os.path.join(to_save_dir, img_name[:-4] + '.jpg'))
                        scipy.misc.imsave(os.path.join(to_save_dir_,
                                                       img_name[:-4] + '_alpha=%.2f.jpg' % alpha),
                                          img_stylized)

        print("Inference is finished.")


    def inference_interpolation_from_patches(self, args, path_to_folder, to_save_dir=None, resize_to_original=True,
                  original_size_inference=False,
                  ckpt_nmbr=None, num_patches=10, style_patch_sz=256):

        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset,
                                                 artists_list=self.options.artists_list)
        augmentor = img_augm.Augmentor(crop_size=[256, 256],#[self.options.image_size, self.options.image_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.10,
                                       saturation_augm_shift=0.10, saturation_augm_scale=0.10,
                                       value_augm_shift=0.10, value_augm_scale=0.10,
                                       affine_trnsfm_prb=1.0, affine_trnsfm_range=0.1)
        # TODO: try to use augmentor defined above

        art_images = []
        art_labels = []
        # Sample 2 artworks for each artist
        # print(self.options.artists_list)
        for artist_slug in self.options.artists_list:
            for _ in range(2):
                art_batch = art_dataset.get_batch(augmentor=None, batch_size=1)
                # print("artist_slug:", artist_slug)
                # print("art_batch:", art_batch)
                while self.options.artists_list[art_batch['artist_slug'][0]] != artist_slug:
                    art_batch = art_dataset.get_batch(augmentor=None, batch_size=1)
                art_images.append(art_batch['image'][0])
                art_labels.append(self.options.artists_list[art_batch['artist_slug'][0]])
        print("Initialized the art dataset and got the batch of different artworks.")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print("Now save sampled art images into the folder.")
        # Create folder to store results and save picked artworks.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d_ws%d' %
                                       (self.initial_step, self.image_size, self.options.window_size)
                                       )
        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)
        print("to_save_dir:", to_save_dir)

        for n in range(len(self.options.artists_list)):
            img, artist_slug = art_images[2 * n], art_labels[2 * n]
            scipy.misc.imsave(os.path.join(to_save_dir,
                                           artist_slug + '_left.jpg'),
                              img)
            img, artist_slug = art_images[2 * n + 1], art_labels[2 * n + 1]
            scipy.misc.imsave(os.path.join(to_save_dir,
                                           artist_slug + '_right.jpg'),
                              img)

        # Create patches collection for each of the input images from the art_images list.
        art_images_dict_patches = []
        for idx, (img, artist_slug) in enumerate(tqdm(zip(art_images, art_labels))):
            dict_patches = {'patch': [],
                            'patch_style_vector': []}
            for patch_id in range(num_patches):
                row = random.choice(range(img.shape[0]-style_patch_sz))
                col = random.choice(range(img.shape[1]-style_patch_sz))
                patch = img[row:row+style_patch_sz, col:col+style_patch_sz]

                patch_style_vector = self.sess.run(self.input_painting_features_style,
                                                   feed_dict={
                                                       self.input_painting: normalize_arr_of_imgs(
                                                           np.expand_dims(patch, axis=0))
                                                   })
                dict_patches['patch'].append(patch)
                dict_patches['patch_style_vector'].append(patch_style_vector)

            # print("dict_patches['patch_style_vector']:",
            #       dict_patches['patch_style_vector'])
            # print("dict_patches['patch_style_vector'][0].shape:",
            #       dict_patches['patch_style_vector'][0].shape)
            dict_patches['max_style_vector'] = np.max(a=np.concatenate(dict_patches['patch_style_vector'],
                                                                       axis=0),
                                                      axis=0, keepdims=True)
            dict_patches['mean_style_vector'] = np.mean(a=np.concatenate(dict_patches['patch_style_vector'],
                                                                       axis=0),
                                                        axis=0, keepdims=True)
            dict_patches['median_style_vector'] = np.median(a=np.concatenate(dict_patches['patch_style_vector'],
                                                                       axis=0),
                                                            axis=0, keepdims=True)

            save_obj(obj=dict_patches,
                     path_to_file=os.path.join(to_save_dir,
                                               'dict_'+artist_slug+'_%s.pckl' % ('left' if idx % 2 == 0 else 'right'))
                     )
            art_images_dict_patches.append(dict_patches)

        # Now lets iterate over images in folder and stylize them with all possible permutations with each possible
        # style vector: max, mean, median.
        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for img_idx, img_path in enumerate(tqdm(names)):
            img = scipy.misc.imread(img_path, mode='RGB')
            # Resize the smallest side of the image to the self.image_size
            img_shape = img.shape[:2]
            if not original_size_inference:
                alpha = float(self.image_size) / float(min(img_shape))
            else:

                alpha = max(float(self.options.window_size) / float(min(img_shape)), 1.0)
            img = scipy.misc.imresize(img, size=[int(min(img.shape[0] * alpha, self.image_size * 2.5)),
                                                 int(min(img.shape[1] * alpha, self.image_size * 2.5))])

            img = enhance_image(img)
            img = np.expand_dims(img, axis=0)
            img_name = os.path.basename(img_path)

            for (img_art1_idx, img_art2_idx) in \
                [(x,y) for (x,y) in product(range(len(art_images_dict_patches)),
                                            range(len(art_images_dict_patches))) if x<y]:

                to_save_dir_ = os.path.join(to_save_dir,
                                            "art_pair_%s_%s_to_%s_%s" %
                                            (art_labels[img_art1_idx], 'left' if img_art1_idx % 2 == 0 else 'right',
                                             art_labels[img_art2_idx], 'left' if img_art2_idx % 2 == 0 else 'right'))
                if not os.path.exists(to_save_dir_):
                    os.makedirs(to_save_dir_)

                # Stylizations using max, mean or median aggregation of style vectors over patches.
                if False:
                    for mode in ['max', 'mean', 'median']:
                        img_art1_style_vector = art_images_dict_patches[img_art1_idx][mode + '_style_vector']
                        img_art2_style_vector = art_images_dict_patches[img_art2_idx][mode + '_style_vector']

                        for alpha in np.arange(0, 1.1, 0.1):
                            style_vector = alpha * img_art1_style_vector + (1. - alpha) * img_art2_style_vector

                            img_stylized = self.sess.run(
                                self.output_photo,
                                feed_dict={
                                    self.input_photo: normalize_arr_of_imgs(img),
                                    self.input_painting_features_style: style_vector,
                                })

                            img_stylized = img_stylized[0]
                            # img = denormalize_arr_of_imgs(normalize_arr(img))
                            img_stylized = denormalize_arr_of_imgs(img_stylized)
                            if resize_to_original:
                                img_stylized = scipy.misc.imresize(img_stylized, size=img_shape)

                            scipy.misc.imsave(os.path.join(to_save_dir_,
                                                           img_name[:-4] + '_alpha=%.2f_%s_style_vector.jpg' % (
                                                           alpha, mode)
                                                           ),
                                              img_stylized)

                # Now stylize using most distinct style patches for both dictionaries.
                if True:
                    mode = 'discrepancy'
                    img_art1_style_vectors = art_images_dict_patches[img_art1_idx]['patch_style_vector']
                    img_art2_style_vectors = art_images_dict_patches[img_art2_idx]['patch_style_vector']

                    best_pair = (0, 0)
                    largest_discrepancy = 0
                    for idx1, idx2 in product(range(len(img_art1_style_vectors)),
                                              range(len(img_art2_style_vectors))):
                        discrepancy = np.linalg.norm(img_art1_style_vectors[idx1] -
                                                     img_art2_style_vectors[idx2])
                        if discrepancy > largest_discrepancy:
                            best_pair = (idx1, idx2)
                            largest_discrepancy = discrepancy

                    img_art1_style_vector = art_images_dict_patches[img_art1_idx]['patch_style_vector'][
                        best_pair[0]]
                    img_art2_style_vector = art_images_dict_patches[img_art2_idx]['patch_style_vector'][
                        best_pair[1]]

                    for alpha in np.arange(0, 1.1, 0.1):
                        style_vector = alpha * img_art1_style_vector + (1. - alpha) * img_art2_style_vector

                        img_stylized = self.sess.run(
                            self.output_photo,
                            feed_dict={
                                self.input_photo: normalize_arr_of_imgs(img),
                                self.input_painting_features_style: style_vector,
                            })

                        img_stylized = img_stylized[0]
                        # img = denormalize_arr_of_imgs(normalize_arr(img))
                        img_stylized = denormalize_arr_of_imgs(img_stylized)
                        if resize_to_original:
                            img_stylized = scipy.misc.imresize(img_stylized, size=img_shape)

                        scipy.misc.imsave(os.path.join(to_save_dir_,
                                                       img_name[:-4] + '_alpha=%.2f_%s_style_vector.jpg' % (
                                                       alpha, mode)
                                                       ),
                                          img_stylized)

        print("Inference from patches with interpolation is finished.")




    def inference_as_train(self, args, path_to_folder, to_save_dir=None,
                  ckpt_nmbr=None):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d_as_train' % (self.initial_step, self.image_size))

        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for img_idx, img_path in enumerate(tqdm(names)):
            img = scipy.misc.imread(img_path, mode='RGB')

            img = scipy.misc.imresize(img, size=2.)
            img_shape = img.shape

            if max(img_shape) > 1800.:
                img = scipy.misc.imresize(img, size=1800. / max(img_shape))
            if max(img_shape) < 800:
                # Resize the smallest side of the img to 800px
                alpha = 800. / float(min(img_shape))
                if alpha < 4.:
                    img = scipy.misc.imresize(img, size=alpha)
                    img = np.expand_dims(img, axis=0)
                else:
                    img = scipy.misc.imresize(img, size=[800, 800])

            img_patch = img[256-128:512-128, 256-128:512-128]
            img = enhance_image(img_patch)
            img = np.expand_dims(img, axis=0)

            img = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img),
                })

            img = img[0]
            # img = denormalize_arr_of_imgs(normalize_arr(img))
            img = denormalize_arr_of_imgs(img)

            img_name = os.path.basename(img_path)
            scipy.misc.imsave(os.path.join(to_save_dir, img_name[:-4] + "_stylized_" +
                                           args.artist_slug + '.jpg'),
                              np.concatenate([img_patch, img], axis=1)
                              )

        print("Inference is finished.")

    def inference_diff_scales(self, args, path_to_folder, to_save_dir=None,
                  ckpt_nmbr=None):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # Create folder to store results.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d_diff_scales' % (self.initial_step, self.image_size))

        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        names = []
        for d in path_to_folder:
            names += glob(os.path.join(d, '*'))
        names = [x for x in names if os.path.basename(x)[0] != '.']
        names.sort()
        for img_idx, img_path in enumerate(tqdm(names)):
            img = scipy.misc.imread(img_path, mode='RGB')

            img = scipy.misc.imresize(img, size=2.)


            img_original = scipy.misc.imresize(img, size=(1280, 1280))


            img = enhance_image(img_original)
            img = np.expand_dims(img, axis=0)

            img_full_stylized = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img),
                })
            print("img_full_stylized.shape:", img_full_stylized.shape)
            img_patch_stylized = self.sess.run(
                self.output_photo,
                feed_dict={
                    self.input_photo: normalize_arr_of_imgs(img[:, 256:512+512, 256:512+512]),
                })
            print("img_patch_stylized.shape:", img_patch_stylized.shape)

            img_name = os.path.basename(img_path)
            scipy.misc.imsave(os.path.join(to_save_dir, img_name[:-4] + "_stylized_" +
                                           args.artist_slug + '.jpg'),
                              np.concatenate([img_original[256:512+512, 256:512+512],
                                              denormalize_arr_of_imgs(normalize_arr(img_full_stylized[0]))[256:512+256, 256:512+256],
                                              denormalize_arr_of_imgs(normalize_arr(img_patch_stylized[0]))],
                                             axis=1)
                              )

        print("Inference is finished.")

    def extract_style_feats(self, args,
                            to_save_dir=None,
                            ckpt_nmbr=None,
                            style_patch_sz=768):
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name)

        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset,
                                                 artists_list=self.options.artists_list)
        augmentor = img_augm.Augmentor(crop_size=[style_patch_sz, style_patch_sz],#[self.options.image_size, self.options.image_size],
                                       vertical_flip_prb=0.,
                                       hsv_augm_prb=1.0,
                                       hue_augm_shift=0.10,
                                       saturation_augm_shift=0.10, saturation_augm_scale=0.10,
                                       value_augm_shift=0.10, value_augm_scale=0.10,
                                       affine_trnsfm_prb=1.0, affine_trnsfm_range=0.1)
        # TODO: try to use augmentor defined above





        print("Initialized the art dataset.")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print("Start inference.")

        if args.continue_train and self.load(self.checkpoint_dir, ckpt_nmbr):
            print(" [*] Load SUCCESS")
        else:
            if args.continue_train and self.load(self.checkpoint_long_dir, ckpt_nmbr):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        print("Now save sampled art images into the folder.")
        # Create folder to store results and save picked artworks.
        if to_save_dir is None:
            to_save_dir = os.path.join(self.root_dir, self.model_name,
                                       'inference_ckpt%d_sz%d_ws%d' %
                                       (self.initial_step, self.image_size, self.options.window_size)
                                       )
        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        results = []
        for _ in tqdm(range(100)):
            art_batch = art_dataset.get_batch(augmentor=augmentor, batch_size=1)
            # art_images.append(art_batch['image'][0])
            # art_labels.append(self.options.artists_list[art_batch['artist_slug'][0]])


            patch_style_vector = self.sess.run(self.input_painting_features_style,
                                               feed_dict={
                                                   self.input_painting: normalize_arr_of_imgs(
                                                       np.expand_dims(art_batch['image'][0], axis=0))
                                               })
            results.append({
                'image' : art_batch['image'][0],
                'artist_slug' : self.options.artists_list[art_batch['artist_slug'][0]],
                'style_vector' : patch_style_vector
            })

        save_obj(obj=results,
                 path_to_file=os.path.join(to_save_dir, 'style_features_list_augmentor_100.pckl'))



        print("All the style vectors are computed.")


    def save(self, step, is_long=False):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if is_long:
            self.saver_long.save(self.sess,
                                 os.path.join(self.checkpoint_long_dir, self.model_name+'_%d.ckpt' % step),
                                 global_step=step)
        else:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_dir, self.model_name + '_%d.ckpt' % step),
                            global_step=step)

    def load(self, checkpoint_dir, ckpt_nmbr=None):
        if ckpt_nmbr:
            if len([x for x in os.listdir(checkpoint_dir) if ('-'+str(ckpt_nmbr)) in x]) > 0:
                print(" [*] Reading checkpoint %d from folder %s." % (ckpt_nmbr, checkpoint_dir))
                ckpt_name = [x for x in os.listdir(checkpoint_dir) if ('-'+str(ckpt_nmbr)) in x][0]
                # if ckpt_name.endswith('.index'):
                #     ckpt_name = ckpt_name[:-6]
                ckpt_name = '.'.join(ckpt_name.split('.')[:-1])
                self.initial_step = ckpt_nmbr
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
        else:
            print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
                print("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                return True
            else:
                return False
