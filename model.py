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
                                      os.listdir(args.path_to_art_dataset),
                                      args.path_to_art_dataset,
                                      args.path_to_content_dataset,
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
        self.saver = tf.train.Saver(max_to_keep=2)
        self.saver_long = tf.train.Saver(max_to_keep=None)
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

            self.output_photo_shift_style = decoder(features=self.input_photo_features,
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
            # Image loss. Minimizes pixel difference between blurred input and blurred output photo.

            self.img_loss = mse_criterion(blur_img(input_tensor=self.input_photo,
                                                   kernel_size=self.options.blurring_kernel_size),
                                          blur_img(input_tensor=self.output_photo,
                                                   kernel_size=self.options.blurring_kernel_size))

            # Fixpoint loss.
            # FP-content loss
            self.fixpoint_loss_photo = tf.reduce_mean(cosine_loss(
                a=tf.reshape(self.output_photo_features, shape=(self.batch_size, -1)),
                b=tf.reshape(self.input_photo_features, shape=(self.batch_size, -1))))

            # FPT-style loss
            self.fixpoint_loss_painting = \
                0.5 * tf.reduce_mean(triplet_loss(
                    anchor=tf.reshape(self.input_painting_features_style, shape=(self.batch_size, -1)),
                    positive=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                    negative=tf.reshape(self.output_photo_shift_style_features_style, shape=(self.batch_size, -1)),
                    margin=self.options.margin)) + \
                0.5 * tf.reduce_mean(triplet_loss(
                    anchor=tf.reshape(self.input_painting_negative_features_style, shape=(self.batch_size, -1)),
                    positive=tf.reshape(self.output_photo_shift_style_features_style, shape=(self.batch_size, -1)),
                    negative=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                    margin=self.options.margin))

            # Disentanglement loss
            # FPD loss
            self.disentanglement_loss = tf.reduce_mean(triplet_loss(
                anchor=tf.reshape(self.output_photo_features_style, shape=(self.batch_size, -1)),
                positive=tf.reshape(self.output_photo_negative_features_style, shape=(self.batch_size, -1)),
                negative=tf.reshape(self.input_painting_features_style, shape=(self.batch_size, -1)),
                margin=0.))

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
                self.d_optim_step = tf.train.AdamOptimizer(self.lr, beta1=0.8).minimize(
                    loss=self.options.loss_weights['discr'] * self.discr_loss +
                         self.options.loss_weights['clsf_cont'] * self.input_photo_clsf_cont_loss +
                         self.options.loss_weights['clsf_style'] * self.input_painting_clsf_style_loss +
                         self.options.loss_weights['content_preservation'] * self.content_preservation_loss,
                    var_list=[self.discr_vars])
                self.g_optim_step = tf.train.AdamOptimizer(self.lr, beta1=0.8).minimize(
                    loss=self.options.loss_weights['discr'] * self.gener_loss +
                         self.options.loss_weights['image'] * self.img_loss +
                         self.options.loss_weights['fp_content'] * self.fixpoint_loss_photo +
                         self.options.loss_weights['fp_style'] * self.fixpoint_loss_painting +
                         self.options.loss_weights['fp_style'] * self.disentanglement_loss +
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

        with tf.control_dependencies(list(discriminator_summary_vals.values())+list(generator_summary_vals.values())):
            self.summary_accumulation = tf.group(ema.apply(list(discriminator_summary_vals.values()) +
                                                           list(generator_summary_vals.values())))

        moving_average = [ema.average(x) for x in list(discriminator_summary_vals.values()) + list(generator_summary_vals.values())]

        self.summary_writing = tf.summary.merge([
            tf.summary.scalar(name, val) for name, val in zip(list(discriminator_summary_vals.keys()) +
                                                              list(generator_summary_vals.keys()),
                                                              moving_average)])


        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

    def train(self, args, ckpt_nmbr=None):
        # Initialize datasets and augmentor.
        content_dataset_places = prepare_dataset.PlacesDataset(path_to_dataset=self.options.path_to_content_dataset)

        art_dataset = prepare_dataset.ArtDataset(path_to_art_dataset=self.options.path_to_art_dataset)
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
        for i in range(2):
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

        discr_success = self.options.discr_success_rate
        alpha = 0.01

        for step in tqdm(range(self.initial_step, self.options.total_steps+1),
                         initial=self.initial_step,
                         total=self.options.total_steps):

            if self.options.discr_success_rate < 0.:
                # Perform both Generator and Discriminator update steps

                # Train generator
                # Get batch from the queue if non-empty.
                while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty():
                    pass
                batch_art = q_art.get()
                batch_art_negative = q_art_negative.get()
                batch_content = q_content.get()
                batch_content_negative = q_content_negative.get()
                # Update
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

                # Train discriminator.
                # Get batch from the queue if non-empty.
                while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty():
                    pass
                batch_art = q_art.get()
                batch_art_negative = q_art_negative.get()
                batch_content = q_content.get()
                batch_content_negative = q_content_negative.get()
                # Update
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

                discr_success = discr_success * (1. - alpha) + alpha * (1. - discr_acc_)
            else:
                # Update only one agent based on its performance
                if discr_success >= self.options.discr_success_rate:
                    # Train generator
                    # Get batch from the queue if non-empty.
                    while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty():
                        pass
                    batch_art = q_art.get()
                    batch_art_negative = q_art_negative.get()
                    batch_content = q_content.get()
                    batch_content_negative = q_content_negative.get()
                    # Update
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
                    # Get batch from the queue if non-empty.
                    while q_art.empty() or q_art_negative.empty() or q_content.empty() or q_content_negative.empty():
                        pass
                    batch_art = q_art.get()
                    batch_art_negative = q_art_negative.get()
                    batch_content = q_content.get()
                    batch_content_negative = q_content_negative.get()
                    # Update
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
