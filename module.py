from __future__ import division
from ops import *
import tensorflow.contrib.layers

PADDING_TYPE = "REFLECT"


def encoder(image, kernels, options, is_training=True, reuse=True, name="encoder"):
    """
    Args:
        image: input tensor, must have
        kernels: number of kernels in conv layers
        is_training: boolean parameter  
        reuse: to create new encoder or use existing
        name: name of the encoder

    Returns: Encoded image.
    """

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], PADDING_TYPE)
        c1 = tf.nn.relu(local_group_norm(input=conv2d(c0, kernels, 3, 1, padding='VALID', name='g_e1_c'),
                                         name='g_e1_bn',
                                         window_size=options.window_size // 2))
        c2 = tf.nn.relu(local_group_norm(input=conv2d(c1, kernels, 3, 2, padding='VALID', name='g_e2_c'),
                                         name='g_e2_bn',
                                         window_size=options.window_size // 4))
        c3 = tf.nn.relu(local_group_norm(conv2d(c2, kernels * 2, 3, 2, padding='VALID', name='g_e3_c'),
                                         name='g_e3_bn'))
        c4 = tf.nn.relu(local_group_norm(conv2d(c3, kernels * 4, 3, 2, padding='VALID', name='g_e4_c'),
                                         name='g_e4_bn',
                                         window_size=options.window_size // 8))
        c5 = tf.nn.relu(local_group_norm(conv2d(c4, kernels * 8, 3, 2, padding='VALID', name='g_e5_c'),
                                         name='g_e5_bn',
                                         window_size=options.window_size // 16))
        return c5


def encoder_style(image, kernels, options, is_training=True, reuse=True, name="encoder_style"):
    """
    Args:
        image: input tensor, must have
        kernels: number of kernels in conv layers
        is_training: boolean parameter
        reuse: to create new encoder or use existing
        name: name of the encoder

    Returns: Encoded image.
    """

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        win_size = 128

        B, H, W = tf.shape(image)[0], tf.minimum(tf.shape(image)[1], 256), tf.minimum(tf.shape(image)[2], 256)

        image = tf.random_crop(image, size=[B, H, W, 3])
        image = tf.reshape(image, shape=[B, H, W, 3])


        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], PADDING_TYPE)
        c1 = tf.nn.relu(local_group_norm(input=conv2d(c0, kernels, 3, 1, padding='VALID', name='g_e1_c'),
                                         name='g_e1_bn',
                                         window_size=win_size // 2))
        c2 = tf.nn.relu(local_group_norm(input=conv2d(c1, kernels, 3, 2, padding='VALID', name='g_e2_c'),
                                         name='g_e2_bn',
                                         window_size=win_size // 4))
        c3 = tf.nn.relu(local_group_norm(conv2d(c2, kernels * 2, 3, 2, padding='VALID', name='g_e3_c'),
                                         name='g_e3_bn'))
        c4 = tf.nn.relu(local_group_norm(conv2d(c3, kernels * 4, 3, 2, padding='VALID', name='g_e4_c'),
                                         name='g_e4_bn',
                                         window_size=win_size // 8))
        c5 = tf.nn.relu(local_group_norm(conv2d(c4, kernels * 8, 3, 2, padding='VALID', name='g_e5_c'),
                                         name='g_e5_bn',
                                         window_size=win_size // 16))

        f1 = tf.reduce_mean(input_tensor=c5,
                            axis=[1, 2])
        f1 = tf.nn.relu(f1)
        f1 = tf.layers.dropout(inputs=f1,
                               rate=0.5,
                               training=is_training)
        f1 = tensorflow.layers.dense(inputs=f1,
                                     units=kernels * 8,
                                     activation=tf.nn.relu,
                                     name='Es_f1')
        f1 = tf.layers.dropout(inputs=f1,
                               rate=0.5,
                               training=is_training)
        f2 = tensorflow.layers.dense(inputs=f1,
                                     units=kernels * 8,
                                     activation=tf.nn.relu,
                                     name='Es_f2')
        f2 = tf.layers.dropout(inputs=f2,
                               rate=0.5,
                               training=is_training)
        f3 = tensorflow.layers.dense(inputs=f2,
                                     units=options.style_dim,
                                     activation=None,
                                     name='Es_f3')

        return tf.nn.l2_normalize(f3, 1)


def decoder(features, kernels, options, style=None, is_training=True, reuse=True, name="decoder"):
    """
    Args:
        features: input tensor, must have
        kernels: number of kernels in conv layers
        is_training: boolean parameter
        reuse: to create new decoder or use existing
        name: name of the encoder

    Returns: Decoded image.
    """

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # print("decoder features", features)
        def residule_block(x, dim, ks=3, s=1, name='res'):
            y = conv2d(x, dim, ks, s, padding='VALID', name=name + '_c1')
            y = local_group_norm(y,
                                 style=style,
                                 name=name + '_normalization1',
                                 window_size=options.window_size // 32)
            y = lrelu(y)
            y = tf.layers.conv2d_transpose(y, dim, ks, s, name=name + '_c2')
            y = local_group_norm(y,
                                 style=style,
                                 name=name + '_normalization2',
                                 window_size=options.window_size // 32)
            y = lrelu(y)
            return y + x

        # define G network with 9 resnet blocks
        num_kernels = features.get_shape().as_list()[-1]
        r1 = residule_block(features, num_kernels, name='g_r1')
        # print("decoder r1", r1)
        r2 = residule_block(r1, num_kernels, name='g_r2')
        r3 = residule_block(r2, num_kernels, name='g_r3')
        r4 = residule_block(r3, num_kernels, name='g_r4')
        r5 = residule_block(r4, num_kernels, name='g_r5')
        r6 = residule_block(r5, num_kernels, name='g_r6')
        r7 = residule_block(r6, num_kernels, name='g_r7')
        r8 = residule_block(r7, num_kernels, name='g_r8')
        r9 = residule_block(r8, num_kernels, name='g_r9')

        d1 = deconv2d(r9, kernels * 8, 3, 2, name='g_d1_dc')
        # print("decoder d1", d1)
        d1 = tf.nn.relu(local_group_norm(input=d1,
                                         style=style,
                                         name='g_d1_bn',
                                         window_size=options.window_size // 16))
        # print("decoder d1 after local_group_norm", d1)
        d2 = deconv2d(d1, kernels * 4, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(local_group_norm(input=d2,
                                         style=style,
                                         name='g_d2_bn',
                                         window_size=options.window_size // 8))

        d3 = deconv2d(d2, kernels * 2, 3, 2, name='g_d3_dc')
        d3 = tf.nn.relu(local_group_norm(input=d3,
                                         style=style,
                                         name='g_d3_bn',
                                         window_size=options.window_size // 4))

        d4 = deconv2d(d3, kernels, 3, 2, name='g_d4_dc')
        d4 = tf.nn.relu(local_group_norm(input=d4,
                                         style=style,
                                         name='g_d4_bn',
                                         window_size=options.window_size // 2))

        d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c')) * 2. - 1.
        return pred


def discriminator(image, num_clsf_clss, kernels, options, is_training=True, reuse=True, name="discriminator"):
    """
    Discriminator agent, that provides us with information about image plausibility at
    different scales.
    Args:
        image: input tensor
        kernels: number of kernels in conv layers
        is_training: boolean parameter
        num_clsf_clss: number of classifier classes
        reuse: to create new discriminator or use existing
        name: name of the discriminator

    Returns:
        dictionary with two keys: discriminator and classifier.
    """
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(local_group_norm(conv2d(image, kernels * 2, ks=5, name='d_h0_conv'),
                                    name='d_bn0',
                                    window_size=options.window_size // 2))
        # h0 is (128 x 128 x self.df_dim)
        h0_pred = conv2d(h0, 1, ks=5, s=1, name='d_h0_pred', activation_fn=None)
        h0 = tf.concat(values=[h0, h0_pred], axis=-1)

        h1 = lrelu(local_group_norm(conv2d(h0, kernels * 2, ks=5, name='d_h1_conv'),
                                    name='d_bn1',
                                    window_size=options.window_size // 4))
        # h1 is (64 x 64 x self.df_dim*2)
        h1_pred = conv2d(h1, 1, ks=10, s=1, name='d_h1_pred', activation_fn=None)
        h1 = tf.concat(values=[h1, h1_pred], axis=-1)

        h2 = lrelu(local_group_norm(conv2d(h1, kernels * 4, ks=5, name='d_h2_conv'),
                                    name='d_bn2',
                                    window_size=options.window_size // 8))
        # h2 is (32x 32 x self.df_dim*4)
        h2_pred = conv2d(h2, 1, ks=3, s=1, name='d_h2_pred')
        h2 = tf.concat(values=[h2, h2_pred], axis=-1)

        h3 = lrelu(local_group_norm(conv2d(h2, kernels * 8, ks=5, name='d_h3_conv'),
                                    name='d_bn3',
                                    window_size=options.window_size // 16))
        # h3 is (16 x 16 x self.df_dim*8)
        h3_pred = conv2d(h3, 1, ks=10, s=1, name='d_h3_pred', activation_fn=None)
        h3 = tf.concat(values=[h3, h3_pred], axis=-1)

        h4 = lrelu(local_group_norm(conv2d(h3, kernels * 8, ks=5, name='d_h4_conv'),
                                    name='d_bn4',
                                    window_size=options.window_size // 32))
        # h4 is (8 x 8 x self.df_dim*16)
        h4_pred = conv2d(h4, 1, ks=3, s=1, name='d_h4_pred')
        h4 = tf.concat(values=[h4, h4_pred], axis=-1)

        h5 = lrelu(local_group_norm(conv2d(h4, kernels * 16, ks=5, name='d_h5_conv'),
                                    name='d_bn5',
                                    window_size=options.window_size // 64))
        # h5 is (4 x 4 x self.df_dim*16)
        h5_pred = conv2d(h5, 1, ks=6, s=1, name='d_h5_pred', activation_fn=None)
        h5 = tf.concat(values=[h5, h5_pred], axis=-1)

        h6 = lrelu(group_norm(conv2d(h5, kernels * 16, ks=5, name='d_h6_conv'),
                              name='d_bn6'))
        # h6 is (2 x 2 x self.df_dim*16)
        h6_pred = conv2d(h6, 1, ks=3, s=1, name='d_h6_pred', activation_fn=None)

        # Concatenate all the prediction vectors across channels
        h_pred = tf.concat(values=[h0_pred,
                                   tf.image.resize_images(images=h1_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR),
                                   tf.image.resize_images(images=h2_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR),
                                   tf.image.resize_images(images=h3_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR),
                                   tf.image.resize_images(images=h4_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR),
                                   tf.image.resize_images(images=h5_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR),
                                   tf.image.resize_images(images=h6_pred,
                                                          size=tf.shape(h0_pred)[1:3],
                                                          method=tf.image.ResizeMethod.BILINEAR)],
                           axis=-1
                           )

        # Content classification
        h_pred_cont = conv2d(input_=h_pred,
                             ks=5,
                             s=1,
                             output_dim=num_clsf_clss,
                             activation_fn=None,
                             name='d_pred_cont_conv1',
                             )
        h_pred_cont = tf.reduce_max(input_tensor=h_pred_cont,
                                    axis=[1, 2],
                                    name='h_pred_cont_max_pool')
        h_pred_cont = tensorflow.layers.dense(h_pred_cont,
                                              units=kernels * 4,
                                              activation=tf.nn.relu,
                                              name='d_h_pred_cont_fc1')
        h_pred_cont = tensorflow.layers.dense(h_pred_cont,
                                              units=kernels * 4,
                                              activation=tf.nn.relu,
                                              name='d_h_pred_cont_fc2')
        h_pred_cont = tensorflow.layers.dense(h_pred_cont,
                                              units=num_clsf_clss,
                                              activation=None,
                                              name='d_h_pred_cont_fc3')

        # Style classification
        h_pred_style = conv2d(input_=h_pred,
                              ks=5,
                              s=1,
                              output_dim=len(options.artists_list),
                              activation_fn=None,
                              name='d_pred_style',
                              )
        # Adversarial loss
        h_pred_discr = conv2d(input_=h_pred,
                              ks=5,
                              s=1,
                              output_dim=1,
                              activation_fn=None,
                              name='d_pred_discr',
                              )

        return {"scale6": h6,
                "content_pred": h_pred_cont,
                "style_pred": h_pred_style,
                "discr_pred": h_pred_discr}


# ====== Define different types of losses applied to discriminator's output. ====== #

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mse_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    eps = 1e-6
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits+eps, labels=labels))
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits+eps, labels=labels))

def cosine_similarity(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b), axis=1)
    return cos_similarity

def cosine_loss(a, b):
    return tf.square(1. - cosine_similarity(a, b))

def triplet_loss(anchor, positive, negative, margin=0.2):

    d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

    loss = tf.maximum(0.0, margin + d_pos - d_neg)
    return loss

def reduce_spatial_dim(input_tensor):
    """
    Since labels and discriminator outputs are of different shapes (and even ranks)
    we should write a routine to deal with that. 
    Args:
        input: tensor of shape [batch_size, spatial_resol_1, spatial_resol_2, depth]
    Returns:
        tensor of shape [batch_size, depth]
    """
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    return input_tensor


def add_spatial_dim(input_tensor, dims_list, resol_list):
    """
        Appends dimensions mentioned in dims_list resol_list times. S
        Args:
            input: tensor of shape [batch_size, depth0]
            dims_list: list of integers with position of new  dimensions to append.
            resol_list: list of integers with corresponding new dimensionalities for each dimension.
        Returns:
            tensor of new shape
        """
    for dim, res in zip(dims_list, resol_list):
        input_tensor = tf.expand_dims(input=input_tensor, axis=dim)
        input_tensor = tf.concat(values=[input_tensor] * res, axis=dim)
    return input_tensor


def repeat_scalar(input_tensor, shape):
    """
    Repeat scalar values.
    :param input_tensor: tensor of shape [batch_size, 1]
    :param shape: new_shape of the element of the tensor
    :return: tensor of the shape [batch_size, *shape] with elements repeated.
    """
    with tf.control_dependencies([tf.assert_equal(tf.shape(input_tensor)[1], 1)]):
        batch_size = tf.shape(input_tensor)[0]
    input_tensor = tf.tile(input_tensor, tf.stack(values=[1, tf.reduce_prod(shape)], axis=0))
    input_tensor = tf.reshape(input_tensor, tf.concat(values=[[batch_size], shape, [1]], axis=0))
    return input_tensor


def blur_img(input_tensor, kernel_size=30):
    return slim.avg_pool2d(inputs=input_tensor, kernel_size=kernel_size, stride=1, padding='VALID')


