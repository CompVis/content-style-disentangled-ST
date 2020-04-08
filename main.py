import argparse
import os
import tensorflow as tf
tf.set_random_seed(228)
from model import Artgan

def parse_list(str_value):
    if ',' in str_value:
        str_value = str_value.split(',')
    else:
        str_value = [str_value]
    return str_value


parser = argparse.ArgumentParser(description='')

################################################# Model Configuration ##################################################
parser.add_argument('--model_name',
                    dest='model_name',
                    default='model1',
                    help='name of the model')

parser.add_argument('--total_steps',
                    dest='total_steps',
                    type=int,
                    default=int(1e6),
                    help='total # of steps')
parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    default=1,
                    help='# images in batch')
parser.add_argument('--image_size',
                    dest='image_size',
                    type=int,
                    default=256*3,
                    help='extract crops of this size')
parser.add_argument('--window_size',
                    dest='window_size',
                    type=int,
                    default=256,
                    help='define size of the smoothing window for local feature normalization layer')
parser.add_argument('--style_dim',
                    dest='style_dim',
                    type=int,
                    default=16,
                    help='Size of the style vector.')
parser.add_argument('--margin',
                    dest='margin',
                    type=float,
                    default=0.2,
                    help='Margin for triplet loss')

parser.add_argument('--ngf',
                    dest='ngf',
                    type=int,
                    default=32,
                    help='# of generator(encoder-decoder) filters in first conv layer')
parser.add_argument('--ndf',
                    dest='ndf',
                    type=int,
                    default=64,
                    help='# of discriminator filters in first conv layer')
parser.add_argument('--ncf',
                    dest='ncf',
                    type=int,
                    default=1,
                    help='# of classifier filters in first conv layer')

parser.add_argument('--input_nc',
                    dest='input_nc',
                    type=int,
                    default=3,
                    help='# of input image channels')
parser.add_argument('--output_nc',
                    dest='output_nc',
                    type=int,
                    default=3,
                    help='# of output image channels')

################################################ Training Configuration ################################################
parser.add_argument('--ptad',
                    dest='path_to_art_dataset',
                    type=str,
                    default='./data/folder_with_diff_styles',
                    help='Directory contains folders each representing different style.')
parser.add_argument('--ptcd',
                    dest='path_to_content_dataset',
                    type=str,
                    default='/path/to/Places2_dataset/data_large',
                    help='Path to Places365 training dataset.')

parser.add_argument('--lr',
                    dest='lr',
                    type=float,
                    default=0.0002,
                    help='initial learning rate for adam')
parser.add_argument('--phase',
                    dest='phase',
                    default='train',
                    help='train, test')

parser.add_argument('--save_freq',
                    dest='save_freq',
                    type=int,
                    default=10000,
                    help='save a model every save_freq steps')
parser.add_argument('--continue_train',
                    dest='continue_train',
                    type=bool,
                    default=False,
                    help='if continue training, load the latest model: 1: true, 0: false')

parser.add_argument('--generator_steps',
                    dest='generator_steps',
                    type=int,
                    default=1,
                    help='Number of generator steps.')
parser.add_argument('--discriminator_steps',
                    dest='discriminator_steps',
                    type=int,
                    default=1,
                    help='Number of discriminator steps.')

parser.add_argument('--bks',
                    dest='blurring_kernel_size',
                    type=int,
                    default=10,
                    help='Size of the average pooling kernel we use to generate blurred images.'
                         'We use blurring to compare the original content image with the generated '
                         'ones after blurring.')
parser.add_argument('--dsr',
                    dest='discr_success_rate',
                    type=float,
                    default=0.8,
                    help='Rate of trials that discriminator will win on average.')

##################################################### Loss weights #####################################################
parser.add_argument('--dlw',
                    dest='discr_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of discriminator loss.')
parser.add_argument('--cclw',
                    dest='clsf_cont_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of content classifier loss.')
parser.add_argument('--cslw',
                    dest='clsf_style_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of style classifier loss.')
parser.add_argument('--ilw',
                    dest='image_loss_weight',
                    type=float,
                    default=200.,
                    help='Weight of image loss.')
parser.add_argument('--cfplw',
                    dest='cont_fp_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of content feature fixpoint loss.')
parser.add_argument('--sfplw',
                    dest='style_fp_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of style feature fixpoint loss.')
parser.add_argument('--cplw',
                    dest='content_preservation_loss_weight',
                    type=float,
                    default=0.,
                    help='Weight of content preservation loss. L2 distance between encoder representations of '
                         'the input photo and transfromed photo.')
parser.add_argument('--tvlw',
                    dest='tv_loss_weight',
                    type=float,
                    default=0.,
                    help='Weight of total variation loss.')


############################################### Inference configuration ################################################

parser.add_argument('--ii_dir',
                    dest='inference_images_dir',
                    type=parse_list,
                    default=['/export/home/asanakoy/workspace/gan_style/data/mscoco_val_200'],
                    help='Directory with images we want to process.')

parser.add_argument('--save_dir',
                    type=str,
                    default=None,
                    help='Directory to save inference output images.')
parser.add_argument('--ckpt_nmbr', dest='ckpt_nmbr', type=int, default=None, help='Checkpoint number we want to use '
                                                                                  'for inference. Might be None(unspecified).')
parser.add_argument('--original_size_inference', dest='original_size_inference', type=int, default=False,
                    help='If True will inference in original image size')

args = parser.parse_args()


def main(_):

    tfconfig = tf.ConfigProto(allow_soft_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Artgan(sess, args)

        if args.phase == 'train':
            print('Start training.')
            model.train(args, ckpt_nmbr=args.ckpt_nmbr)

        sess.close()

if __name__ == '__main__':
    tf.app.run()
