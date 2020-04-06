from __future__ import division
import math
import pprint
import scipy.misc
from scipy.ndimage.filters import gaussian_filter

import numpy as np
from ops import *
import os
from skimage import exposure
import pickle


def get_one_hot_encoded_vector(l, i):
    v = np.zeros(shape=l, dtype=np.float32)
    v[i] = 1.
    return v


def get_batch(data, augmentor, artists_list, techniques_list, batch_size=1, is_panting=True):
    """
    Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
     also containing artist name, technique name, and period of creation for given artist.
     In case of content images we have only the 'path' column.
    Args:
        data: dataframe with columns ['path', 'artist_slug', 'technique', 'period']
        augmentor: Augmentor object responsible for augmentation pipeline
        artists_list: list of names of artists
        techniques_list: list of names of techniques
        batch_size: size of batch
        is_panting: describes whether the data dataframe is for paintings or for content images

    Returns:
        4 batches for batch of paintings
        1 batch - otherwise
    """

    batch_image = []
    batch_technique_label = []
    batch_artist_label = []
    batch_period_label = []

    for _ in range(batch_size):
        row = data.sample(n=1)
        image = scipy.misc.imread(name=row['path'].values[0], mode='RGB')
        if not is_panting:
            image = scipy.misc.imresize(image, size=2.)
        else:
            image_shape = image.shape
            # Previous rescaling version.
            if False:
                if max(image_shape) > 1800.:
                    image = scipy.misc.imresize(image, size=1800./max(image_shape))
                if max(image_shape) < 800:
                    image = scipy.misc.imresize(image, size=[800, 800])
            # New rescaling version.
            else:
                if max(image_shape) > 1800.:
                    image = scipy.misc.imresize(image, size=1800./max(image_shape))
                if max(image_shape) < 800:
                    # Resize the smallest side of the image to 800px
                    alpha = 800. / float(min(image_shape))
                    if alpha < 4.:
                        image = scipy.misc.imresize(image, size=alpha)
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = scipy.misc.imresize(image, size=[800, 800])

        batch_image.append(augmentor(image).astype(np.float32))
        if is_panting:
            batch_artist_label.append(get_one_hot_encoded_vector(l=len(artists_list),
                                                                 i=artists_list.index(row['artist_slug'].values[0])))
            batch_technique_label.append(get_one_hot_encoded_vector(l=len(techniques_list),
                                                                    i=techniques_list.index(row['technique'].values[0])))
            batch_period_label.append(row['date'].values[0])

    # Now return a batch in correct form

    if is_panting:
        batch_image = np.asarray(batch_image)
        batch_technique_label = np.asarray(batch_technique_label)
        batch_artist_label = np.asarray(batch_artist_label)
        # Will turn the array into 2-D array of shape [batch_size, 1].
        batch_period_label = np.atleast_2d(np.asarray(batch_period_label))
        return {"image": batch_image,
                "artist": batch_artist_label,
                "technique": batch_technique_label,
                "period": batch_period_label.T}
    return {"image": np.asarray(batch_image)}


def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save

    Returns:

    """
    def batch_to_img(batch):
        return np.reshape(batch,
                          newshape=(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3]))

    to_save = np.concatenate([batch_to_img(input_painting_batch),
                              batch_to_img(input_photo_batch),
                              batch_to_img(output_photo_batch)],
                             axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)

    scipy.misc.imsave(filepath, arr=to_save)


def normalize_arr(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    arr = (arr - np.mean(arr))/np.std(arr)
    arr = np.tanh(arr)

    return arr


def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr/127.5 - 1.
    # return (arr - np.mean(arr)) / np.std(arr)


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5


def group_images_into_square_grid(images_list):
    """
    Auxiliary function responsible for transforming list of images
    into single grid like image having almost square form.
    Args:
        images_list: list of images. They may have different size but dimensionality must coincide.
    Returns:
        single numpy array of concatenated images from images_list
    """
    size = (images_list[0].shape[0], images_list[0].shape[0])
    rows = int(np.floor(np.sqrt(len(images_list))))
    cols = int(np.ceil(float(len(images_list)) / rows))

    images_list = [scipy.misc.imresize(x, size=size) for x in images_list]
    images_list += [np.zeros_like(images_list[0], dtype=np.uint8)] * (rows * cols - len(images_list))

    left_idx = 0
    #     print("Number of images in the list:", len(images_list))
    #     print("Number of rows: %d, cols: %d" % (rows, cols))

    result = np.concatenate([np.concatenate(images_list[row * cols:(row + 1) * cols], axis=1)
                             for row in range(rows)], axis=0)
    return result


def validate(path_to_input_images_folder,
             save_dir,
             save_prefix,
             num_images=100,
             images_per_group=25,
             sizes=[768, 1024, 1280],
             block_save_size=2048,
             process_routine=(lambda x: x)
             ):
    """
    This function is devoted to validating results during training on the same collections of images.
    We sample first num_images from the folder path_to_input_images_folder, group them by images_per_group
    samples into one and resize to sizes mentioned in sizes list. Afterwards this groups are fed into the
    network using process_routing function responsible for both pre- and postprocessing. Finally generated
    images are saved in save_dir folder with all necessary information.
    Args:
        path_to_input_images_folder: path to folder from which we draw images for validation
        save_dir: where we save images. Will be created, if necessary
        save_prefix: attach prefix with additional information to each of the saved images, e.g. training step
        num_images: number of images we will validate on
        images_per_group: how much images do we save in single grid
        sizes: let ou network process images in these sizes, e.g. [512, 1024, 1280]
        block_save_size: size of saved image in pixels
        process_routine: the full pipeline including image preprocessing, NN generation and postprocessing

    Returns:
        None
    """
    # Check folders exists.
    assert os.path.isdir(path_to_input_images_folder), "Folder %s doesn't exist." % path_to_input_images_folder

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    image_paths = [os.path.join(path_to_input_images_folder, x) for x in
                   sorted(os.listdir(path_to_input_images_folder))]

    for size in sizes:
        for left_idx in np.arange(0, num_images, images_per_group):
            img_group = [scipy.misc.imresize(scipy.misc.imread(image_path, mode='RGB'), size=(size, size))
                         for image_path in image_paths[left_idx:left_idx + images_per_group]]
            # print("size:", size, "left_idx:", left_idx)
            # print("Before routine: img_group[0].shape:", img_group[0].shape)
            img_group = [process_routine(img) for img in img_group]
            # print("After routine: img_group[0].shape:", img_group[0].shape)
            img_group = group_images_into_square_grid(img_group)
            img_group = scipy.misc.imresize(img_group, size=(block_save_size, block_save_size))
            scipy.misc.imsave(
                name=os.path.join(save_dir,
                                  save_prefix + '_batch%d_sz%d.jpg' % (int(left_idx // images_per_group + 1), size)),
                arr=img_group)

def enhance_image(img):
    return img
    #return exposure.rescale_intensity(img)


def save_obj(obj, path_to_file):
    with open(path_to_file, 'wb') as f:
        pickle.dump(obj, f, 2)

def load_obj(path_to_file):
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)
