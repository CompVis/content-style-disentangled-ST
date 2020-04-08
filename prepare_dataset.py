from __future__ import print_function
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
import scipy.misc
import utils
from collections import Counter
import random

PATH_TO_WIKIART = '/export/home/asanakoy/workspace/wikiart/images'


ARTISTS_LIST = ['paul-cezanne', 'vincent-van-gogh', 'amedeo-modigliani', 'camille-pissarro', 'pierre-auguste-renoir',
                'childe-hassam', 'paul-gauguin', 'alfred-sisley', 'claude-monet', 'berthe-morisot']
TECHNIQUES_LIST = ['oil', 'watercolor', 'chalk', 'pastel']


class ArtDataset():
    def __init__(self, path_to_art_dataset):
        """
        Initialize ArtDataset object. Images
        :param path_to_art_dataset: path to dataset with images. Inside of the directory there are folders of images
         for different styles/artists.
        """
        self.path_to_dataset = path_to_art_dataset
        self.artists_list = os.listdir(self.path_to_dataset)

        paths = []
        artist_slugs = []
        start_time = time.time()
        for category_idx, artist_slug in enumerate(tqdm(self.artists_list)):
            for file_name in tqdm(os.listdir(os.path.join(self.path_to_dataset, artist_slug))):
                paths.append(os.path.join(self.path_to_dataset, artist_slug, file_name))
                artist_slugs.append(artist_slug)


        self.dataset = pd.DataFrame(np.array([paths, artist_slugs]).T, columns=['path', 'artist_slug'])
        print("\n")
        print("Finished. Constructed Art dataset of %d images." % len(self.dataset))
        print("Time elapsed: %fs." % (time.time() - start_time))

    def get_batch(self, augmentor, batch_size=1):
        """
        Creates a batch of images from a folder self.path_to_art_dataset.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of s batch
        Returns:
            dictionary with fields: image, artist_slug, artist_slug_onehot
            each containing a batch of corresponding values
        """
    
        batch_image = []
        batch_artist_slug = []
        batch_artist_slug_onehot = []
    
        for _ in range(batch_size):
            row = self.dataset.sample(n=1)
            image = scipy.misc.imread(name=row['path'].values[0], mode='RGB')

            if max(image.shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800./max(image.shape))
            if max(image.shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image.shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            if augmentor is not None:
                batch_image.append(augmentor(utils.enhance_image(image)).astype(np.float32))
            else:
                batch_image.append(utils.enhance_image(image).astype(np.float32))
            batch_artist_slug.append(self.artists_list.index(row['artist_slug'].values[0]))
            batch_artist_slug_onehot.append(
                utils.get_one_hot_encoded_vector(l=len(self.artists_list),
                                                 i=self.artists_list.index(row['artist_slug'].values[0]))
            )

        # Now return a batch in correct form
        batch_image = np.asarray(batch_image)
        batch_artist_slug_onehot = np.asarray(batch_artist_slug_onehot)

        return {"image": batch_image,
                "artist_slug": batch_artist_slug,
                "artist_slug_onehot": batch_artist_slug_onehot}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228, copy_batch_times=4):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            for _ in range(copy_batch_times):
                queue.put(batch)


class PlacesDataset():
    categories_names = \
        ['/a/abbey', '/a/arch', '/a/amphitheater', '/a/aqueduct', '/a/arena/rodeo', '/a/athletic_field/outdoor',
         '/b/badlands', '/b/balcony/exterior', '/b/bamboo_forest', '/b/barn', '/b/barndoor', '/b/baseball_field',
         '/b/basilica', '/b/bayou', '/b/beach', '/b/beach_house', '/b/beer_garden', '/b/boardwalk', '/b/boathouse',
         '/b/botanical_garden', '/b/bullring', '/b/butte', '/c/cabin/outdoor', '/c/campsite', '/c/campus',
         '/c/canal/natural', '/c/canal/urban', '/c/canyon', '/c/castle', '/c/church/outdoor', '/c/chalet',
         '/c/cliff', '/c/coast', '/c/corn_field', '/c/corral', '/c/cottage', '/c/courtyard', '/c/crevasse',
         '/d/dam', '/d/desert/vegetation', '/d/desert_road', '/d/doorway/outdoor', '/f/farm', '/f/fairway',
         '/f/field/cultivated', '/f/field/wild', '/f/field_road', '/f/fishpond', '/f/florist_shop/indoor',
         '/f/forest/broadleaf', '/f/forest_path', '/f/forest_road', '/f/formal_garden', '/g/gazebo/exterior',
         '/g/glacier', '/g/golf_course', '/g/greenhouse/indoor', '/g/greenhouse/outdoor', '/g/grotto', '/g/gorge',
         '/h/hayfield', '/h/herb_garden', '/h/hot_spring', '/h/house', '/h/hunting_lodge/outdoor', '/i/ice_floe',
         '/i/ice_shelf', '/i/iceberg', '/i/inn/outdoor', '/i/islet', '/j/japanese_garden', '/k/kasbah',
         '/k/kennel/outdoor', '/l/lagoon', '/l/lake/natural', '/l/lawn', '/l/library/outdoor', '/l/lighthouse',
         '/m/mansion', '/m/marsh', '/m/mausoleum', '/m/moat/water', '/m/mosque/outdoor', '/m/mountain',
         '/m/mountain_path', '/m/mountain_snowy', '/o/oast_house', '/o/ocean', '/o/orchard', '/p/park',
         '/p/pasture', '/p/pavilion', '/p/picnic_area', '/p/pier', '/p/pond', '/r/raft', '/r/railroad_track',
         '/r/rainforest', '/r/rice_paddy', '/r/river', '/r/rock_arch', '/r/roof_garden', '/r/rope_bridge',
         '/r/ruin', '/s/schoolhouse', '/s/sky', '/s/snowfield', '/s/swamp', '/s/swimming_hole',
         '/s/synagogue/outdoor', '/t/temple/asia', '/t/topiary_garden', '/t/tree_farm', '/t/tree_house',
         '/u/underwater/ocean_deep', '/u/utility_room', '/v/valley', '/v/vegetable_garden', '/v/viaduct',
         '/v/village', '/v/vineyard', '/v/volcano', '/w/waterfall', '/w/watering_hole', '/w/wave',
         '/w/wheat_field', '/z/zen_garden', '/a/alcove', '/a/apartment-building/outdoor', '/a/artists_loft',
         '/b/building_facade', '/c/cemetery']
    # categories_names = categories_names[:10]
    categories_names = [x[1:] for x in categories_names]
    def __init__(self, path_to_dataset):
        paths = []
        categories = []

        nmbr_skipped = 0
        categories_skipped = []
        start_time = time.time()
        for category_idx, category_name in enumerate(tqdm(self.categories_names)):
            if os.path.exists(os.path.join(path_to_dataset, category_name)):
                for file_name in tqdm(os.listdir(os.path.join(path_to_dataset, category_name))):
                    paths.append(os.path.join(path_to_dataset, category_name, file_name))
                    categories.append(category_name)
            else:
                print("Category %s can't be found in path %s. Skip it." %
                      (category_name, os.path.join(path_to_dataset, category_name)))
                nmbr_skipped += 1
                categories_skipped.append(category_name)

        self.dataset = pd.DataFrame(np.array([paths, categories]).T, columns=['path', 'category'])
        print("\n")
        print("Finished. Constructed Places2 dataset of %d images." % len (self.dataset))
        print("Time elapsed: %fs. Categories skipped: %d." % (time.time() - start_time, nmbr_skipped))
        print("Following categories are skipped:", categories_skipped, '\n' * 1)

    def get_batch(self, augmentor, batch_size=1):
        """
        Generate bathes of images with attached labels(place category) in two different formats:
        textual and one-hot-encoded.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch we return
        Returns:
            dictionary with fields: image, label_text, label_onehot
            each containing a batch of corresponding values
        """

        batch_image = []
        batch_class = []
        for _ in range(batch_size):
            row = self.dataset.sample(n=1)
            image = scipy.misc.imread(name=row['path'].values[0], mode='RGB')
            image_class = row['category'].values[0]
            image = scipy.misc.imresize(image, size=2.)
            image_shape = image.shape

            if max(image_shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image_shape))
            if max(image_shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image_shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            batch_image.append(augmentor(utils.enhance_image(image)).astype(np.float32))
            batch_class.append(image_class)

        return {"image": np.asarray(batch_image),
                "label_text": batch_class,
                "label_onehot": np.array(
                    [utils.get_one_hot_encoded_vector(l=len(self.categories_names),
                                                      i=self.categories_names.index(x)) for x in batch_class])
                }

    def initialize_batch_worker(self, queue, augmentor, batch_size = 1, seed = 228, copy_batch_times=4):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            for _ in range(random.choice(range(2, copy_batch_times+1))):
                queue.put(batch)


class CocoDataset():
    def __init__(self):
        pass



