import scipy
from random import randint
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from scipy.ndimage import rotate, shift
from utils import one_hot_encode, baseN


class OmniglotDataLoader:
    def __init__(self, args):
        self.data = []
        self.image_size = (args.image_width, args.image_height)
        self.max_angle = args.max_angle
        self.max_shift = args.max_shift
        for dirname, subdirname, filelist in os.walk(args.data_dir):
            if filelist:
                self.data.append([Image.open(dirname + '/' + filename).copy() for filename in filelist])

        self.train_data = self.data[:args.n_train_classes]
        self.test_data = self.data[-args.n_test_classes:]

    def fetch_batch(self, args, mode='train', sample_strategy='random', augment=True):
        n_classes, batch_size, seq_length = args.n_classes, args.batch_size, args.seq_length
        if mode == 'train':
            data = self.train_data
        elif mode == 'test':
            data = self.test_data
        classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
        if sample_strategy == 'random':  # #(sample) per class may not be equal (sec 7)
            seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        elif sample_strategy == 'uniform':  # #(sample) per class are equal
            seq = np.array([np.concatenate([[j] * int(seq_length / n_classes) for j in range(n_classes)])
                            for _ in range(batch_size)])
            for i in range(batch_size):
                np.random.shuffle(seq[i, :])
        seq_pic = [[self.augment(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))],
                                 only_resize=not augment)
                    for j in seq[i, :]]
                   for i in range(batch_size)]
        if args.label_type == 'one_hot':
            seq_encoded = one_hot_encode(seq, n_classes)
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1)
        elif args.label_type == 'five_hot':
            label_dict = [[[int(j) for j in list(baseN(i, 5)) + [0] * (5 - len(baseN(i, 5)))]
                           for i in np.random.choice(range(5 ** 5), replace=False, size=n_classes)]
                          for _ in range(batch_size)]
            seq_encoded_ = np.array([[label_dict[b][i] for i in seq[b]] for b in range(batch_size)])
            seq_encoded = np.reshape(one_hot_encode(seq_encoded_, dim=5), newshape=[batch_size, seq_length, -1])
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, 25]), seq_encoded[:, :-1, :]], axis=1)
        return seq_pic, seq_encoded_shifted, seq_encoded

    def random_augmentation(self):
        """generates random rotation angel and pixel shift"""
        angle = np.random.uniform(-self.max_angle, self.max_angle, size=1)
        shifts = randint(-self.max_shift, self.max_shift + 1), randint(-self.max_shift, self.max_shift + 1)
        return angle, shifts

    def augment(self, orig_image, only_resize=False):
        if only_resize:
            # only invert and resize the image
            image = np.array(ImageOps.invert(orig_image.convert('L')).resize(self.image_size), dtype=np.float32)
            np_image = np.array(image, dtype=np.float32).reshape((self.image_size[0] * self.image_size[1]))
        else:
            image = np.array(ImageOps.invert(orig_image.convert('L')), dtype=np.float32)
            angle, shifts = self.random_augmentation()
            # Rotate the image
            rotated = np.maximum(np.minimum(rotate(image, angle=angle, mode='nearest', reshape=False), 1.), 0.)
            # Shift the image
            shifted = shift(rotated, shift=shifts)
            # Resize the image
            resized = np.asarray(scipy.misc.imresize(shifted, size=self.image_size), dtype=np.float32) / 255.
            np_image = resized.reshape((self.image_size[0] * self.image_size[1]))
        max_value = np.max(np_image)  # normalization is important
        if max_value > 0.:
            np_image = np_image / max_value
        return np_image
