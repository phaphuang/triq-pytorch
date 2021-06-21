import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import torch


class GroupGenerator(Dataset):
    """
    Generator to supply group image data, individual dataset should go to individual group because they can have different resolutions
    """
    def __init__(self, image_file_groups, score_groups, batch_size=16, image_aug=True, shuffle=False, imagenet_pretrain=False):
        self.image_file_groups = image_file_groups
        self.score_groups = score_groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.imagenet_pretrain = imagenet_pretrain
        if image_aug:
            # do image augmentation by left-right flip
            self.seq = iaa.Sequential([iaa.Fliplr(0.5)])
        self.image_aug = image_aug
        self.on_epoch_end()

    def __len__(self):
        return sum(self.group_length)

    def on_epoch_end(self):
        if self.shuffle:
            # shuffle both group orders and image orders in each group
            images_scores = list(zip(self.image_file_groups, self.score_groups))
            random.shuffle(images_scores)
            self.image_file_groups, self.score_groups = zip(*images_scores)

            self.index_groups = []
            self.group_length = []
            for i in range(len(self.image_file_groups)):
                self.index_groups.append(np.arange(len(self.image_file_groups[i])))
                self.group_length.append(len(self.image_file_groups[i]) // self.batch_size)

            for i in range(len(self.index_groups)):
                np.random.shuffle(self.index_groups[i])
        else:
            self.index_groups = []
            self.group_length = []
            for i in range(len(self.image_file_groups)):
                self.index_groups.append(np.arange(len(self.image_file_groups[i])))
                self.group_length.append(len(self.image_file_groups[i]) // self.batch_size)

    def __getitem__(self, item):
        lens = 0
        idx_0 = len(self.group_length) - 1
        for i, data_len in enumerate(self.group_length):
            lens += data_len
            if item < lens:
                idx_0 = i
                break
        item -= (lens - self.group_length[idx_0])

        image = np.asarray(Image.open(self.image_file_groups[idx_0][item]), dtype=np.float32)
        if self.imagenet_pretrain:
            # ImageNet normalization
            image /= 127.5
            image -= 1.
        else:
            # Normalization based on the combined database consisting of KonIQ-10k and LIVE-Wild datasets
            image[:, :, 0] -= 117.27205081970828
            image[:, :, 1] -= 106.23294835284031
            image[:, :, 2] -= 94.40750328714887
            image[:, :, 0] /= 59.112836751661085
            image[:, :, 1] /= 55.65498543815568
            image[:, :, 2] /= 54.9486100975773

        y_score = self.score_groups[idx_0][item]

        #print(image.shape)
        image = image.transpose((2, 0, 1))

        if self.image_aug:
            image_aug = self.seq(images=image)
            #print(image_aug.shape)
            return (torch.from_numpy(np.asarray(image_aug)), torch.from_numpy(np.asarray(y_score)))
        else:
            return (torch.from_numpy(np.asarray(image)), torch.from_numpy(np.asarray(y_score)))