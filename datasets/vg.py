import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class VG(Dataset):

    def __init__(self, root, phase, transform=None, target_transform=None):
        root_path = os.path.abspath(root)
        self.root = os.path.join(root_path, 'VG_100K')
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = os.path.join(self.root, 'images')

        img_list_path = os.path.join(self.root, '{}_list_500.txt'.format(self.phase))
        labels_path = os.path.join(self.root, 'vg_category_200_labels_index.json')

        with open(img_list_path, 'r') as fh:  # strip each name; the last line may have no newline
            self.img_names = [line.rstrip('\n') for line in fh if line.strip()]

        # labels : numpy.ndarray, shape->(len(vg), 200)
        # value range->(-1 means label don't exist, 1 means label exist)
        total_labels = json.load(open(labels_path, 'r'))
        self.num_classes = 200
        self.labels = np.zeros((len(self.img_names), self.num_classes)).astype(int) - 1

        for i in range(len(self.img_names)):
            self.labels[i][total_labels[self.img_names[i]]] = 1

        # value range->(-1 means label don't exist, 0 means not sure whether the label exists, 1 means label exist)
        print('[dataset] VG 100K classification phase=%s number of classes=%d  number of images=%d' % (
            self.phase, self.num_classes, len(self.img_names)))

    def class_sample_num(self):
        known_class_num = np.abs(self.labels).sum(0)
        known_class_pos_num = np.clip(self.labels, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num, known_class_num), axis=0)
        print('Number of pos-label per class: {}\n'
              'Number of known-label per class: {}'.format(self.known_class_label[0], self.known_class_label[1]))

    def __getitem__(self, index):
        path, target = self.img_names[index], self.labels[index]
        img = Image.open(os.path.join(self.img_path, path)).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, index, path), target

    def __len__(self):
        return len(self.img_names)

    def get_number_classes(self):
        return self.num_classes

