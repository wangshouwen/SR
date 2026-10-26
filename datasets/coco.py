import os
import json
import subprocess
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

urls = {'train_img': 'http://images.cocodataset.org/zips/train2014.zip',
        'val_img': 'http://images.cocodataset.org/zips/val2014.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}


def download_coco2014(root, phase):
    work_dir = os.getcwd()
    tmpdir = os.path.join(root, 'tmp/')
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if phase == 'train':
        filename = 'train2014.zip'
    elif phase == 'val':
        filename = 'val2014.zip'
    cached_file = os.path.join(tmpdir, filename)
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls[phase + '_img'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls[phase + '_img'], shell=True)
        os.chdir(root)
    # extract file
    img_data = os.path.join(root, filename.split('.')[0])
    if not os.path.exists(img_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    print('[dataset] Done!')

    # train/val images/annotations
    cached_file = os.path.join(tmpdir, 'annotations_trainval2014.zip')
    if not os.path.exists(cached_file):
        print('Downloading: "{}" to {}\n'.format(urls['annotations'], cached_file))
        os.chdir(tmpdir)
        subprocess.call('wget ' + urls['annotations'], shell=True)
        os.chdir(root)
    annotations_data = os.path.join(root, 'annotations')
    if not os.path.exists(annotations_data):
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        command = 'unzip {} -d {}'.format(cached_file, root)
        os.system(command)
    print('[annotation] Done!')

    annotations_data = os.path.join(root, 'annotations')
    anno = os.path.join(root, '{}_anno.json'.format(phase))
    img_id = {}
    annotations_id = {}
    if not os.path.exists(anno):
        annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
        annotations = annotations_file['annotations']
        category = annotations_file['categories']
        category_id = {}
        for cat in category:
            category_id[cat['id']] = cat['name']
        cat2idx = categoty_to_idx(sorted(category_id.values()))
        images = annotations_file['images']
        for annotation in annotations:
            if annotation['image_id'] not in annotations_id:
                annotations_id[annotation['image_id']] = set()
            annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])
        for img in images:
            if img['id'] not in annotations_id:
                continue
            if img['id'] not in img_id:
                img_id[img['id']] = {}
            img_id[img['id']]['file_name'] = img['file_name']
            img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        anno_list = []
        for k, v in img_id.items():
            anno_list.append(v)
        json.dump(anno_list, open(anno, 'w'))
        if not os.path.exists(os.path.join(root, 'category.json')):
            json.dump(cat2idx, open(os.path.join(root, 'category.json'), 'w'))
        del img_id
        del anno_list
        del images
        del annotations_id
        del annotations
        del category
        del category_id
    print('[json] Done!')
    os.chdir(work_dir)


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class COCO2014(Dataset):
    def __init__(self, root, phase, transform=None, target_transform=None):
        root_path = os.path.abspath(root)
        self.root = os.path.join(root_path, 'MSCOCO_2014')
        if phase == 'test':
            self.phase = 'val'
        else:
            self.phase = phase
        self.img_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        images = []
        for img in self.img_list:
            name = img['file_name']
            label_list = img['labels']
            label = np.zeros(self.num_classes) - 1
            label[label_list] = 1
            item = (name, label)
            images.append(item)
        self.images = images
        names, labels = list(zip(*self.images))
        labels = np.stack(labels, 0)
        self.img_names = names
        self.labels = labels
        print('[dataset] COCO2014 classification phase={} number of classes={}  number of images={}'.format(
            self.phase, self.num_classes, len(self.img_list)))

    def class_sample_num(self):
        known_class_num = np.abs(self.labels).sum(0)
        known_class_pos_num = np.clip(self.labels, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num, known_class_num), axis=0)
        print('Number of pos-label per class: {}\n'
              'Number of known-label per class: {}'.format(self.known_class_label[0], self.known_class_label[1]))

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        if not os.path.exists(list_path):
            download_coco2014(self.root, self.phase)
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __getitem__(self, index):
        path, target = self.img_names[index], self.labels[index]
        img = Image.open(os.path.join(self.root, '{}2014'.format(self.phase), path)).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, index, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return self.num_classes