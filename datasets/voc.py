import csv
import os
import tarfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image



object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls2007 = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


urls2012 = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
    'test_images_2012': 'http://pjreddie.com/media/files/VOC2012test.tar',
}


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.
    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.
    Returns
    -------
    filename : str
        The location of the downloaded file.
    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
    return data


def read_object_labels(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                # labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc(root, ds_name):
    ds_name = ds_name.upper()
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', ds_name, 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')
    if ds_name == "VOC2007":
        urls = urls2007
        year = "2007"
    else:
        urls = urls2012
        year = "2012"

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
            download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_'+year])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['trainval_'+year], cached_file))
            download_url(urls['trainval_'+year], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    prefix = '' if ds_name == 'VOC2007' else '2012_'
    test_image = os.path.join(path_devkit, ds_name+'/JPEGImages/'+prefix+'000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_images_'+year])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_images_'+year], cached_file))
            download_url(urls['test_images_'+year], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    if ds_name == "VOC2007":
        test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
        if not os.path.exists(test_anno):

            # download test annotations
            parts = urlparse(urls['test_anno_2007'])
            filename = os.path.basename(parts.path)
            cached_file = os.path.join(tmpdir, filename)

            if not os.path.exists(cached_file):
                print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
                download_url(urls['test_anno_2007'], cached_file)

            # extract file
            print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
            cwd = os.getcwd()
            tar = tarfile.open(cached_file, "r")
            os.chdir(root)
            tar.extractall()
            tar.close()
            os.chdir(cwd)
            print('[dataset] Done!')


class Voc2007(Dataset):
    def __init__(self, root, phase, transform=None, target_transform=None):
        self.root = os.path.abspath(root)
        self.path_devkit = os.path.join(self.root, 'VOCdevkit')
        self.path_images = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        if phase == 'train':
            self.phase = 'trainval'
        else:
            self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc(self.root, ds_name='VOC2007')

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + self.phase + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2007', self.phase)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        names, labels = list(zip(*self.images))
        self.img_names = names
        labels = np.stack(labels, 0)
        self.labels = labels

        print('[dataset] VOC 2007 classification phase=%s number of classes=%d  number of images=%d' % (
            self.phase, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.img_names[index], self.labels[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, index, path), target

    def __len__(self):
        return len(self.images)

    def class_sample_num(self):
        known_class_num = np.abs(self.labels).sum(0)
        known_class_pos_num = np.clip(self.labels, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num,known_class_num), axis=0)
        print('Number of pos-label per class: {}\n'
              'Number of known-label per class: {}'.format(self.known_class_label[0], self.known_class_label[1]))

    def get_number_classes(self):
        return len(self.classes)


class Voc2012(Dataset):
    def __init__(self, root, phase, transform=None, target_transform=None):
        self.root = os.path.abspath(root)
        self.path_devkit = os.path.join(self.root, 'VOCdevkit')
        self.path_images = os.path.join(self.root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        if phase == 'test':
            self.phase = 'val'
        else:
            self.phase = phase
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        download_voc(self.root, ds_name='VOC2012')

        # define path of csv file
        path_csv = os.path.join(self.root, 'files', 'VOC2012')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + self.phase + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, 'VOC2012', self.phase)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        names, labels = list(zip(*self.images))
        labels = np.stack(labels, 0)
        self.img_names = names
        self.labels = labels
        self.class_sample_num()

        print('[dataset] VOC 2012 classification phase=%s number of classes=%d  number of images=%d' % (
            self.phase, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.img_names[index], self.labels[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, index, path), target

    def __len__(self):
        return len(self.images)

    def class_sample_num(self):
        known_class_num = np.abs(self.labels).sum(0)
        known_class_pos_num = np.clip(self.labels, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num,known_class_num), axis=0)
        print('Number of pos-label per class: {}\n'
              'Number of known-label per class: {}'.format(self.known_class_label[0], self.known_class_label[1]))

    def get_number_classes(self):
        return len(self.classes)

