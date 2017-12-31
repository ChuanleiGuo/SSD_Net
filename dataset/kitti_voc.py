import os
import xml.etree.ElementTree as ET
import numpy as np
from dataset.imdb import Imdb
from evaluate.eval_voc import voc_eval
import cv2

class KITTIVoc(Imdb):
    """
    Implement of Imdb for KITTI dataset in VOC format

    # Parameters

    image_set: str
        set to be used, can be train, val, test
    root_path: str
        root path of the dataset
    shuffle: boolean
        whether to initial shuffle the image list
    is_train: boolean
        if true, will load annotaions
    class_names: str
        class names seperated by comma
    names: str
        the file which stores the class_names, will take effect if class_names is None
    true_negative_images: bool
        whether to include true_negative_images
    """
    def __init__(self, image_set, root_path, shuffle=True, is_train=True, class_names=None,
            names="kitti_voc.names", true_negative_images=False):
        super(KITTIVoc, self).__init__("KITTI_" + image_set)
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "training", "image_2")
        self.extension = ".png"
        self.is_train = is_train
        self.true_negative_images = true_negative_images

        if class_names is not None:
            self.classes = class_names.strip().split(',')
        else:
            self.classes = self._load_class_names(names,
                os.path.join(os.path.dirname(__file__), "names"))

        self.config = {
            "use_difficult": True,
            'comp_id': 'comp4'
        }
        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()

        if not self.true_negative_images:
            self._filter_image_with_no_gt()

    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _filter_image_with_no_gt(self):
        """
        filter images that have no ground-truth labels.
        use case: when you wish to work only on a subset of pascal classes, you have 2 options:
            1. use only the sub-dataset that contains the subset of classes
            2. use all images, and images with no ground-truth will count as true-negative images
        :return:
        self object with filtered information
        """

        # filter images that do not have any of the specified classes
        self.labels = [f[np.logical_and(f[:, 0] >= 0, f[:, 0] <= self.num_classes-1), :] for f in self.labels]
        # find indices of images with ground-truth labels
        gt_indices = [idx for idx, f in enumerate(self.labels) if not f.size == 0]

        self.labels = [self.labels[idx] for idx in gt_indices]
        self.image_set_index = [self.image_set_index[idx] for idx in gt_indices]
        old_num_images = self.num_images
        self.num_images = len(self.labels)

        print ('filtering images with no gt-labels. can abort filtering using *true_negative* flag')
        print ('... remaining {0}/{1} images.  '.format(self.num_images, old_num_images))

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        # Parameters

        shuffle: bool
            whether to shuffle the image list

        # Returns

        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.root_path, self.image_set + ".txt")
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_lines = [x.strip() for x in f.readlines()]
            image_set_lines = [x.split(' ')[0].strip() for x in image_set_lines]
            image_set_index = [x.split('/')[-1][:-4] for x in image_set_lines]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, "..", "label_2car", "xml", index + ".xml")
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            label = []

            for obj in root.iter("object"):
                difficult = int(obj.find("difficult").text)
                cls_name = obj.find("name").text
                if cls_name not in self.classes:
                    cls_id = len(self.classes)
                else:
                    cls_id = self.classes.index(cls_name)
                xml_box = obj.find("bndbox")
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax, difficult])
            temp.append(np.array(label))
        return temp

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]
