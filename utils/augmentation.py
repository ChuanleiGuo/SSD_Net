import torch
from torchvision import transforms
import numpy as np
from numpy import random
import types

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class Compose(object):
    """compose several augmentations together"""
    def __init__(self, trans):
        self._transforms = trans

    def __call__(self, img, boxes=None, labels=None):
        for t in self._transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Lambda(object):
    """applies a lambda as a transform"""
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self._lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self._lambd(img, boxes, labels)

class ConvertFromInts(object):
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self._mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self._mean
        return img.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels
