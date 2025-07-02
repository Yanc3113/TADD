import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import pandas as pd
from random import shuffle
import pickle
from utils import load_value_file
import random

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def temporal_cropping(frame_indices, sample_duration):

    rand_end = max(0, len(frame_indices) - sample_duration - 1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + sample_duration, len(frame_indices))

    out = frame_indices[begin_index:end_index]

    for index in out:
        if len(out) >= sample_duration:
            break
        out.append(index)

    return out


def make_dataset_clip(yfcc_root, yfcc_results, target_classes):
    dataset = []
    res = yfcc_results
    video_list = []
    segment_list = []
    label_list = []
    weight_list = []
    weighting = False
    if 'prob' in res.keys():
        print('There is weight')
        weighting = True
    for i, c in enumerate(target_classes):

        video_list.extend(res['video_name'][c])
        #print(res['segment'][c])
        segment_list.extend(res['segment'][c])
        label_list.extend([i] * len(res['segment'][c]))
        if weighting:
            weight_list.extend(res['prob'][c])

    for i in range(len(segment_list)):
        video_path = os.path.join(yfcc_root, video_list[i][:-4])
        if not os.path.exists(video_path):
            video_path = os.path.join('/scratch/BS/pool1/yxian/data/yfcc100m/video_jpeg/', video_list[i][:-4])
            if not os.path.exists(video_path):
                print("%s does not exist!!!!" % video_path)
                continue
        #print(segment_list[i])
        frame_indices = list(range(segment_list[i][0], segment_list[i][1]+1))
        label = label_list[i]

        if weighting:
            sample = {
                'video': video_path,
                'frame_indices': frame_indices,
                'label': label,
                'weight': weight_list[i]
            }
        else:
            sample = {
                'video': video_path,
                'frame_indices': frame_indices,
                'label': label
            }
        dataset.append(sample)

    return dataset


class YFCC100M(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 yfcc_root,
                 yfcc_results,
                 target_classes,
                 spatial_transform=None,
                 temporal_transform=None,
                 weighting=False,
                 get_loader=get_default_video_loader):


        self.data = make_dataset_clip(yfcc_root, yfcc_results, target_classes)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.weighting = weighting

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        if len(clip) == 0:
            print(path)
            print(frame_indices)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]['label']

        if self.weighting:
            if 'weight' in self.data[index].keys():
                clip = self.data[index]['weight'] * clip

        return clip, target

    def __len__(self):
        return len(self.data)
