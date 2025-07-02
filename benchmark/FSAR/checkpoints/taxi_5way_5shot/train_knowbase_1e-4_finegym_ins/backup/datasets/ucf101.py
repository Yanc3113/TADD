import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

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
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
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


def make_dataset(root_path, train_list_path, n_samples_for_each_video):


    video_names, labels, original_list = load_video_list(train_list_path)

    dataset = []
    count = 0
    final_labels = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('%s does not exist!!!' % (video_path))
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
            print('%s does not exist n_frames!!!' % (video_path))
            continue
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print('%s has 0 frame!!!' % (video_path))
            continue

        count = count + 1
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'frame_indices': list(range(begin_t, end_t+1)),
            'n_frames': n_frames,
            'video_id': video_names[i],
            'label': labels[i]
        }
        for j in range(n_samples_for_each_video):
            sample_j = copy.deepcopy(sample)
            dataset.append(sample_j)
            final_labels.append(labels[i])

    print('Load %d videos' % (count))

    return dataset, final_labels


def load_video_list(path):
    video_names = []
    labels = []
    index = 0
    class_to_idx = {}
    original_list =[]
    with open(path, 'r') as fp:
        for line in fp:
            line_split = line.strip().split('/')
            class_name = line_split[0]
            #path = '{}/{}'.format(class_name.replace(' ', '_'), video_id)
            path = line.strip()
            video_names.append(path)
            if class_name not in class_to_idx.keys():
                class_to_idx[class_name] = index
                index += 1
            original_list.append(line.strip())
            labels.append(class_to_idx[class_name])

    return video_names, labels, original_list

class UCF101(data.Dataset):
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
                 root_path,
                 train_list_path,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(root_path, train_list_path, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

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
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)