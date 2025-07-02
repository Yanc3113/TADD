import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import pandas as pd

from utils import load_value_file
import random

from ipdb import set_trace
from glob import glob
import os.path as osp
from tqdm import tqdm
import numpy as np
from utils import AverageMeter, calculate_accuracy, check_which_nan, check_tensor_nan
import pickle
import hashlib
from scipy import interpolate

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
    filename_leng = len(os.path.splitext(os.listdir(video_dir_path)[0])[0])
    tmpl = 'frames{:05d}.jpg'.format(filename_leng)
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, tmpl.format(i))
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

def get_class_labels(annotations):
    class_labels_map = {}
    index = 0
    for item in annotations:
        if item['label'] not in class_labels_map.keys():
            class_labels_map[item['label']] = index
            index += 1
    return class_labels_map

def get_video_names_and_annotations(data, subset, video_names):

    cur_video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        print(this_subset)
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                cur_name = '{}/{}'.format(label, key)
                print(cur_name)

                if cur_name in video_names:
                    annotations.append(value['annotations'])
                    cur_video_names.append(cur_name)

    print(len(cur_video_names))
    print(len(video_names))

    return cur_video_names, annotations

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
            #video_id = line_split[1][:11]
            #path = '{}/{}'.format(class_name.replace(' ', '_'), video_id)
            path = line_split[1]
            video_names.append(path)
            if class_name not in class_to_idx.keys():
                class_to_idx[class_name] = index
                index += 1
            original_list.append(line.strip())
            labels.append(class_to_idx[class_name])
    return video_names, labels


def make_dataset(video_names, labels, n_samples_for_each_video):

    dataset = []
    for i in range(len(video_names)):

        video_path = video_names[i]
        '''
        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        '''
        n_frames = len(os.listdir(video_path))

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'frame_indices': list(range(begin_t, end_t+1)),
            'label': labels[i],
            'n_frames': n_frames,
            'video_id': osp.basename(video_names[i]),
        }
        for j in range(n_samples_for_each_video):
            sample_j = copy.deepcopy(sample)
            dataset.append(sample_j)

    return dataset


def make_diving48V2_video_names(root_path, list_path):


    video_names, labels = load_video_list(list_path)

    count = 0
    final_labels = []
    final_videos = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('%s does not exist!!!' % (video_path))
            continue
        '''
        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
            print('%s does not exist n_frames!!!' % (video_path))
            continue
        n_frames = int(load_value_file(n_frames_file_path))
        '''
        n_frames = len(os.listdir(video_path))
        if n_frames <= 0:
            print('%s has 0 frame!!!' % (video_path))
            continue

        final_videos.append(video_path)
        final_labels.append(labels[i])
        count += 1
    print('Load %d videos' % (count))

    return final_videos, final_labels


class diving48V2VideoList(data.Dataset):
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
                 video_list,
                 label_list,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 args=None):
        self.data = make_dataset(video_list, label_list, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        '''CLIP_related'''
        self.CLIP_visual_fea_reg = args.CLIP_visual_fea_reg
        self.CLIP_visual_fea_dir = args.CLIP_visual_fea_reg.rstrip('*/')
        self.args = args
        if self.args.CLIP_visual_fea_preload:
            self.preload()

    def preload(self):
        self.loaded = {}
        cands_pths = glob(self.args.CLIP_visual_fea_reg)
        print(f'start preload!')
        # test_load = torch.load(cands_pths[0])
        # tensor_size = test_load.size()
        for cand_pth in tqdm(cands_pths):
            iid = osp.splitext(osp.basename(cand_pth))[0]
            self.loaded[iid] = torch.load(cand_pth, map_location='cpu')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        video_id = self.data[index]['video_id']
        n_frames = self.data[index]['n_frames']
        frame_indices = self.data[index]['frame_indices']
        # set_trace()
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        #print(frame_indices)
        # set_trace()
        
        if not self.args.ablation_removeOrig:
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                # for those augmentations with random feature, such as ['c', 'tl', 'tr', 'bl', 'br'] positions
                # for crop, randomly select one.
                if not self.args.CLIPzeroshotEval:
                    self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        else:
            clip = torch.tensor(0)

        target = self.data[index]['label']
        # set_trace()
        '''load for clip vis fea'''
        if self.args.sample_mode=='dense':
            idx_f = np.linspace(0, n_frames-1, num=self.args.clip_visfea_sampleNum, dtype=np.int32) + 1
            min_f_ind, max_f_ind = min(frame_indices),  max(frame_indices)
            start = (idx_f>=min_f_ind).nonzero()[0][0]
            end = (idx_f<=max_f_ind).nonzero()[0][-1]
            if self.args.CLIP_visual_fea_preload:
                clip_visual_fea = self.loaded[video_id][start:end+1]#.unsqueeze(-1)
            else:
                clip_visual_fea = torch.load(osp.join(self.CLIP_visual_fea_dir, video_id)+'.pt')[start:end+1]#.unsqueeze(-1)
            # set_trace()
        elif self.args.sample_mode=='sparse':
            f=interpolate.interp1d(np.linspace(0,self.args.clip_visfea_sampleNum-1,self.args.clip_visfea_sampleNum),
             np.linspace(0,self.args.clip_visfea_sampleNum-1,self.args.clip_visfea_sampleNum), 
             kind='nearest')
            # set_trace()
            aligned_indices = frame_indices/(n_frames)*(self.args.clip_visfea_sampleNum-1)
            interpolated_indices = f(aligned_indices).astype(np.int64)
            if self.args.CLIP_visual_fea_preload:
                clip_visual_fea = torch.index_select(self.loaded[video_id], dim=0, index=interpolated_indices)#.unsqueeze(-1)
            else:
                clip_visual_fea = torch.index_select(torch.load(osp.join(self.CLIP_visual_fea_dir, video_id)+'.pt'), dim=0, index=torch.from_numpy(interpolated_indices))#.unsqueeze(-1)
        else:
            raise
        # set_trace()
        if self.args.temporal_modeling!='':
            aggre_clip_visual_fea = clip_visual_fea
        else:
            aggre_clip_visual_fea = clip_visual_fea.mean(dim=0, keepdim=True)

        if check_tensor_nan(aggre_clip_visual_fea):
            print(path, video_id, n_frames)
            raise
        if self.args.return_id:
            return clip, target, aggre_clip_visual_fea, video_id
        else:
            return clip, target, aggre_clip_visual_fea


    def __len__(self):
        return len(self.data)


def append_yfcc(dataset, yfcc_root, yfcc_results, topk):
    res = yfcc_results
    video_list = []
    segment_list = []
    label_list = []
    for c in range(5):
        video_list.extend(res['video_name'][c][:topk])
        segment_list.extend(res['segment'][c][:topk])
        label_list.extend([c] * topk)

    for i in range(len(segment_list)):
        video_path = os.path.join(yfcc_root, video_list[i][:-4])
        if not os.path.exists(video_path):
            print("%s does not exist!!!!" % video_path)
            continue

        frame_indices = list(range(segment_list[i][0], segment_list[i][1]+1))
        label = int(label_list[i])
        sample = {
            'video': video_path,
            'frame_indices': frame_indices,
            'label': torch.LongTensor([label])[0]
        }
        dataset.append(sample)

    return dataset