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
    tmpl = '{{:0{}d}}.jpg'.format(filename_leng)
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

# def load_video_list(path):
#     video_names = []
#     labels = []
#     index = 0
#     class_to_idx = {}
#     original_list =[]
#     with open(path, 'r') as fp:
#         for line in fp:
#             line_split = line.strip().split('/')
#             class_name = line_split[0]
#             #path = '{}/{}'.format(class_name.replace(' ', '_'), video_id)
#             path = line.strip()
#             #print(path)
#             video_names.append(path.replace('HandStandPushups', 'HandstandPushups'))##################################### delete mp4
#             if class_name not in class_to_idx.keys():
#                 class_to_idx[class_name] = index
#                 index += 1
#             original_list.append(line.strip())
#             labels.append(class_to_idx[class_name])

#     return video_names, labels
import re

def load_video_list(path):
    video_names = []
    labels = []
    index = 0
    class_to_idx = {}
    original_list = []

    print(f"Loading video list from: {path}")  # 初始调试信息，显示路径
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            print(f"Processing line: {line}")  # 每行处理调试信息
            
            # 使用正则表达式分隔符进行分割，以支持多种分隔符情况
            line_split = re.split(r'[\\/]+', line)
            print(f"Split line into: {line_split}")  # 分割后的路径部分
            
            # 假设第一个部分是类别名
            if len(line_split) >= 2:
                class_name = line_split[0]
                video_name = line_split[-1]
                print(f"Identified class name: {class_name}, video name: {video_name}")  # 输出类名和视频文件名

                video_path = line  # 保留完整的路径
                video_names.append(video_path)
                print(f"Added video path: {video_path}")  # 路径添加确认

                # 检查类别是否已存在于映射中
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = index
                    print(f"Mapping new class '{class_name}' to index {index}")  # 映射新类别
                    index += 1
                else:
                    print(f"Class '{class_name}' already mapped to index {class_to_idx[class_name]}")  # 已存在类别

                original_list.append(video_path)
                labels.append(class_to_idx[class_name])  # 仅使用主类别的索引作为标签
                print(f"Appended label {class_to_idx[class_name]} for class '{class_name}'")  # 标签添加确认
            else:
                print(f"Error: Unexpected line format '{line}'")  # 异常情况提示

    # 输出最终的类别到索引的映射信息
    print("Final class to index mapping:", class_to_idx)
    print("Generated labels:", labels)
    print(f"Total unique labels count: {len(set(labels))}")  # 检查唯一标签的数量
    print(f"Total video count: {len(video_names)}")  # 总视频数量确认
    
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
            'video_id': osp.splitext(osp.join(*video_names[i].split(os.sep)[-2:]))[0],
        }
        for j in range(n_samples_for_each_video):
            sample_j = copy.deepcopy(sample)
            dataset.append(sample_j)

    return dataset


def make_ucf_video_names(root_path, list_path):


    video_names, labels = load_video_list(list_path)

    count = 0
    final_labels = []
    final_videos = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
            
        video_path = os.path.join(root_path, video_names[i]).replace('\\', '/')
            # 去掉 .mp4 后缀（如果存在）
        if video_path.endswith('.mp4'):
            video_path = video_path[:-4]

            # 打印最终路径进行调试
        print("Final video_path:", video_path)
        print(video_names[i])
        # import pdb; pdb.set_trace();
        # here the winsows path is different from linux path,needed to be modified. wroing answer would ne 'C:/Futures/Knowledge-Prompting-for-FSAR/data_car/RGB_20s_frame_fps_5\\Driver_on_the_Phone/video_13_1.mp4', which is wrong.
        
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
    # print('Load %d videos' % (count))

    return final_videos, final_labels


class UCFVideoList(data.Dataset):
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
        # import pdb; pdb.set_trace()
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
        # base_path = "C:/Futures/Knowledge-Prompting-for-FSAR/data_car/RGB_20s_frame_fps_5/" #car
        base_path = "D:/Data/Frame_Plat_fps_5/" #plat
        video_id = os.path.normpath(self.data[index]['video_id']).replace('\\', '/')
        video_id = os.path.relpath(video_id, base_path)
        n_frames = self.data[index]['n_frames']
        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        #print(frame_indices)
        
        if not self.args.ablation_removeOrig:
            clip = self.loader(path, frame_indices)
            # set_trace()
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
            interpolated_indices = f(aligned_indices).astype(np.int32)
            if self.args.CLIP_visual_fea_preload:
                clip_visual_fea = torch.index_select(self.loaded[video_id], dim=0, index=interpolated_indices)#.unsqueeze(-1)
            else:
                # clip_visual_fea = torch.index_select(torch.load(osp.join(self.CLIP_visual_fea_dir, video_id)+'.pt'), dim=0, index=torch.from_numpy(interpolated_indices))#.unsqueeze(-1)
                # 构建文件路径
                # # import pdb; pdb.set_trace()
                # print(f"Trying to load visual feature from: {self.CLIP_visual_fea_dir}")
                # print(f"Trying to load visual feature plus from: {video_id}")
                # clip_visual_fea_path = osp.join(self.CLIP_visual_fea_dir, video_id) + '.pt'  ##############################
                # print(f"Trying to load visual feature from: {self.CLIP_visual_fea_dir}")
                # print(f"Trying to load visual feature plus from: {video_id}")

                # 提取相对路径部分，去除前面的不必要部分，比如根目录路径
                # 假设你要从 `data_car/RGB_20s_frame_fps_5/` 后面开始保留
                # base_path = "C:/Futures/Knowledge-Prompting-for-FSAR/data_car/RGB_20s_frame_fps_5/"
                # relative_video_id = os.path.relpath(video_id, base_path)

                # 拼接路径并加上 .pt，保留种类/视频文件名的目录结构
                # clip_visual_fea_path = osp.join(self.CLIP_visual_fea_dir, relative_video_id) + '.pt'
                clip_visual_fea_path = osp.join(self.CLIP_visual_fea_dir, video_id) + '.pt'

                # 将反斜杠替换成正斜杠，确保路径格式一致
                clip_visual_fea_path = clip_visual_fea_path.replace('\\', '/')

                # 检查文件是否存在，如果不存在则抛出异常
                if not osp.exists(clip_visual_fea_path):
                    raise FileNotFoundError(f"File not found: {clip_visual_fea_path}")


                # import pdb; pdb.set_trace()
                

                # 打印文件路径
                # print(f"Loading CLIP visual feature from: {clip_visual_fea_path}")
                if not osp.exists(clip_visual_fea_path):
                    raise FileNotFoundError(f"File not found: {clip_visual_fea_path}")


                # 加载并选择特征
                clip_visual_fea = torch.index_select(torch.load(clip_visual_fea_path), dim=0, index=torch.from_numpy(interpolated_indices))#.unsqueeze(-1)
            # set_trace()
        else:
            raise
        # set_trace()
        if self.args.temporal_modeling!='':
            aggre_clip_visual_fea = clip_visual_fea
        else:
            aggre_clip_visual_fea = clip_visual_fea.mean(dim=0, keepdim=True)
        if check_tensor_nan(aggre_clip_visual_fea):
            set_trace()
        # set_trace()
        return clip, target, aggre_clip_visual_fea

    def __len__(self):
        return len(self.data)