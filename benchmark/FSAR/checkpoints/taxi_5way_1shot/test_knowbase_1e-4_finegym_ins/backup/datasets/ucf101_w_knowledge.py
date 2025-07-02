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


# def video_loader(video_dir_path, frame_indices, image_loader):
#     video = []
#     filename_leng = len(os.path.splitext(os.listdir(video_dir_path)[0])[0])
#     tmpl = '{{:0{}d}}.png'.format(filename_leng)
#     for i in frame_indices:
#         image_path = os.path.join(video_dir_path, tmpl.format(i))
#         if os.path.exists(image_path):
#             video.append(image_loader(image_path))
#         else:
#             return video

#     return video
def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    # 修改模板为 'frame_{:06d}.png' 以匹配帧命名格式
    tmpl = 'frame_{:06d}.png'  # 你帧的实际命名格式
    for i in frame_indices:
        # 确保路径拼接正确，使用 os.path.join 并替换反斜杠
        image_path = os.path.join(video_dir_path, tmpl.format(i)).replace("\\", "/")
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(f"Frame does not exist: {image_path}")  # 打印缺失的帧
            return video  # 如果帧不存在，直接返回已加载的部分视频

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


# def make_dataset(root_path, train_list_path, n_samples_for_each_video):


#     video_names, labels, original_list = load_video_list(train_list_path)

#     dataset = []
#     count = 0
#     final_labels = []
#     for i in range(len(video_names)):
#         if i % 1000 == 0:
#             print('dataset loading [{}/{}]'.format(i, len(video_names)))

#         video_path = os.path.join(root_path, video_names[i])
#         if not os.path.exists(video_path):
#             print('%s does not exist!!!' % (video_path))
#             continue

#         '''
#         n_frames_file_path = os.path.join(video_path, 'n_frames')
#         if not os.path.exists(n_frames_file_path):
#             print('%s does not exist n_frames!!!' % (video_path))
#             continue
#         n_frames = int(load_value_file(n_frames_file_path))
#         '''
#         n_frames = len(os.listdir(video_path))
#         if n_frames <= 0:
#             print('%s has 0 frame!!!' % (video_path))
#             continue

#         count = count + 1
#         begin_t = 1
#         end_t = n_frames
#         sample = {
#             'video': video_path,
#             'frame_indices': list(range(begin_t, end_t+1)),
#             'n_frames': n_frames,
#             'video_id': osp.splitext(video_names[i])[0],
#             'label': labels[i]
#         }
#         for j in range(n_samples_for_each_video):
#             sample_j = copy.deepcopy(sample)
#             dataset.append(sample_j)
#             final_labels.append(labels[i])

#     print('Load %d videos' % (count))

#     return dataset, final_labels


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
#             video_names.append((path+'.mp4').replace('HandStandPushups', 'HandstandPushups'))
#             if class_name not in class_to_idx.keys():
#                 class_to_idx[class_name] = index
#                 index += 1
#             original_list.append(line.strip())
#             labels.append(class_to_idx[class_name])

#     return video_names, labels, original_list



def make_dataset(root_path, train_list_path, n_samples_for_each_video):
    video_names, labels, original_list = load_video_list(train_list_path)

    dataset = []
    count = 0
    final_labels = []
    for i in range(len(video_names)):
        # if i % 1000 == 0:
            # print('dataset loading [{}/{}]'.format(i, len(video_names)))

        # 使用 os.path.join 来拼接路径，确保跨平台兼容性
        video_path = os.path.join(root_path, video_names[i]).replace("\\", "/")  # 确保路径使用 '/'

        if not os.path.exists(video_path):
            # print('%s does not exist!!!' % (video_path))
            continue

        n_frames = len(os.listdir(video_path))
        # if n_frames <= 0:
        #     print('%s has 0 frame!!!' % (video_path))
            # continue

        count += 1
        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'frame_indices': list(range(begin_t, end_t+1)),
            'n_frames': n_frames,
            'video_id': os.path.splitext(video_names[i])[0],
            'label': labels[i]
        }
        for j in range(n_samples_for_each_video):
            sample_j = sample.copy()
            dataset.append(sample_j)
            final_labels.append(labels[i])

    # print('Load %d videos' % (count))

    return dataset, final_labels



def load_video_list(path):
    video_names = []
    labels = []
    index = 0
    class_to_idx = {}
    original_list = []
    with open(path, 'r') as fp:
        for line in fp:
            line_split = line.strip().split('/')
            class_name = line_split[0]
            # 这里保持路径，不加.mp4
            path = line.strip()
            video_names.append(path.replace('HandStandPushups', 'HandstandPushups'))  # 保持帧目录，不加.mp4后缀
            if class_name not in class_to_idx:
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
                 get_loader=get_default_video_loader,
                 args=None):
        self.data, self.class_names = make_dataset(root_path, train_list_path, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.CLIP_visual_fea_reg = args.CLIP_visual_fea_reg
        self.CLIP_visual_fea_dir = args.CLIP_visual_fea_reg.rstrip('*/')
        self.args = args
        if self.args.CLIP_visual_fea_preload:
            self.preload()

    def preload(self):
        self.loaded = {}
        cands_pths = glob(self.args.CLIP_visual_fea_reg)
        # print(f'start preload!')
        # test_load = torch.load(cands_pths[0])
        # tensor_size = test_load.size()
        for cand_pth in tqdm(cands_pths):
            iid = osp.splitext(osp.basename(cand_pth))[0]
            self.loaded[iid] = torch.load(cand_pth, map_location='cpu')

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is class_index of the target class.
    #     """
    #     path = self.data[index]['video']
    #     video_id = self.data[index]['video_id']
    #     n_frames = self.data[index]['n_frames']
    #     frame_indices = self.data[index]['frame_indices']

    #     if self.temporal_transform is not None:
    #         frame_indices = self.temporal_transform(frame_indices)
            
    #     if not self.args.ablation_removeOrig:
    #         clip = self.loader(path, frame_indices)
    #         # set_trace()
    #         if self.spatial_transform is not None:
    #             # for those augmentations with random feature, such as ['c', 'tl', 'tr', 'bl', 'br'] positions
    #             # for crop, randomly select one.
    #             self.spatial_transform.randomize_parameters()
    #             clip = [self.spatial_transform(img) for img in clip]
    #         clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    #     else:
    #         clip = torch.tensor(0)

    #     target = self.data[index]
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     '''load for clip vis fea'''
    #     if self.args.sample_mode=='dense':
    #         idx_f = np.linspace(0, n_frames-1, num=self.args.clip_visfea_sampleNum, dtype=np.int32) + 1
    #         min_f_ind, max_f_ind = min(frame_indices),  max(frame_indices)
    #         start = (idx_f>=min_f_ind).nonzero()[0][0]
    #         end = (idx_f<=max_f_ind).nonzero()[0][-1]
    #         if self.args.CLIP_visual_fea_preload:
    #             clip_visual_fea = self.loaded[video_id][start:end+1]#.unsqueeze(-1)
    #         else:
    #             clip_visual_fea = torch.load(osp.join(self.CLIP_visual_fea_dir, video_id)+'.pt')[start:end+1]#.unsqueeze(-1)
    #         # set_trace()
    #     elif self.args.sample_mode=='sparse':
    #         f=interpolate.interp1d(np.linspace(0,self.args.clip_visfea_sampleNum-1,self.args.clip_visfea_sampleNum),
    #          np.linspace(0,self.args.clip_visfea_sampleNum-1,self.args.clip_visfea_sampleNum), 
    #          kind='nearest')
    #         aligned_indices = frame_indices/(n_frames)*(self.args.clip_visfea_sampleNum-1)
    #         interpolated_indices = f(aligned_indices).astype(np.int32)
    #         if self.args.CLIP_visual_fea_preload:
    #             clip_visual_fea = torch.index_select(self.loaded[video_id], dim=0, index=interpolated_indices)#.unsqueeze(-1)
    #         else:
    #             clip_visual_fea = torch.index_select(torch.load(osp.join(self.CLIP_visual_fea_dir, video_id)+'.pt'), dim=0, index=torch.from_numpy(interpolated_indices))#.unsqueeze(-1)
    #         # set_trace()
    #     else:
    #         raise
    #     # set_trace()
    #     if self.args.temporal_modeling!='':
    #         aggre_clip_visual_fea = clip_visual_fea
    #     else:
    #         aggre_clip_visual_fea = clip_visual_fea.mean(dim=0, keepdim=True)
    #     if check_tensor_nan(aggre_clip_visual_fea):
    #         set_trace()
    #     # set_trace()
    #     return clip, target, aggre_clip_visual_fea
    
    def __getitem__(self, index):
        # print(f"Fetching data for index: {index}")

        path = self.data[index]['video']
        video_id = self.data[index]['video_id']
        n_frames = self.data[index]['n_frames']
        frame_indices = self.data[index]['frame_indices']

        print(f"Video path: {path}")
        print(f"Video ID: {video_id}, Number of frames: {n_frames}")
        print(f"Frame indices before temporal transform: {frame_indices}")

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            # print(f"Frame indices after temporal transform: {frame_indices}")

        if not self.args.ablation_removeOrig:
            # print(f"Loading frames from path: {path}")
            clip = self.loader(path, frame_indices)
            # print(f"Loaded frames: {len(clip)} frames")
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            # print(f"Clip shape after stacking and permutation: {clip.shape}")
        else:
            clip = torch.tensor(0)
            # print(f"Ablation mode active, skipping clip creation.")

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
            # print(f"Target after transformation: {target}")

        '''load for clip vis fea'''
        if self.args.sample_mode == 'dense':
            idx_f = np.linspace(0, n_frames - 1, num=self.args.clip_visfea_sampleNum, dtype=np.int32) + 1
            min_f_ind, max_f_ind = min(frame_indices), max(frame_indices)
            start = (idx_f >= min_f_ind).nonzero()[0][0]
            end = (idx_f <= max_f_ind).nonzero()[0][-1]
            if self.args.CLIP_visual_fea_preload:
                clip_visual_fea = self.loaded[video_id][start:end + 1]
            else:
                clip_visual_fea = torch.load(osp.join(self.CLIP_visual_fea_dir, video_id) + '.pt')[start:end + 1]

        # elif self.args.sample_mode == 'sparse':
        #     f = interpolate.interp1d(np.linspace(0, self.args.clip_visfea_sampleNum - 1, self.args.clip_visfea_sampleNum),
        #                             np.linspace(0, self.args.clip_visfea_sampleNum - 1, self.args.clip_visfea_sampleNum),
        #                             kind='nearest')
        #     aligned_indices = frame_indices / (n_frames) * (self.args.clip_visfea_sampleNum - 1)
        #     interpolated_indices = f(aligned_indices).astype(np.int32)
        #     if self.args.CLIP_visual_fea_preload:
        #         clip_visual_fea = torch.index_select(self.loaded[video_id], dim=0, index=interpolated_indices)
        #     else:
        #         clip_visual_fea = torch.index_select(torch.load(osp.join(self.CLIP_visual_fea_dir, video_id) + '.pt'),
        #                                             dim=0, index=torch.from_numpy(interpolated_indices))
        elif self.args.sample_mode == 'sparse':
            # 创建插值函数
            f = interpolate.interp1d(
                np.linspace(0, self.args.clip_visfea_sampleNum - 1, self.args.clip_visfea_sampleNum),
                np.linspace(0, self.args.clip_visfea_sampleNum - 1, self.args.clip_visfea_sampleNum),
                kind='nearest'
            )
            
            # 计算 aligned_indices 并打印调试信息
            aligned_indices = frame_indices / n_frames * (self.args.clip_visfea_sampleNum - 1)
            print("Aligned indices:", aligned_indices)
            
            # 插值后的索引
            interpolated_indices = f(aligned_indices).astype(np.int32)
            print("Interpolated indices:", interpolated_indices)
            
            # 检查索引是否超出范围
            if interpolated_indices.max() >= self.args.clip_visfea_sampleNum:
                raise IndexError(f"Interpolated index out of range. Max index: {interpolated_indices.max()}, "
                                f"clip_visfea_sampleNum: {self.args.clip_visfea_sampleNum}")
            
            # 检查是否预加载视觉特征
            if self.args.CLIP_visual_fea_preload:
                try:
                    clip_visual_fea = torch.index_select(self.loaded[video_id], dim=0, index=torch.from_numpy(interpolated_indices))
                except IndexError as e:
                    print(f"IndexError in preload: video_id={video_id}, indices={interpolated_indices}")
                    raise e
            else:
                # 尝试加载文件并索引
                visual_fea_path = osp.join(self.CLIP_visual_fea_dir, video_id) + '.pt'
                print("Loading visual feature file:", visual_fea_path)
                try:
                    clip_visual_fea = torch.index_select(
                        torch.load(visual_fea_path),
                        dim=0,
                        index=torch.from_numpy(interpolated_indices)
                    )
                except IndexError as e:
                    print(f"IndexError in loading: video_id={video_id}, indices={interpolated_indices}")
                    raise e

        else:
            raise ValueError(f"Invalid sample mode: {self.args.sample_mode}")

        # Aggregate visual features
        if self.args.temporal_modeling != '':
            aggre_clip_visual_fea = clip_visual_fea
        else:
            aggre_clip_visual_fea = clip_visual_fea.mean(dim=0, keepdim=True)

        if check_tensor_nan(aggre_clip_visual_fea):
            # print(f"NaN detected in aggregated visual features.")
            set_trace()

        return clip, target, aggre_clip_visual_fea

    

    def __len__(self):
        return len(self.data)