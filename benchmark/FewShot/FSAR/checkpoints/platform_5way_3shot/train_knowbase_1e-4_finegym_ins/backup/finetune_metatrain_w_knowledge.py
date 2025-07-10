import os, sys
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from ipdb import set_trace

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalRandomCrop_TSNstyle
from target_transforms import ClassLabel, VideoID
from utils import Logger
from train_w_knowledge import train_epoch
from validation_w_knowledge import val_epoch
from datasets.kinetics_w_knowledge import Kinetics
from datasets.something_w_knowledge import Something
from datasets.ucf101_w_knowledge import UCF101
from datasets.hmdb51_w_knowledge import HMDB51
from datasets.diving48V2_w_knowledge import diving48V2
from datasets.finegym_w_knowledge import finegym
import numpy as np
import random

from lin_utils import backup_files, join_multiple_txts

if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # setup_seed(20)

    opt = parse_opts()

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    else:
        print(f'{opt.result_path} exists! but i keep going')
        # raise ValueError(f'{opt.result_path} exists!')

    '''backup'''
    if not os.path.exists(opt.this_launch_script):
        opt.this_launch_script = '/'.join(opt.this_launch_script.split('/')[1:])
    os.makedirs(os.path.join(opt.result_path, 'backup'))
    backup_files(
        [opt.this_launch_script,
        'datasets',
        'models',
        'building_model_w_knowledge.py',
        'build_knowledge_model.py',
        'finetune_metatrain_w_knowledge.py',
        'model.py',
        'opts.py',
        ],
        os.path.join(opt.result_path, 'backup')
        )
    # set_trace()

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    '''amp'''
    opt.scaler = torch.cuda.amp.GradScaler(enabled=opt.is_amp)
    
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
  
    if opt.model.endswith("w_knowledge") and opt.ablation_removeOrig:
        try:
            model.prepare_for_ablation_orig()
        except:
            model.module.prepare_for_ablation_orig()
    if opt.model.endswith("w_knowledge") and not opt.is_w_knowledge:
        try:
            model.prepare_for_ablation_knowledge()
        except:
            model.module.prepare_for_ablation_knowledge()
            
    from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    if opt.layer_lr is not None:
        parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
    else:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    # set_trace()
    # print(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # if not opt.no_cuda:
    criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)


    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        
        # temporal_transform = TemporalRandomCrop(opt.sample_duration)
        temporal_transform = eval('''{'dense':TemporalRandomCrop, 'sparse':TemporalRandomCrop_TSNstyle}''')[opt.sample_mode](opt.sample_duration)
        target_transform = ClassLabel()
        if opt.dataset == 'kinetics':
            training_data = Kinetics(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)

        elif opt.dataset == 'something':
            training_data = Something(
                # os.path.join(opt.video_path, *(['*']*opt.n_vid_store_layers)),
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)
        elif opt.dataset == 'ucf101':
            training_data = UCF101(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)
        elif opt.dataset == 'hmdb51':
            training_data = HMDB51(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)
        elif opt.dataset == 'diving48V2':
            training_data = diving48V2(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)
        elif opt.dataset == 'finegym':
            training_data = finegym(
                opt.video_path,
                opt.train_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_samples_for_each_video,
                args=opt)

        print(len(training_data))

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)


        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.adam:
            eps = 1e-08 if not opt.is_amp else 1e-03
            optimizer = optim.Adam(parameters, lr=opt.learning_rate, betas=(0.5, 0.999), eps=eps)
        else:
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #    optimizer, 'min', patience=opt.lr_patience)
        if opt.dataset in ['kinetics', 'ucf101', 'hmdb51']:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 60], gamma=0.1)
    if not opt.no_val:
        if opt.model.startswith('CLIP_RN'):
            spatial_transform = model.module.preprocess
        else:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method
            ])
        # temporal_transform = TemporalRandomCrop(opt.sample_duration)
        temporal_transform = eval('''{'dense':TemporalRandomCrop, 'sparse':TemporalRandomCrop_TSNstyle}''')[opt.sample_mode](opt.sample_duration)
        target_transform = ClassLabel()
        if opt.dataset == 'kinetics':
            validation_data = Kinetics(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)
        elif opt.dataset == 'something':
            validation_data = Something(
                opt.val_video_path,
                # os.path.join(opt.val_video_path, *(['*']*opt.n_vid_store_layers)),
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)
        elif opt.dataset == 'ucf101':
            validation_data = UCF101(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)
        elif opt.dataset == 'hmdb51':
            validation_data = HMDB51(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)
        elif opt.dataset == 'diving48V2':
            validation_data = diving48V2(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)
        elif opt.dataset == 'finegym':
            validation_data = finegym(
                opt.val_video_path,
                opt.val_list_path,
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform,
                n_samples_for_each_video=opt.n_val_samples,
                args=opt)

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

        print('# of validation clips %d' % len(validation_data))

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.KnowAssistCLIPzs or opt.fuse_mode!='no' or opt.with_clip_zeroshot:
        if opt.dataset in ['kinetics', 'kinetics100']:
            clsName_list_tobeSelcted_pth = '/home/linhanxi/github/actionKnowledgeXfewshot/few-shot-video-classification/data/kinetics100/data_splits/k100_train_classes.txt'
        elif opt.dataset=='diving48V2':
            clsName_list_pth = '/home/linhanxi/diving48_240p/Diving48_vocab.json'
            import json
            loaded = json.load(open(clsName_list_pth,'rb'))
            clsName_list_tobeSelcted_pth = [" ".join(name) for name in loaded[:36]]
        else:
            raise ValueError
        labelTexts = join_multiple_txts(clsName_list_tobeSelcted_pth)
        # set_trace()
        model.module.select_labelTexts([_ for _ in labelTexts])
        model.module.build_cls_head_4CLIPzs()
        # validate complete zero-shot perf
        _ = val_epoch(0, val_loader, model, criterion, opt,
                                        val_logger)
    else:
        labelTexts = None
    
    # set_trace()
    if opt.l1regu:
        from l1_regu import l1_regularizer
        is_l1_regu_fn = lambda x: x in ['module.general_knowledge_model.2.weight',
        'module.general_knowledge_model.2.bias']
        # is_l1_regu_fn = lambda x: x in ['module.general_knowledge_model.0.weight',
        # 'module.general_knowledge_model.0.bias']
        l1_regularizer_ins = l1_regularizer(opt, is_l1_regu_fn)

    else:
        l1_regularizer_ins = None

    # set_trace()
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger, l1_regularizer=l1_regularizer_ins)


        if not opt.no_val:
            if i % opt.val_every == 0:
                validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger, l1_regularizer=l1_regularizer_ins)
        scheduler.step()

