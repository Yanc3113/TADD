import torch
from torch import nn

from models import resnet,  r2plus1d, avts


def generate_model(opt):
    # *********************build model object*********************
    assert opt.clip_model in [
        'resnet',  'r2plus1d', 'avts', 'i3d', 'CLIP_RN50', 'r2plus1d_w_knowledge'
    ]

    if opt.knowledge_model=='dwconv_fc':
        # vis_arch_name = opt.model.split('_')[1]
        from build_knowledge_model import knowledge_model
        model = knowledge_model(opt)
        # from ipdb import set_trace;set_trace()
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    elif opt.clip_model.endswith('w_knowledge'):
        # vis_arch_name = opt.model.split('_')[1]
        from building_model_w_knowledge import r2plus1d_w_knowledge
        model = r2plus1d_w_knowledge(opt)
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    elif opt.clip_model == 'resnet':
        assert opt.clip_model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters
        from models.resnet import get_fine_tuning_parameters_layer_lr

        if opt.clip_model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.clip_model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.clip_model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.clip_model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                no_last_fc=opt.no_last_fc)
        elif opt.clip_model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.clip_model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.clip_model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    elif opt.clip_model.lower() in ['r2+1d', 'r2.5d', 'r2plus1d']:
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
        print("Making R2+1D model, depth", opt.clip_model_depth)
        model = r2plus1d.r2plus1d_34(num_classes=opt.n_classes)

    elif opt.clip_model.lower() in ['avts']:
        from models.avts import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
        print("Making AVTS model")
        model = avts.mc3_avts()
        model.add_module("flatten", torch.nn.Flatten(1))
        model.add_module("fc", nn.Linear(256, opt.n_classes))

    elif opt.clip_model == 'i3d':
        from models import i3res
        if opt.clip_model_depth == 50:
            print("Making I3D model, depth", opt.clip_model_depth)
            model = i3res.i3res_50(sample_size=opt.sample_size, sample_duration=opt.sample_duration, num_classes=opt.n_classes)
        elif opt.clip_model_depth == 34:
            print("Making I3D model, depth", opt.clip_model_depth)
            model = i3res.i3res_34(sample_size=opt.sample_size, sample_duration=opt.sample_duration, num_classes=opt.n_classes)
    elif opt.clip_model.startswith('CLIP_'):
        vis_arch_name = opt.clip_model.split('_')[1]
        from models.CLIP_TSM import TSM
        model = TSM(opt.n_classes, opt.clip_duration, vis_arch_name, 8, opt)


    # *********************distribute model*********************
    if opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
            opt.n_threads = int(opt.n_threads / opt.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  opt.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # *********************load model weights*********************
    if opt.pretrain_path:
        # print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])

        model.module.fc = nn.Linear(model.module.fc.in_features,
                                    opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        if opt.layer_lr is not None:
            parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
        else:
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)

        return model, parameters

    # else:
    #     if opt.layer_lr is not None:
    #         parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
    #     else:
    #         parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    #     return model, parameters

    return model