import torch
from torch import nn

from models import r2plus1d
from build_knowledge_model import knowledge_model
from ipdb import set_trace

# def generate_model(opt):
#     if opt.model.startswith('CLIP_'):
#         vis_arch_name = opt.model.split('_')[1]
#         from models.CLIP_TSM import TSM
#         model = TSM(opt.n_finetune_classes, opt.sample_duration, vis_arch_name, 8, opt)
#         # return model, model.parameters()
#     elif opt.knowledge_model=='dwconv_fc':
#         model = knowledge_model(opt)
#         from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
#     elif opt.model.endswith('w_knowledge'):
#         # vis_arch_name = opt.model.split('_')[1]
#         from building_model_w_knowledge import r2plus1d_w_knowledge
#         model = r2plus1d_w_knowledge(opt)
#         from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
#         # input=torch.randn(2, 3, 16, 112, 112).cuda()
#         # CLIP_vis_branch = torch.randn(2, 16, 512).cuda()
#         # model(input, CLIP_vis_branch)
#         # set_trace()
#     else:
#         from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
#         print("Making R2+1D model, depth", opt.model_depth)
#         model = r2plus1d.r2plus1d_34(num_classes=opt.n_classes)

#     if not opt.no_cuda:
#         if opt.distributed:
#             # For multiprocessing distributed, DistributedDataParallel constructor
#             # should always set the single device scope, otherwise,
#             # DistributedDataParallel will use all available devices.
#             if opt.gpu is not None:
#                 torch.cuda.set_device(opt.gpu)
#                 model.cuda(opt.gpu)
#                 # When using a single GPU per process and per
#                 # DistributedDataParallel, we need to divide the batch size
#                 # ourselves based on the total number of GPUs we have
#                 opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
#                 opt.n_threads = int(opt.n_threads / opt.ngpus_per_node)
#                 model = torch.nn.parallel.DistributedDataParallel(model,
#                                                                 device_ids=[
#                                                                     opt.gpu])
#             else:
#                 model.cuda()
#                 # DistributedDataParallel will divide and allocate batch_size to all
#                 # available GPUs if device_ids are not set
#                 model = torch.nn.parallel.DistributedDataParallel(model)
#         elif opt.gpu is not None:
#             torch.cuda.set_device(opt.gpu)
#             model = model.cuda(opt.gpu)
#         else:
#             # DataParallel will divide and allocate batch_size to all available GPUs
#             if opt.pretrain_path != '/BS/xian18/work/fsv/pretrained_model/R25D34_Sports1M.pkl' \
#                 and opt.pretrain_path != '/BS/xian18/work/fsv/pretrained_model/mc3_audioset.pth' \
#                     and opt.pretrain_path != '/BS/xian18/work/fsv/pretrained_model/mc3_avts_kinetics.pth':
#                 model = torch.nn.DataParallel(model).cuda()

#             #if opt.pretrain_path == '':
#             #    model = model.cuda()
#             #    model = nn.DataParallel(model, device_ids=None)

#         if opt.pretrain_path:
#             print('!!!!!!!!!!')
#             print('loading pretrained model {}'.format(opt.pretrain_path))
#             pretrain = torch.load(opt.pretrain_path)
#             #assert opt.arch == pretrain['arch']
#             if opt.pretrain_path == '/BS/xian18/work/fsv/pretrained_model/R25D34_Sports1M.pkl':
#                 model.load_state_dict(pretrain['model'])
#                 model = torch.nn.DataParallel(model).cuda()
#             elif opt.pretrain_path == '/BS/xian18/work/fsv/pretrained_model/mc3_audioset.pth':
#                 model.load_state_dict(pretrain)
#                 model.add_module("flatten", torch.nn.Flatten(1))
#                 model.add_module("fc", nn.Linear(256, opt.n_finetune_classes))
#                 model = torch.nn.DataParallel(model).cuda()
#             elif opt.pretrain_path == '/BS/xian18/work/fsv/pretrained_model/mc3_avts_kinetics.pth':
#                 model.load_state_dict(pretrain)
#                 model.add_module("flatten", torch.nn.Flatten(1))
#                 model.add_module("fc", nn.Linear(256, opt.n_finetune_classes))
#                 model = torch.nn.DataParallel(model).cuda()
#             else:
#                 if not opt.model.endswith('w_knowledge'):
#                     model.load_state_dict(pretrain['state_dict'])

#             if opt.model == 'densenet':
#                 model.module.classifier = nn.Linear(
#                     model.module.classifier.in_features, opt.n_finetune_classes)
#                 model.module.classifier = model.module.classifier.cuda()
#             elif opt.gpu is not None:
#                 pass
#             elif not opt.model.endswith('w_knowledge'):
#                 model.module.fc = nn.Linear(model.module.fc.in_features,
#                                             opt.n_finetune_classes)
#                 model.module.fc = model.module.fc.cuda()

#             if opt.layer_lr is not None:
#                 parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
#             else:
#                 parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
#             return model, parameters
#     else:
#         if opt.pretrain_path:
#             print('loading pretrained model {}'.format(opt.pretrain_path))
#             pretrain = torch.load(opt.pretrain_path)
#             assert opt.arch == pretrain['arch']

#             model.load_state_dict(pretrain['state_dict'])

#             if opt.model == 'densenet':
#                 model.classifier = nn.Linear(
#                     model.classifier.in_features, opt.n_finetune_classes)
#             else:
#                 model.fc = nn.Linear(model.fc.in_features,
#                                             opt.n_finetune_classes)

#             parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
#             return model, parameters
#     from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
#     if opt.layer_lr is not None:
#         parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
#     else:
#         parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
#     # set_trace()
#     return model, parameters
    # return model, model.parameters()
    
def generate_model(opt):
    if opt.model.startswith('CLIP_'):
        vis_arch_name = opt.model.split('_')[1]
        from models.CLIP_TSM import TSM
        model = TSM(opt.n_finetune_classes, opt.sample_duration, vis_arch_name, 8, opt)
    elif opt.knowledge_model == 'dwconv_fc':
        model = knowledge_model(opt)
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    elif opt.model.endswith('w_knowledge'):
        from building_model_w_knowledge import r2plus1d_w_knowledge
        model = r2plus1d_w_knowledge(opt)
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    else:
        from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
        print("Making R2+1D model, depth", opt.model_depth)
        model = r2plus1d.r2plus1d_34(num_classes=opt.n_classes)

    if not opt.no_cuda:
        if opt.distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
                opt.n_threads = int(opt.n_threads / opt.ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model = model.cuda(opt.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 确保不加载预训练模型，直接训练新的模型
    print('Training from scratch, no pre-trained weights loaded.')

    from models.r2plus1d import get_fine_tuning_parameters, get_fine_tuning_parameters_layer_lr
    if opt.layer_lr is not None:
        parameters = get_fine_tuning_parameters_layer_lr(model, opt.ft_begin_index, opt.layer_lr)
    else:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)

    return model, parameters

