import torch
import torch.nn as nn
from models import r2plus1d
from ipdb import set_trace
import clip
from utils import AverageMeter, calculate_accuracy, check_which_nan, check_tensor_nan
from temporal_modeling import bottleneck_CNN
from lin_utils import backup_files, join_multiple_txts
import random


class r2plus1d_w_knowledge(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.target_dim_selections = '''{'r2plus1d_w_knowledge-34':512}'''
        self.reduce_ratio_selections = '''{'r2plus1d_w_knowledge-34':16}'''
        self.build_base_model()
        # set_trace()
        self.proposals_fea = torch.load(self.args.proposals_fea_pth, map_location=self.device).to(torch.float)
        if not self.args.ablation_onlyCLIPvisfea:
            self.args.match_result_dim = self.proposals_fea.size(0)
        else:
            self.args.match_result_dim = 512
        self.build_knowledge_fuser()
        self.build_origfea_fuser()
        if args.fuse_mode!='no':
            self.build_CLIP_zeroshot_fuser()
        self.scaled_cosine_simer = scaled_cosine_sim(self.args.CLIP_visual_arch, enabled_grad_flow=args.grad_enabled_in_match, is_softmax=False)
        if args.fuse_mode=='no':
            self.new_fc = nn.Linear(self.original_dim, self.args.n_finetune_classes).to(self.device)
        elif args.fuse_mode=='cat':
            self.new_fc = nn.Linear(self.original_dim+self.args.n_finetune_classes, self.args.n_finetune_classes).to(self.device)
        else:
            raise
        if args.KnowAssistCLIPzs:
            torch.nn.init.zeros_(self.new_fc.weight)
            torch.nn.init.zeros_(self.new_fc.bias)
        if args.fuse_mode=='cat':
            torch.nn.init.zeros_(self.new_fc.weight)
            torch.nn.init.zeros_(self.new_fc.bias)
            for i in range(self.args.n_finetune_classes):
                torch.nn.init.constant_(self.new_fc.weight[i,self.original_dim+i], 1)
            self.add_labelTexts_dataset(args.dataset)
        # set_trace()
        # if not self.args.is_w_knowledge:
        #     self.prepare_for_ablation()

    def build_base_model(self):
        self.base_model = torch.nn.DataParallel((r2plus1d.r2plus1d_34(num_classes=self.args.n_classes)))
        if self.args.pretrain_path:
            pretrain = torch.load(self.args.pretrain_path)
            self.base_model.load_state_dict(pretrain['state_dict'])
        # remove the original fc head
        self.original_dim = eval(self.target_dim_selections)[self.args.arch] if self.args.ablation_removeOrig else self.base_model.module.fc.in_features
        self.base_model.module.fc = nn.Identity()

    def build_knowledge_fuser(self):
        target_dim = eval(self.target_dim_selections)[self.args.arch]
        reduce_ratio = eval(self.reduce_ratio_selections)[self.args.arch]
        if self.args.dropout_w_knowledge!=0. or self.args.testtime_dropout!=0.:
            assert (self.args.dropout_w_knowledge==0. or self.args.testtime_dropout==0.)
            dropout = nn.Dropout(self.args.dropout_w_knowledge + self.args.testtime_dropout)
        else:
            dropout = nn.Identity()
        # **temporal modeling for knowledge**
        if self.args.temporal_modeling=='bottleneck_CNN':
            temporal_model = bottleneck_CNN(depth=self.args.temporal_depth, kernel_size=self.args.temporal_kernel_size, padding=(self.args.temporal_kernel_size-1)//2, in_planes=self.args.match_result_dim, out_planes=target_dim, channel_reduction_ratio=1, bottleneck_reduction=4, hashead=False, firstlayer_reduction=8)
        elif self.args.temporal_modeling=='linear_cls':
            temporal_model = nn.AdaptiveAvgPool1d(1)
        elif self.args.temporal_modeling=='two_linears_cls':
            temporal_model = nn.Sequential(
            nn.Conv1d(self.args.match_result_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
            )
        elif self.args.temporal_modeling=='TSM1':
            from temporal_shift import TemporalShift1D
            temporal_model = nn.Sequential(
            TemporalShift1D(nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim//reduce_ratio, 1)),
            nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
            nn.ReLU(inplace=True),
            TemporalShift1D(nn.Conv1d(self.args.match_result_dim//reduce_ratio, target_dim, 1)),
            nn.BatchNorm1d(target_dim),
            nn.AdaptiveAvgPool1d(1)
            )
        elif self.args.temporal_modeling=='TSM2':
            from temporal_shift import TemporalShift1D
            temporal_model = nn.Sequential(
            nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim//reduce_ratio, 1),
            nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
            nn.ReLU(inplace=True),
            TemporalShift1D(nn.Conv1d(self.args.match_result_dim//reduce_ratio, target_dim, 1)),
            nn.BatchNorm1d(target_dim),
            nn.AdaptiveAvgPool1d(1)
            )
        else:
            temporal_model = nn.Sequential(
            nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim//reduce_ratio, 1),
            nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.args.match_result_dim//reduce_ratio, target_dim, 1),
            nn.BatchNorm1d(target_dim),
            )
        # set_trace()
        self.knowledge_early_fuser = nn.Sequential(
            temporal_model,
            dropout
        ).to(self.device)
        self.fuse_relu = nn.ReLU(inplace=True)
    
    def build_origfea_fuser(self):
        target_dim = eval(self.target_dim_selections)[self.args.arch]
        if self.args.dropout_w_knowledge!=0.:
            dropout = nn.Dropout(self.args.dropout_w_knowledge)
        else:
            dropout = nn.Identity()
        self.origfea_fuser = nn.Sequential(
            nn.Conv1d(target_dim, target_dim, 1),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(inplace=True),
            dropout)

    def build_CLIP_zeroshot_fuser(self):
        if self.args.dropout_w_knowledge!=0.:
            dropout = nn.Dropout(self.args.dropout_w_knowledge)
        else:
            dropout = nn.Identity()
        self.CLIP_zeroshot_fuser = nn.Sequential(
            # nn.BatchNorm1d(self.args.n_finetune_classes),
            dropout)

    def prepare_for_ablation_knowledge(self):
        self.fuse_relu = nn.Identity()
        self.scaled_cosine_simer = fake_module(torch.zeros(1,1).to(self.device))
        self.knowledge_early_fuser = fake_module(torch.zeros(1,1).to(self.device))
        self.origfea_fuser  = nn.Identity()

    def prepare_for_ablation_orig(self):
        self.fuse_relu = nn.Identity()
        self.origfea_fuser  = fake_module(torch.zeros(1,1).to(self.device))
        self.base_model = nn.Identity()

    def forward(self, input, CLIP_vis_branch):
        # produce knowledge_probs online
        if not self.args.ablation_onlyCLIPvisfea:
            match = self.scaled_cosine_simer(CLIP_vis_branch, self.proposals_fea).transpose(-1,-2).contiguous()
        else:
            match = CLIP_vis_branch.transpose(-1,-2)
        # torch.autograd.grad(match.mean(), CLIP_vis_branch, retain_graph=True)[0].max()
        # print(match.size()) # 32， 1 512
        if 'FLOPs' in self.args.result_path:
            from lin_utils import flops
            from fvcore.nn import FlopCountAnalysis
            from ipdb import set_trace; set_trace()
            FlopCountAnalysis(self.knowledge_early_fuser, match).total() + FlopCountAnalysis(self.new_fc, fused_fea).total()

            # flops(model, (inputs, clip_visfeas))
        # from ipdb import set_trace;set_trace()
        knowledge = self.knowledge_early_fuser(match).squeeze(-1)
        original_fea = self.origfea_fuser(self.base_model(input).unsqueeze(-1)).squeeze(-1)
        if check_tensor_nan(knowledge) or check_tensor_nan(original_fea):
            set_trace()
        fused_fea = self.fuse_relu(original_fea + knowledge)
        logits = self.new_fc(fused_fea)
        return logits

    def cat_fuse_forward(self, input, CLIP_vis_branch, force_open_mask=False):
        # produce knowledge_probs online
        if not self.args.ablation_onlyCLIPvisfea:
            match = self.scaled_cosine_simer(CLIP_vis_branch, self.proposals_fea).transpose(-1,-2).contiguous()
        else:
            match = CLIP_vis_branch.transpose(-1,-2)
        # torch.autograd.grad(match.mean(), CLIP_vis_branch, retain_graph=True)[0].max()
        # print(match.size()) # 32， 1 512
        if 'FLOPs' in self.args.result_path:
            from lin_utils import flops
            from fvcore.nn import FlopCountAnalysis
            from ipdb import set_trace; set_trace()
            FlopCountAnalysis(self.knowledge_early_fuser, match).total() + FlopCountAnalysis(self.new_fc, fused_fea).total()

            # flops(model, (inputs, clip_visfeas))
        # from ipdb import set_trace;set_trace()
        knowledge = self.knowledge_early_fuser(match).squeeze(-1)
        original_fea = self.origfea_fuser(self.base_model(input).unsqueeze(-1)).squeeze(-1)
        if check_tensor_nan(knowledge) or check_tensor_nan(original_fea):
            set_trace()
        zeroshot_logits = self.scaled_cosine_simer(CLIP_vis_branch, self.cur_textfeats4labelTexts).transpose(-1,-2).contiguous()
        zeroshot_logits = zeroshot_logits.mean(-1)
        fused_zeroshot_logits = self.CLIP_zeroshot_fuser(zeroshot_logits) 
        if random.random()>0.5 and (self.training or force_open_mask):
            fused_zeroshot_logits = fused_zeroshot_logits*0
            pass
        # if random.random()>0.5
        fused_fea = self.fuse_relu(original_fea + knowledge)
        # from ipdb import set_trace;set_trace()
        cls_input = torch.cat([fused_fea, fused_zeroshot_logits], dim=-1)
        logits = self.new_fc(cls_input)
        return logits

    def interpreter(self, input, CLIP_vis_branch, labels):
        # https://discuss.pytorch.org/t/how-do-i-calculate-the-gradients-of-a-non-leaf-variable-w-r-t-to-a-loss-function/5112/2?u=zhi_li
        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g.detach()
            h = v.register_hook(hook)
            return h
        # produce knowledge_probs online
        if not self.args.ablation_onlyCLIPvisfea:
            match = self.scaled_cosine_simer(CLIP_vis_branch, self.proposals_fea).transpose(-1,-2).contiguous()
        else:
            match = CLIP_vis_branch.transpose(-1,-2)
        h = require_nonleaf_grad(match)
        # torch.autograd.grad(match.mean(), CLIP_vis_branch, retain_graph=True)[0].max()
        # print(match.size()) # 32， 1 512
        knowledge = self.knowledge_early_fuser(match).squeeze(-1)
        original_fea = self.origfea_fuser(self.base_model(input).unsqueeze(-1)).squeeze(-1)
        if check_tensor_nan(knowledge) or check_tensor_nan(original_fea):
            set_trace()
        fused_fea = self.fuse_relu(original_fea + knowledge)
        logits = self.new_fc(fused_fea)
        # torch.autograd.grad(logits.mean(), fused_fea, retain_graph=True)[0].max()
        # torch.autograd.grad(logits.mean(), match, retain_graph=True)[0].max()
        # if self.args.grad_enabled_in_embeddin:
        #     from ipdb import set_trace;set_trace()
        return logits, match, h

    def knowAssist_CLIPzeroshot(self, visInput, CLIP_vis_branch):
        assert hasattr(self, 'cur_textfeats4labelTexts')
        KnowAssist_logits = self.forward(visInput, CLIP_vis_branch)
        zeroshot_logits = self.scaled_cosine_simer(CLIP_vis_branch, self.cur_textfeats4labelTexts).transpose(-1,-2).contiguous()
        zeroshot_logits = zeroshot_logits.mean(-1)
        # set_trace()
        combined_logits = KnowAssist_logits + zeroshot_logits
        return combined_logits

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def add_labelTexts(self, labelTexts):
        self.labelTexts = labelTexts
        model, preprocess = clip.load("ViT-B/16")
        text_tokens = clip.tokenize(["This is " + desc for desc in labelTexts]).cuda() 
        self.register_buffer('textfeats4labelTexts', model.encode_text(text_tokens).float())
        # self.textfeats4labelTexts = model.encode_text(text_tokens).float()
        del model

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def add_labelTexts_dataset(self, dataset):
        if dataset in ['kinetics', 'kinetics100']:
            clsName_list_pth = '/home/linhanxi/github/actionKnowledgeXfewshot/few-shot-video-classification/data/kinetics100/data_splits/k100_classes.txt'
        else:
            raise ValueError
        labelTexts = join_multiple_txts(clsName_list_pth)
        self.add_labelTexts(list(labelTexts))

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def select_labelTexts(self, labelTexts):
        tem = []
        for labelText in labelTexts:
            tem.append(self.textfeats4labelTexts[self.labelTexts.index(labelText)])
        self.register_buffer('cur_textfeats4labelTexts', torch.stack(tem, dim=0))
        # self.cur_textfeats4labelTexts = torch.stack(tem, dim=0)


class scaled_cosine_sim(nn.Module):
    def __init__(self, visual_arch, enabled_grad_flow=False, is_softmax=True):
        super().__init__()
        model, preprocess = clip.load(visual_arch, device='cpu')
        self.scale = model.logit_scale.detach().cpu().exp()
        del model, preprocess
        self.enabled_grad_flow = enabled_grad_flow
        self.is_softmax = is_softmax

    def __call__(self, aa, bb):
        with torch.set_grad_enabled(self.enabled_grad_flow):
            if not isinstance(aa, torch.Tensor):
                aa=torch.tensor(aa)
                bb=torch.tensor(bb)
            if self.scale.dtype!=torch.float16:
                aa = aa.float()
                bb = bb.float()
            if not self.enabled_grad_flow:
                aa /= aa.norm(dim=-1, keepdim=True)
                bb /= bb.norm(dim=-1, keepdim=True)
            else:
                aa = aa / aa.norm(dim=-1, keepdim=True)
                bb = bb / bb.norm(dim=-1, keepdim=True)
            sim_mat = (self.scale* aa @ bb.T)
            if self.is_softmax:
                probs = torch.nn.functional.softmax(sim_mat, dim=-1)
            else:
                probs = sim_mat
            return probs


class fake_module(nn.Module):
    def __init__(self, the_very_return) -> None:
        super().__init__()
        self.the_very_return = the_very_return

    def forward(self, *pargs, **kargs):
        return self.the_very_return


if __name__=='__main__':
    ins = r2plus1d_w_knowledge(args)
