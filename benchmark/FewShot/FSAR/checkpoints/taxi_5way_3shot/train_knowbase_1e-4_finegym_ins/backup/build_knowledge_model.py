import torch
import torch.nn as nn
from models import r2plus1d
from ipdb import set_trace
import clip
from utils import AverageMeter, calculate_accuracy, check_which_nan, check_tensor_nan
from temporal_modeling import bottleneck_CNN
from lin_utils import backup_files, join_multiple_txts
import random
from vivit import Transformer
from module import transpose_axis, squeezer, residual_cw_block
from partshift_conv import ShiftModule
from torch.distributions.utils import probs_to_logits, logits_to_probs

def calc_entropy(logits):
    probs = logits_to_probs(logits)
    p_log_p = logits * probs
    return -p_log_p.sum(-1)


class knowledge_model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.reduce_ratio = 1 if self.args.ablation_onlyCLIPvisfea else 64
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.proposals_fea = torch.load(self.args.proposals_fea_pth, map_location=self.device).to(torch.float)
        if not self.args.ablation_onlyCLIPvisfea:
            self.args.match_result_dim = self.proposals_fea.size(0)
            if self.args.embeddin_bias:
                self.embeddin_bias = nn.Parameter(torch.zeros_like(self.proposals_fea))
        else:
            self.args.match_result_dim = 512

        '''
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
            for i in range(self.args.n_finetune_classes):
                torch.nn.init.constant_(self.new_fc.weight[i,self.original_dim+i], 1)
            self.add_labelTexts_dataset(args.dataset)
        '''
        # set_trace()
        # if not self.args.is_w_knowledge:
        #     self.prepare_for_ablation()
        self.hidden_dim = self.args.match_result_dim//self.reduce_ratio
        self.scaled_cosine_simer = scaled_cosine_sim(self.args.CLIP_visual_arch, enabled_grad_flow=args.grad_enabled_in_match, is_softmax=False)
        self.with_clip_zeroshot = self.args.with_clip_zeroshot
        if self.with_clip_zeroshot:
            self.way_to_use_zeroshot = self.args.way_to_use_zeroshot
            self.add_labelTexts_dataset(args.dataset)
            self.fused_predictor = self.predictor_basedon_general_knowledge_and_zeroshot()
            self.scaled_cosine_simer_zs = scaled_cosine_sim(self.args.CLIP_visual_arch, enabled_grad_flow=args.grad_enabled_in_match, is_softmax=True)
        self.build_general_knowledge_modeling()
        self.build_specific_knowledge_modeling()
        self.build_cls_head()

    def build_base_model(self):
        raise NotImplementedError

    def build_general_knowledge_modeling(self):
        reduce_ratio = self.reduce_ratio
        t_ksize = 3
        # set_trace()

        if self.args.ablation_onlyLinear:
            self.general_knowledge_model = nn.Sequential(
            nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim//reduce_ratio, 1),
            nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
            ).to(self.device)
        else:
            self.general_knowledge_model = nn.Sequential(
                #nn.Dropout(0.9),
                # nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim, t_ksize, padding=t_ksize//2, groups=self.args.match_result_dim),
                nn.BatchNorm1d(self.args.match_result_dim),
                nn.Dropout(0.05),
                nn.Conv1d(self.args.match_result_dim, self.args.match_result_dim//reduce_ratio, 1),
                nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.args.match_result_dim//reduce_ratio, self.args.match_result_dim//reduce_ratio, t_ksize, padding=t_ksize//2, groups=self.args.match_result_dim//reduce_ratio),
                nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.args.match_result_dim//reduce_ratio, self.args.match_result_dim//reduce_ratio, t_ksize, padding=t_ksize//2, groups=self.args.match_result_dim//reduce_ratio),
                nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
                nn.ReLU(inplace=True),
                #residual_cw_block(self.args.match_result_dim,reduce_ratio,t_ksize),
                ## nn.AdaptiveAvgPool1d(1)
                transpose_axis((1,2)),
                Transformer(self.args.match_result_dim//reduce_ratio, 1, 3, 64, self.args.match_result_dim//reduce_ratio*4, 0.),
                # nn.Dropout(0.05),
                transpose_axis((1,2)),
            ).to(self.device)

    def build_specific_knowledge_modeling(self):
        reduce_ratio = self.reduce_ratio
        t_ksize = 3
        if self.args.ablation_onlyLinear:
            self.specific_knowledge_model = nn.Identity()
        else:
            self.specific_knowledge_model = nn.Sequential(
                nn.Conv1d(self.args.match_result_dim//reduce_ratio, self.args.match_result_dim//reduce_ratio, t_ksize, padding=t_ksize//2, groups=self.args.match_result_dim//reduce_ratio),
                nn.BatchNorm1d(self.args.match_result_dim//reduce_ratio),
                nn.ReLU(inplace=True),
                nn.Dropout(0.05)
                ).to(self.device)

    def build_cls_head(self):
        reduce_ratio = self.reduce_ratio
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.args.match_result_dim//reduce_ratio, self.args.n_finetune_classes, 1),
            squeezer()
            )

    def build_cls_head_4CLIPzs(self):
        reduce_ratio = self.reduce_ratio
        zeroshot_pre_size = self.cur_textfeats4labelTexts.size(0)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.args.match_result_dim//reduce_ratio+zeroshot_pre_size, self.args.n_finetune_classes, 1),
            squeezer()
            ).to(self.device)

    def predictor_basedon_general_knowledge_and_zeroshot(self):
        if self.way_to_use_zeroshot=='naive_sum':
            class naive_sum(nn.Module):
                def __init__(self, cls_head) -> None:
                    super().__init__()
                    self.cls_head = cls_head
                def forward(self, general_knowledge, specific_knowledge, zeroshot_logits):
                    # set_trace()
                    return self.cls_head(specific_knowledge).squeeze() + zeroshot_logits
            return naive_sum(self.cls_head)
        elif self.way_to_use_zeroshot=='stacking':
            class stacker(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                def forward(self, general_knowledge, specific_knowledge, zeroshot_logits):
                    # set_trace()
                    stacked = torch.cat([specific_knowledge.mean(-1,keepdim=True), zeroshot_logits.mean(-1,keepdim=True)], 1)
                    return stacked
            return stacker()
        elif self.way_to_use_zeroshot=='adaptive_fuse':
            class adaptive_fusor(nn.Module):
                def __init__(self, cls_head, in_planes) -> None:
                    super().__init__()
                    self.cls_head = cls_head
                    self.gate_generator = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Conv1d(in_planes, 1, 1)
                    )
                def forward(self, general_knowledge, specific_knowledge, zeroshot_logits):
                    gate = self.gate_generator(general_knowledge).squeeze(-1).sigmoid()
                    # set_trace()
                    return gate*(self.cls_head(specific_knowledge).squeeze()) + (1-gate)*zeroshot_logits
            return adaptive_fusor(self.cls_head, self.args.match_result_dim//self.reduce_ratio)
        elif self.way_to_use_zeroshot=='adaptive_fuseV2':
            # knowledge and entropy-based adaptive fuse
            class adaptive_fusorV2(nn.Module):
                def __init__(self, cls_head, in_planes) -> None:
                    super().__init__()
                    self.cls_head = cls_head
                    self.gate_generator = nn.Sequential(
                        nn.AdaptiveAvgPool1d(1),
                        nn.Conv1d(in_planes+1, 1, 1)
                    )
                def forward(self, general_knowledge, specific_knowledge, zeroshot_logits):
                    knowledge_logits = self.cls_head(specific_knowledge).squeeze()
                    bs,t = general_knowledge.size(0), general_knowledge.size(-1)
                    entropys = calc_entropy(knowledge_logits)[:,None,None].expand(bs,1,t)
                    # set_trace()
                    gate = self.gate_generator(torch.cat((general_knowledge,entropys), dim=1)).squeeze(-1).sigmoid()
                    return gate*knowledge_logits + (1-gate)*zeroshot_logits
            return adaptive_fusorV2(self.cls_head, self.args.match_result_dim//self.reduce_ratio )
        else:
            raise

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
            set_trace()
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

    def forward(self, input, CLIP_vis_branch):
        if not self.args.ablation_onlyCLIPvisfea:
            if self.args.embeddin_bias:
               match = self.scaled_cosine_simer(CLIP_vis_branch, self.proposals_fea+self.embeddin_bias).transpose(-1,-2).contiguous() 
            else:
                match = self.scaled_cosine_simer(CLIP_vis_branch, self.proposals_fea).transpose(-1,-2).contiguous()
        else:
            match = CLIP_vis_branch.transpose(-1,-2)

        # if 'FLOPs' in self.args.result_path:
        #     from lin_utils import flops
        #     from fvcore.nn import FlopCountAnalysis
        #     from ipdb import set_trace; set_trace()
        #     FlopCountAnalysis(self.general_knowledge_model, match).total() + FlopCountAnalysis(self.specific_knowledge_model, general_knowledge).total()

        general_knowledge = self.general_knowledge_model(match)
        specific_knowledge = self.specific_knowledge_model(general_knowledge)

        if 'FLOPs' in self.args.result_path:
            from thop import profile
            from ipdb import set_trace; set_trace()
            flops1, params1 = profile(self.general_knowledge_model, (match, ))
            flops2, params2 = profile(self.specific_knowledge_model, (general_knowledge, ))

        
        if self.with_clip_zeroshot:
            zeroshot_logits = self.clip_zeroshot(CLIP_vis_branch)
            # set_trace()
            cls_input = self.fused_predictor(general_knowledge, specific_knowledge, zeroshot_logits)
        else:
            cls_input = specific_knowledge
        logits = self.cls_head(cls_input)
        # set_trace()
        return logits

    def clip_zeroshot(self, CLIP_vis_branch):
        # set_trace()
        zeroshot_logits = self.scaled_cosine_simer_zs(CLIP_vis_branch, self.cur_textfeats4labelTexts).transpose(-1,-2).contiguous()
        return zeroshot_logits

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
        #set_trace()
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
        general_knowledge = self.general_knowledge_model(match)
        specific_knowledge = self.specific_knowledge_model(general_knowledge)
        if self.with_clip_zeroshot:
            zeroshot_logits = self.clip_zeroshot(CLIP_vis_branch)
            # set_trace()
            logits = self.fused_predictor(general_knowledge, specific_knowledge, zeroshot_logits)
        else:
            logits = self.cls_head(specific_knowledge).squeeze()
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
        elif dataset in ['diving48V2']:
            clsName_list_pth = '/home/linhanxi/diving48_240p/Diving48_vocab.json'
            import json
            loaded = json.load(open(clsName_list_pth,'rb'))
            clsName_list_pth = [" ".join(name) for name in loaded]
        else:
            raise ValueError(f'{dataset}')
        labelTexts = join_multiple_txts(clsName_list_pth)
        self.labelTexts = labelTexts
        self.add_labelTexts(list(labelTexts))

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def select_labelTexts(self, labelTexts):
        self.cur_labelTexts = labelTexts
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
