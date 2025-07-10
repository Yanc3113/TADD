import clip
from ipdb import set_trace
import torch
from models.temporal_shift import make_temporal_shift


class TSM(torch.nn.Module):
    def __init__(self, num_class, num_segments, vis_arch_name, shift_div, opt, freeze_backbone=False):
        super(TSM, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.num_class = num_class
        self.num_segments = num_segments
        self.shift_div = shift_div
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        backbone, preprocess = clip.load(vis_arch_name, jit=False, device=torch.device("cpu"))
        self.preprocess = preprocess
        self.base_model = backbone.visual
        self.base_model.to(device=self.device)
        if opt.shift:
            make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place='blockres', temporal_pool=False)
        self.new_fc = torch.nn.Linear(opt.emb_dim, num_class)
        self.new_fc.to(device=self.device)
        # self.base_model = torch.nn.Sequential(*[self.base_model, new_fc])
        # set_trace()
    
    def forward(self, input):
        # input: torch.Size([64, 3, 16, 112, 112]) => [bs, C, clip_duration, H, W]
        bs, C, clip_duration, H, W = input.size()
        # set_trace()
        assert clip_duration==self.num_segments, (clip_duration,self.num_segments)
        input = input.transpose(1,2).reshape(bs*clip_duration, C, H, W)
        if self.freeze_backbone:
            with torch.no_grad():
                output = self.base_model(input)
        else:
            output = self.base_model(input)
        # output: torch.Size([bs*clip_duration, C])
        # print(output.shape, self.new_fc)
        # print(output.dtype)
        # set_trace()
        output = self.new_fc(output)
        out_dim = output.size(-1)
        # set_trace()
        output = output.reshape(bs, clip_duration, out_dim).mean(1)
        return output

if __name__=='__main__':
    CLIP_TSM = TSM(64, 16, 'RN50', 8, None)
    input = torch.zeros(1,3,16,224,224).cuda()
    CLIP_TSM(input)
    set_trace()
