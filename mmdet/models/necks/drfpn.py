import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS

class SRB(nn.Module):
    def __init__(self, inplane, outplane):
        super(SRB, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.deconv = nn.ConvTranspose2d(outplane, outplane, 3, stride = 2, padding = 1, output_padding = 1)
        self.offset_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)
        
        self.spatial_weight = nn.Sequential(
                                nn.Conv2d(outplane*2, outplane, kernel_size=1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(outplane, 1, kernel_size=3, padding=1)
                            )

    def offset_warp(self, input, offset, size):
        #input => original high_feature
        out_h, out_w = size # size of low_feature
        n, c, h, w = input.size() #size of high_feature

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device) #[1, 1, 1, 2]
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w) # [out_h, out,w]
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1) #[out_w, out_h]
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2) #[out_h, out_w, 2]
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device) #[batch, out_h, out_w, 2]
        
        grid = grid + offset.permute(0, 2, 3, 1) / norm
        
        output = F.grid_sample(input, grid)

        return output
    def forward(self, low_feature, h_feature):
        h_feature_origin = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = self.deconv(h_feature)
        offset = self.offset_make(torch.cat([h_feature, low_feature], 1))
        weight = self.spatial_weight(torch.cat([h_feature, low_feature], 1))
        h_feature_gen = self.offset_warp(h_feature_origin, offset, size=size)
        
        
        h_feature_gen = h_feature_gen*weight
        h_feature_origin = F.interpolate(h_feature_origin, size=size, mode='nearest')
        h_feature_gen = h_feature_gen + h_feature_origin
        return h_feature_gen
   
class CRB(nn.Module): # top-down
    def __init__(self,
                supervise_channels,
                input_channels,
                output_channels,
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=False)):
        super(CRB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.supervise_conv1x1 = ConvModule(
                supervise_channels,
                output_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
        self.input_conv3x3 = ConvModule(
                input_channels,
                output_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

    def forward(self, supervise_feats, input_feats):
        channel_attention_weight = self.supervise_conv1x1(self.avg_pool(supervise_feats))
        input_feats = self.input_conv3x3(input_feats)
        output = input_feats * channel_attention_weight
        return output

class PSPModule(nn.Module):
    def __init__(self, in_channel, out_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.bottleneck = nn.Conv2d(out_channel * (len(sizes) + 1), out_channel, kernel_size=1)
        self.relu = nn.ReLU()  
        self.reduceDim_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
 
    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)
 
    def forward(self, feats):
        feats = self.reduceDim_conv(feats)
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

@NECKS.register_module()
class DRFPN(nn.Module):
    """
    Dual Refinement Feature Pyramid Network.

    This is an implementation of - Dual Refinement Feature Pyramid Networks for Object
    Detection 

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(DRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.backbone_end_level):
            
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            
            self.fpn_convs.append(fpn_conv)

        self.srb_list = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1):
            self.srb_list.append(
                SRB(inplane=self.out_channels, outplane=self.out_channels//2)
            )
            
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    input_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    input_channels = out_channels
                extra_fpn_conv = ConvModule(
                    input_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        #add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        #for i in range(self.start_level + 1, self.backbone_end_level):
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        # channel attention pathway
        self.crb_list = nn.ModuleList() #ga means global attention
        for i in range(self.start_level, self.backbone_end_level-1):
            ca = CRB(out_channels, out_channels, out_channels)
            self.crb_list.append(ca)

        self.PPM_head = PSPModule(in_channels[-1], out_channels)
        #init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.crb_list.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.srb_list.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.PPM_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.in_channels)-1-self.start_level)
        ]
        laterals.append(self.PPM_head(inputs[-1]))
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self.srb_list[i-1](laterals[i-1],laterals[i])

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        # bu1_top_featuremap = inter_outs[used_backbone_levels-1] # top featuremap for global attention
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = self.downsample_convs[i](inter_outs[i]) + self.crb_list[i](inter_outs[i], inter_outs[i + 1])
        
        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
