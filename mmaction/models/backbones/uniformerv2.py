# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
import torch.nn.functional as F
from mmengine.model.weight_init import trunc_normal_
from mmaction.registry import MODELS
from einops import rearrange
torch.use_deterministic_algorithms(True, warn_only=True)

logger = MMLogger.get_current_instance()

MODEL_PATH = 'https://download.openmmlab.com/mmaction/v1.0/recognition'
_MODELS = {
    'ViT-B/16':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-base-p16-res224_clip-rgb_20221219-b8a5da86.pth'),
    'ViT-L/14':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-large-p14-res224_clip-rgb_20221219-9de7543e.pth'),
    'ViT-L/14_336':
    os.path.join(MODEL_PATH, 'uniformerv2/clipVisualEncoder',
                 'vit-large-p14-res336_clip-rgb_20221219-d370f9e5.pth'),
}



class QuickGELU(BaseModule):
    """Quick GELU function. Forked from https://github.com/openai/CLIP/blob/d50
    d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py.

    Args:
        x (torch.Tensor): The input features of shape :math:`(B, N, C)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class MultiScaleFireAttention(BaseModule):
    """增强型多尺度火焰烟雾检测模块
    Args:
        d_model (int): 输入通道数
        dw_reduction (float): 通道压缩率，默认1.5
        pos_kernel_sizes (List[int]): 多尺度卷积核尺寸，默认[3,5,7]
        init_cfg: 权重初始化配置
    """

    def __init__(
            self,
            d_model: int,
            dw_reduction: float = 1.5,
            pos_kernel_sizes: List[int] = [3, 5, 7],
            init_cfg: Optional[dict] = None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.re_d_model = int(d_model // dw_reduction)

        # 多尺度深度卷积分支
        self.ms_conv = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm3d(d_model),
                nn.Conv3d(d_model, self.re_d_model, kernel_size=1),
                nn.Conv3d(
                    self.re_d_model, self.re_d_model,
                    kernel_size=(k, 1, 1), stride=1,
                    padding=(k // 2, 0, 0), groups=self.re_d_model
                ),
                nn.Conv3d(self.re_d_model, d_model, kernel_size=1)
            ) for k in pos_kernel_sizes
        ])

        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv3d(d_model, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(d_model, d_model // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(d_model // 8, d_model, kernel_size=1),
            nn.Sigmoid()
        )

        # 残差连接
        self.residual_conv = nn.Conv3d(d_model, d_model, kernel_size=1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for conv in self.ms_conv:
            nn.init.constant_(conv[3].weight, 0)
            nn.init.constant_(conv[3].bias, 0)
        nn.init.xavier_uniform_(self.spatial_att[0].weight)
        nn.init.xavier_uniform_(self.channel_att[1].weight)
        nn.init.xavier_uniform_(self.channel_att[3].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        multi_scale_feats = [conv(x) for conv in self.ms_conv]
        # 特征融合（元素级最大值融合增强小目标检测）
        fused_feat = torch.stack(multi_scale_feats).max(dim=0)[0]

        # 空间注意力增强
        spatial_weights = self.spatial_att(fused_feat)
        fused_feat = fused_feat * spatial_weights

        # 通道注意力调制
        channel_weights = self.channel_att(fused_feat)
        fused_feat = fused_feat * channel_weights

        # 残差连接（保留原始特征）
        return fused_feat + self.residual_conv(x)

class Local_MHRA(BaseModule):
    """Local MHRA.

    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
            self,
            d_model: int,
            dw_reduction: float = 1.5,
            pos_kernel_size: int = 3,
            init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )
        # 替换 nn.AdaptiveAvgPool3d 为自定义确定性池化模块
        self.spp = nn.ModuleList([
            nn.AdaptiveAvgPool3d(pool_size)
            for pool_size in [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
        ])
        # 通道注意力模块
        self.channel_att = nn.Sequential(
            nn.Conv3d(d_model * 4, d_model, kernel_size=1),
            nn.Sigmoid()
        )

        # init zero
        logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_feat = self.pos_embed(x)
        # 多尺度特征提取（使用自定义池化模块，逻辑与原代码一致）
        pooled_feats = [F.interpolate(pool(x), size=x.shape[2:]) for pool in self.spp]
        # 特征融合增强
        fused_feat = torch.cat([base_feat] + pooled_feats, dim=1)
        att_weights = self.channel_att(fused_feat)
        return base_feat * att_weights


class ResidualAttentionBlock(BaseModule):
    """Local UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA.
            Defaults to True.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
        dw_reduction: float = 1.5,
        no_lmhra: bool = False,
        double_lmhra: bool = True,

        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        logger.info(f'No L_MHRA: {no_lmhra}')
        logger.info(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = MultiScaleFireAttention(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)


        # self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        # self.S_Adapter = Adapter(d_model)
        # self.T_Adapter = Adapter(d_model)
        #self.mona_Adapter = Mona(in_dim=d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor, T: int = 8) -> torch.Tensor:
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)



        # MHSA

        x = x + self.drop_path(self.attention(self.ln_1(x)))


        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        xn = self.ln_2(x)
        x = x + self.drop_path(self.mlp(xn))
        return x


class Extractor(BaseModule):
    """Global UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        drop_out (float): Stochastic dropout rate.
            Defaults to 0.0.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_factor: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(BaseModule):
    """Backbone:

    Args:
        width (int): Number of input channels in local UniBlock.
        layers (int): Number of layers of local UniBlock.
        heads (int): Number of attention head in local UniBlock.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA
            in local UniBlock. Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        dw_reduction: float = 1.5,
        no_lmhra: bool = True,
        double_lmhra: bool = False,
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,

        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [
            x.item()
            for x in torch.linspace(0, backbone_drop_path_rate, layers)
        ]
        self.resblocks = ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=False if i in {8} else no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])

        # global block
        assert n_layers == len(return_list)
        #self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.dpe = ModuleList([
            nn.Conv3d(
                n_dim,
                n_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=n_dim) for _ in range(n_layers)
        ])
        for m in self.dpe:
            nn.init.constant_(m.bias, 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dec = ModuleList([
            Extractor(
                n_dim,
                n_head,
                mlp_factor=mlp_factor,
                dropout=mlp_dropout[i],
                drop_path=dpr[i],
            ) for i in range(n_layers)
        ])
        # weight sum
        self.norm = nn.LayerNorm(n_dim)
        # self.balance = nn.Parameter(torch.zeros((n_dim)))
        # self.sigmoid = nn.Sigmoid()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1)**0.5)
        #cls_token = self.temporal_cls_token.repeat(1, N, 1)


        j = -1
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2,
                                              0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats.clone()).view(
                    N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        #weight = self.sigmoid(self.balance)
        residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
        out = self.norm(residual)
        return out



@MODELS.register_module()
class UniFormerV2(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = False,

        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        # pretrain
        clip_pretrained: bool = True,
        pretrained: Optional[str] = None,
        frozen_stages: int = -1,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)


        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )

    #     self._freeze_stages()
    #
    # def _freeze_stages(self) -> None:
    #     """
    #     根据 self.frozen_stages 冻结骨干网络前几层：
    #     - >=0：冻结 conv1 和 ln_pre
    #     - >=1：冻结 transformer.resblocks 中的前 frozen_stages 层
    #     """
    #     # 冻结 patch embed + ln_pre
    #     if self.frozen_stages >= 0:
    #         self.conv1.eval()
    #         self.ln_pre.eval()
    #         self.class_embedding.requires_grad_(False)
    #         self.positional_embedding.requires_grad_(False)
    #         for dpe in self.transformer.dpe:
    #             dpe.eval()
    #             for param in dpe.parameters():
    #                 param.requires_grad = False
    #         for param in self.conv1.parameters():
    #             param.requires_grad = False
    #         for param in self.ln_pre.parameters():
    #             param.requires_grad = False
    #         for extractor in self.transformer.dec:
    #             extractor.eval()
    #             for name, param in extractor.named_parameters():
    #                 if 'Adapter' in name:
    #                     continue
    #                 param.requires_grad = False
    #     # 冻结 transformer 的若干层
    #     if self.frozen_stages >= 1:
    #         for idx, block in enumerate(self.transformer.resblocks):
    #             if idx < self.frozen_stages:
    #                 block.eval()
    #                 for name, param in block.named_parameters():
    #                     if  'Adapter' in name or 'SEA' in name or 'TEA' in name:
    #                         continue
    #                     param.requires_grad = False
    #
    #             else:
    #                 break
    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def _load_pretrained(self, pretrained: str = None) -> None:
        """Load CLIP pretrained visual encoder.

        The visual encoder is extracted from CLIP.
        https://github.com/openai/CLIP

        Args:
            pretrained (str): Model name of pretrained CLIP visual encoder.
                Defaults to None.
        """
        assert pretrained is not None, \
            'please specify clip pretraied checkpoint'

        model_path = _MODELS[pretrained]
        logger.info(f'Load CLIP pretrained model from {model_path}')
        state_dict = _load_checkpoint(model_path, map_location='cpu')
        state_dict_3d = self.state_dict()
        for k in state_dict.keys():
            if k in state_dict_3d.keys(
            ) and state_dict[k].shape != state_dict_3d[k].shape:
                if len(state_dict_3d[k].shape) <= 2:
                    logger.info(f'Ignore: {k}')
                    continue
                logger.info(f'Inflate: {k}, {state_dict[k].shape}' +
                            f' => {state_dict_3d[k].shape}')
                time_dim = state_dict_3d[k].shape[2]
                state_dict[k] = self._inflate_weight(state_dict[k], time_dim)
        self.load_state_dict(state_dict, strict=False)

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x)
        return out