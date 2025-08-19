import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(current_dir, '../../../'))

models_mamba_dir = os.path.join(parent_dir, 'VSSD_Backbone', 'VSSD', 'models')
if models_mamba_dir not in sys.path:
    sys.path.insert(0, models_mamba_dir)

decoder_mamba_dir = os.path.join(parent_dir, 'mamba_decoder', 'models')
if decoder_mamba_dir not in sys.path:
    sys.path.insert(0, decoder_mamba_dir)

from mamba2 import Backbone_VMAMBA2
from mamba2 import MambaCrossAttention
from MTMamba import MTMamba

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
import torch.utils.checkpoint as cp
from einops import rearrange,repeat

@SEM_SEG_HEADS_REGISTRY.register()
class OpenMambaHead(nn.Module):

    @configurable
    def __init__(
        self,
        clip_model_name,
        mask_in_chans: int,
        num_channels: int,
        use_checkpoint: bool,
        num_output_maps: int,
        vssd: Backbone_VMAMBA2,
        mca: MambaCrossAttention,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        if '_base' in clip_model_name:
            clip_dim = 640
        elif '_large' in clip_model_name:
            clip_dim = 768
        
        self.fuse = nn.Conv2d(clip_dim, num_channels, 1)
        
        self.vssd = vssd

        self.mca = mca

        self.decoder = MTMamba()
        self.norm = nn.LayerNorm(384)
        self.final = nn.Conv2d(384, num_output_maps, 1)
        
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, clip_dim, kernel_size=1),
        )
        

    @classmethod
    def from_config(cls, cfg):

        vssd_kwargs = Backbone_VMAMBA2.from_config(cfg)
        vssd_module = Backbone_VMAMBA2(**vssd_kwargs)

        mca_kwargs = MambaCrossAttention.from_config(cfg)
        mca_module = MambaCrossAttention(**mca_kwargs)

        return {
            "clip_model_name": cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME,
            "mask_in_chans": cfg.MODEL.OPEN_MAMBA.MASK_IN_CHANNELS,
            "num_channels": cfg.MODEL.OPEN_MAMBA.NUM_CHANNELS,
            "use_checkpoint": cfg.MODEL.OPEN_MAMBA.USE_CHECKPOINT,
            "num_output_maps": cfg.MODEL.OPEN_MAMBA.NUM_OUTPUT_MAPS,
            "vssd": vssd_module,
            "mca": mca_module,
        }

    def forward(self, clip_feature, masks, text_features, training_flag):

        N = masks.size(1)
        B = masks.size(0)
        masks = rearrange(masks, 'B N H W -> (B N) H W').unsqueeze(dim=1)
        text_features = repeat(text_features, 't c -> (b n) t c', n=N, b=B)
        
        clip_feature = repeat(clip_feature, "B C H W -> (B N) C H W", N=N)
        
        H,W = clip_feature.shape[-2:]

        masks = F.interpolate(masks.float(), size=(H*4,W*4),
                                                mode='bilinear', align_corners=False)
        masks = self.mask_downscaling(masks)
        
        outputs = clip_feature + masks

        def _inner_forward(outputs, text_features):
            outputs = self.fuse(outputs)
            
            outputs = F.interpolate(outputs.float(), size=(32, 32), mode='bilinear', align_corners=False)

            # applying the VSSD backbone, reference: https://github.com/YuHengsss/VSSD
            outputs = self.vssd(outputs)

            # MCA module
            outputs = self.mca(outputs, text_features)

            # mamba based decoder
            outputs = self.decoder(outputs)

            outputs = outputs.permute(0, 2, 3, 1) 
            outputs = self.norm(outputs.contiguous())
            outputs = outputs.permute(0, 3, 1, 2) 
            
            outputs = self.final(outputs.contiguous())

            outputs = rearrange(outputs, '(B N) C H W -> B (N C) H W',N=N)
    
            return outputs

        if self.use_checkpoint and self.training:
            outputs = cp.checkpoint(_inner_forward, outputs,use_reentrant=False)
        else:
            outputs = _inner_forward(outputs, text_features)
        return outputs

def build_open_mamba(cfg,name):
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg)

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x