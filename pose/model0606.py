import torch
from torch import nn
from collections import OrderedDict
import math
from torch.nn import functional as F
import typing

from timm.models import create_model
from pose.utils import qua2mat, o6d2mat

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from pose.vim.datasets import build_dataset
from pose.vim.engine import train_one_epoch, evaluate
from pose.vim.losses import DistillationLoss
from pose.vim.samplers import RASampler
from pose.vim.augment import new_data_aug_generator

from contextlib import suppress

import pose.vim.models_mamba

import pose.vim.utils

# log about
import mlflow

class Embedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 N_freqs: int,
                 logscale: bool=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        """
        all param fixed
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x: torch.tensor):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(tensors=out, dim=-1)

class Vim(nn.Module):
    def __init__(self,
                 num_class: int=1000,
                 model_type: str='small',
                 img_size: int=224):
        super().__init__()
        if model_type == 'small':
            vim_name = 'vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
            checkpoint_name = '/data1/tmx/POPE/weights/vim_s_midclstok_ft_81p6acc.pth'
        elif model_type == 'tiny':
            vim_name = 'vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
            checkpoint_name = '/data1/tmx/POPE/weights/vim_t_midclstok_ft_78p3acc.pth'
        else:
            pass
        self.model = create_model(vim_name,
                                  pretrained=False,
                                  num_classes=num_class,
                                  drop_rate=0.0,
                                  drop_path_rate=0.1,
                                  drop_block_rate=None,
                                  img_size=img_size)

        checkpoint = torch.load(checkpoint_name, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.model.patch_embed.num_patches
        num_extra_tokens = self.model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        self.model.load_state_dict(checkpoint_model, strict=False)


        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.tensor):
        return self.model(x)

class MoCoPE(nn.Module):
    def __init__(self,
                 num_sample: int=500,
                 mode: str='6d',
                 vim_model_type: type='small',
                 train_type: str='mkpts+vim'):
        super().__init__()
        self.pts_size = 2
        self.N_freqs = 9
        self.inner_size = 32
        self.train_type = train_type
        self.embedding = Embedding(in_channels=self.pts_size,
                                   N_freqs=self.N_freqs,
                                   logscale=False)
        self.transformer_mkpts = nn.Transformer(d_model=2 * self.pts_size * (2 * self.N_freqs + 1), nhead=1)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=2 * self.pts_size * (2 * self.N_freqs + 1) * num_sample,
                      out_features=2 * (2 * self.N_freqs + 1) * num_sample),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2 * (2 * self.N_freqs + 1) * num_sample,
                      out_features=2000),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

        # vim
        self.vim_model_type = vim_model_type
        self.vim = Vim()
        # self.mlp_img = nn.Sequential(
        #     nn.Linear(in_features=1000,
        #               out_features=512),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p=0.2),
        # )

        # Fusion
        self.mkpts_as_q = nn.Transformer(d_model=1000, nhead=1)
        self.vim_as_q = nn.Transformer(d_model=1000, nhead=1)

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=4000,
                    out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,
                    out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512,
                    out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256,
                    out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=128,
                    out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32,
                    out_features=32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )

        # Pose
        self.mode = mode
        self.translation_head = nn.Linear(in_features=self.inner_size, out_features=3)
        if self.mode == 'matrix':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=9)
        elif self.mode == 'quat':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=4)
        elif self.mode == '6d':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=6)
        else:
            pass

    def convert2matrix(self, x: torch.tensor):
        if self.mode == 'matrix':
            matrix = x.view(x.shape[0], 3, 3)
        elif self.mode == 'quat':
            matrix = qua2mat(x)
        elif self.mode == '6d':
            matrix = o6d2mat(x)
        else:
            pass
        return matrix

    def forward(self,
                batch_mkpts0: torch.tensor,
                batch_mkpts1: torch.tensor,
                batch_img0: torch.tensor,
                batch_img1: torch.tensor):
        if self.train_type == 'mkpts+vim':
            x_mkpts = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
            x_mkpts = self.transformer_mkpts(x_mkpts, x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], -1)
            x_mkpts = self.mlp1(x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], 2, -1)

            x_img0 = self.vim(batch_img0).unsqueeze(1)
            # x_img0 = self.mlp_img(x_img0)
            x_img1 = self.vim(batch_img1).unsqueeze(1)
            # x_img1 = self.mlp_img(x_img1).unsqueeze(1)
            x_img = torch.cat((x_img0, x_img1), dim=1)

            x_mkpts_as_q = self.mkpts_as_q(x_img, x_mkpts)
            x_vim_as_q = self.vim_as_q(x_mkpts, x_img)

            x = torch.cat((x_mkpts_as_q, x_vim_as_q), dim=-1)
        elif self.train_type == 'mkpts':
            pass
        elif self.train_type == 'vim':
            pass
        else:
            pass

        x = x.reshape(x.shape[0], -1)

        output = self.mlp2(x)

        pred_trans = self.translation_head(output)
        pred_rot = self.convert2matrix(self.rotation_head(output))

        return pred_trans, pred_rot
