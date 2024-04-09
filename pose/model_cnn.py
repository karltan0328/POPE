import torch
from torch import nn
from collections import OrderedDict
import math

from timm.models.layers import trunc_normal_
from pose.convnextv2 import convnextv2
from pose.utils import qua2mat, o6d2mat


class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = convnextv2.convnextv2_large(num_classes=num_classes)
        checkpoint = torch.load('weights/convnextv2_large_22k_384_ema.pt')
        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
                'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = self.remap_checkpoint_keys(checkpoint_model)
        self.load_state_dict(checkpoint_model)

        # manually initialize fc layer
        trunc_normal_(self.model.head.weight, std=2e-5)
        torch.nn.init.constant_(self.model.head.bias, 0.)

        # 冻结参数
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model.head.weight.requires_grad = True
        # self.model.head.bias.requires_grad = True

    def remap_checkpoint_keys(self, checkpoint: dict):
        new_ckpt = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('encoder'):
                k = '.'.join(k.split('.')[1:]) # remove encoder in the name
            if k.endswith('kernel'):
                k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
                new_k = k + '.weight'
                if len(v.shape) == 3: # resahpe standard convolution
                    kv, in_dim, out_dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(2, 1, 0).\
                        reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif len(v.shape) == 2: # reshape depthwise convolution
                    kv, dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(1, 0).\
                        reshape(dim, 1, ks, ks).transpose(3, 2)
                continue
            elif 'ln' in k or 'linear' in k:
                k = k.split('.')
                k.pop(-2) # remove ln and linear in the name
                new_k = '.'.join(k)
            else:
                new_k = k
            new_ckpt[new_k] = v

        # reshape grn affine parameters and biases
        for k, v in new_ckpt.items():
            if k.endswith('bias') and len(v.shape) != 1:
                new_ckpt[k] = v.reshape(-1)
            elif 'grn' in k:
                new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
        return new_ckpt

    def load_state_dict(self, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self.model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                self.model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                self.model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                self.model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))

    def forward(self, x: torch.tensor):
        return self.model(x)

class Mkpts_Reg_Model(nn.Module):
    def __init__(self, num_sample: int, mode: str='6d'):
        super().__init__()

        self.inner_size = 32

        self.convnextv2 = ConvNeXtV2(num_classes=self.inner_size)

        self.mode = mode
        # self.translation_head_mkpts = nn.Linear(in_features=self.inner_size, out_features=3)
        self.translation_head_rot = nn.Linear(in_features=self.inner_size, out_features=3)
        if self.mode == 'matrix':
            # self.rotation_head_mkpts = nn.Linear(in_features=self.inner_size, out_features=9)
            self.rotation_head_rot = nn.Linear(in_features=self.inner_size, out_features=9)
        elif self.mode == 'quat':
            # self.rotation_head_mkpts = nn.Linear(in_features=self.inner_size, out_features=4)
            self.rotation_head_rot = nn.Linear(in_features=self.inner_size, out_features=4)
        elif self.mode == '6d':
            # self.rotation_head_mkpts = nn.Linear(in_features=self.inner_size, out_features=6)
            self.rotation_head_rot = nn.Linear(in_features=self.inner_size, out_features=6)

    def convert2matrix(self, x: torch.tensor):
        if self.mode == 'matrix':
            matrix = x.view(x.shape[0], 3, 3)
        elif self.mode == 'quat':
            matrix = qua2mat(x)
        elif self.mode == '6d':
            matrix = o6d2mat(x)
        return matrix

    def forward(self,
                batch_mkpts0: torch.tensor,
                batch_mkpts1: torch.tensor,
                batch_img0: torch.tensor,
                batch_img1: torch.tensor):
        # x_mkpts = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
        # x_mkpts = self.transformer(x_mkpts)
        # x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], -1)
        # x_mkpts = self.mlp(x_mkpts)
        # print(x_mkpts.shape) # (B, self.inner_size)

        x_img = torch.cat((batch_img0, batch_img1), dim=-1)
        x_img = self.convnextv2(x_img)
        # print(x_img.shape) # (B, self.inner_size)

        # pred_trans_mkpts = self.translation_head_mkpts(x_mkpts)
        # pred_rot_mkpts = self.convert2matrix(self.rotation_head_mkpts(x_mkpts))
        # print(pred_trans_mkpts.shape, pred_rot_mkpts.shape) # (B, 3), (B, 3, 3)

        pred_trans_img = self.translation_head_rot(x_img)
        pred_rot_img = self.convert2matrix(self.rotation_head_rot(x_img))
        # print(pred_trans_img.shape, pred_rot_img.shape) # (B, 3), (B, 3, 3)

        # pred_trans = (pred_trans_mkpts + pred_trans_img) / 2
        # pred_rot = (pred_rot_mkpts + pred_rot_img) / 2
        # return pred_trans, pred_rot
        return pred_trans_img, pred_rot_img
