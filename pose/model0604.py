import torch
from torch import nn
from collections import OrderedDict
import math
from torch.nn import functional as F
import typing

from pose.convnextv2 import convnextv2
from pose.utils import qua2mat, o6d2mat, remap_checkpoint_keys, load_state_dict, _get_activation_fn

class Embedding(nn.Module):
    def __init__(self, in_channels: int, N_freqs: int, logscale: bool=True):
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

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes: int=1000, model_type: str='base'):
        """
        num_classes: fixed
        model_type: paramble
        """
        super().__init__()
        self.model = convnextv2.__dict__['convnextv2_' + model_type]()

        # load checkpoint
        if model_type == 'huge':
            pt_name = 'convnextv2_huge_22k_384_ema.pt'
        elif model_type == 'large':
            pt_name = 'convnextv2_large_22k_384_ema.pt'
        elif model_type == 'base':
            pt_name = 'convnextv2_base_22k_384_ema.pt'
        elif model_type == 'tiny':
            pt_name = 'convnextv2_tiny_22k_384_ema.pt'
            pass
        elif model_type == 'nano':
            pt_name = 'convnextv2_nano_22k_384_ema.pt'
            pass
        else:
            pass
        try:
            checkpoint = torch.load('/home2/mingxintan/POPE/weights/' + pt_name, map_location='cpu')
        except:
            checkpoint = torch.load('/root/autodl-tmp/POPE/weights/' + pt_name, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = self.model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
                'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(self.model, checkpoint_model, prefix='')

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)
        # torch.nn.init.constant_(model.head.bias, 0.)

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.tensor):
        return self.model(x)

# class Transformer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         """
#         all param fixed
#         """
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)

#     def forward(self,
#                 src: torch.Tensor,
#                 tgt: torch.Tensor,
#                 src_mask: typing.Optional[torch.Tensor] = None,
#                 tgt_mask: typing.Optional[torch.Tensor] = None,
#                 memory_key_padding_mask: typing.Optional[torch.Tensor] = None,
#                 tgt_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
#         r"""Pass the input through the cross-attention encoder layer.

#         Args:
#             src: the sequence to the encoder layer (required).
#             tgt: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             tgt_mask: the mask for the tgt sequence (optional).
#             memory_key_padding_mask: the mask for the src keys per batch (optional).
#             tgt_key_padding_mask: the mask for the tgt keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         tgt2 = self.self_attn(tgt, src, src, attn_mask=tgt_mask,
#                               key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         return tgt

class MoCoPE(nn.Module):
    def __init__(self, num_sample: int, mode: str='6d', cnn_model_type: str='base', train_type: str='mkpts+cnn'):
        """
        num_sample: paramble
        mode: paramble
        cnn_model_type: paramble
        train_type: paramble
        """
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
                      out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

        # CNN
        self.cnn_model_type = cnn_model_type
        self.convnextv2 = ConvNeXtV2(model_type=self.cnn_model_type)
        self.mlp_img = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )

        # Fusion
        self.mkpts_as_q = nn.Transformer(d_model=512, nhead=1)
        self.cnn_as_q = nn.Transformer(d_model=512, nhead=1)

        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=2048,
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
        if self.train_type == 'mkpts+cnn':
            x_mkpts = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
            x_mkpts = self.transformer_mkpts(x_mkpts, x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], -1)
            x_mkpts = self.mlp1(x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], 2, -1)

            x_img0 = self.convnextv2(batch_img0)
            x_img1 = self.convnextv2(batch_img1)
            x_img0 = self.mlp_img(x_img0).unsqueeze(1)
            x_img1 = self.mlp_img(x_img1).unsqueeze(1)
            x_img = torch.cat((x_img0, x_img1), dim=1)

            qmkpts = self.mkpts_as_q(x_img, x_mkpts)
            qimg = self.cnn_as_q(x_mkpts, x_img)

            x = torch.cat((qmkpts, qimg), dim=-1)
        elif self.train_type == 'mkpts':
            x_mkpts = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
            x_mkpts = self.transformer_mkpts(x_mkpts, x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], -1)
            x_mkpts = self.mlp1(x_mkpts)
            x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], 2, -1)

            qmkpts = self.mkpts_as_q(x_mkpts, x_mkpts)

            x = torch.cat((qmkpts, qmkpts), dim=-1)
        elif self.train_type == 'cnn':
            x_img0 = self.convnextv2(batch_img0)
            x_img1 = self.convnextv2(batch_img1)
            x_img0 = self.mlp_img(x_img0).unsqueeze(1)
            x_img1 = self.mlp_img(x_img1).unsqueeze(1)
            x_img = torch.cat((x_img0, x_img1), dim=1)

            qimg = self.cnn_as_q(x_img, x_img)

            x = torch.cat((qimg, qimg), dim=-1)
        else:
            pass

        x = x.reshape(x.shape[0], -1)

        output = self.mlp2(x)

        pred_trans = self.translation_head(output)
        pred_rot = self.convert2matrix(self.rotation_head(output))

        return pred_trans, pred_rot
