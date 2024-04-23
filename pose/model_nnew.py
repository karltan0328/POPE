import torch
from torch import nn
from collections import OrderedDict
import math
from torch.nn import functional as F
import typing

from pose.convnextv2 import convnextv2
from pose.utils import qua2mat, o6d2mat

class Embedding(nn.Module):
    def __init__(self, in_channels: int, N_freqs: int, logscale: bool=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
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
    def __init__(self, num_classes: int=1000):
        super().__init__()
        self.model = convnextv2.convnextv2_base(num_classes=num_classes)
        checkpoint = torch.load('/root/autodl-tmp/POPE/weights/convnextv2_base_22k_384_ema.pt')
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

        # 冻结参数
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def remap_checkpoint_keys(self, checkpoint: dict):
        new_ckpt = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('encoder'):
                k = '.'.join(k.split('.')[1:]) # remove encoder in the name
            if k.endswith('kernel'):
                k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
                neWk = k + '.weight'
                if len(v.shape) == 3: # resahpe standard convolution
                    kv, in_dim, out_dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[neWk] = v.permute(2, 1, 0).\
                        reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif len(v.shape) == 2: # reshape depthwise convolution
                    kv, dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[neWk] = v.permute(1, 0).\
                        reshape(dim, 1, ks, ks).transpose(3, 2)
                continue
            elif 'ln' in k or 'linear' in k:
                k = k.split('.')
                k.pop(-2) # remove ln and linear in the name
                neWk = '.'.join(k)
            else:
                neWk = k
            new_ckpt[neWk] = v

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

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: typing.Optional[torch.Tensor] = None,
                tgt_mask: typing.Optional[torch.Tensor] = None,
                memory_key_padding_mask: typing.Optional[torch.Tensor] = None,
                tgt_key_padding_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the cross-attention encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            tgt: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_key_padding_mask: the mask for the src keys per batch (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, src, src, attn_mask=tgt_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class Mkpts_Reg_Model(nn.Module):
    def __init__(self, num_sample: int, mode: str='6d'):
        super().__init__()
        self.pts_size = 2
        self.N_freqs = 9
        self.inner_size = 32
        self.embedding = Embedding(in_channels=self.pts_size,
                                   N_freqs=self.N_freqs,
                                   logscale=False)
        self.transformer_mkpts = Transformer(d_model=2 * self.pts_size * (2 * self.N_freqs + 1), nhead=4)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features=2 * self.pts_size * (2 * self.N_freqs + 1) * 500,
                      out_features=2 * (2 * self.N_freqs + 1) * 500),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=2 * (2 * self.N_freqs + 1) * 500,
                      out_features=2000),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
        )
        self.convnextv2 = ConvNeXtV2()
        self.transformer_imgs = Transformer(d_model=1000, nhead=4)
        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=2000,
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
        x_mkpts = self.embedding(torch.cat((batch_mkpts0, batch_mkpts1), dim=-1))
        x_mkpts = self.transformer_mkpts(x_mkpts, x_mkpts)
        # print(x_mkpts.shape)
        x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], -1)
        x_mkpts = self.mlp1(x_mkpts)
        x_mkpts = x_mkpts.reshape(x_mkpts.shape[0], 2, 1000)

        x_img0 = self.convnextv2(batch_img0).unsqueeze(1)
        x_img1 = self.convnextv2(batch_img1).unsqueeze(1)
        x_img = torch.cat((x_img0, x_img1), dim=1)

        # output = self.transformer_imgs(x_img, x_mkpts) # onepose 0.672908
        output = self.transformer_imgs(x_mkpts, x_img) # onepose 0.733294
        output = output.reshape(output.shape[0], -1)

        output = self.mlp2(output)

        pred_trans = self.translation_head(output)
        pred_rot = self.convert2matrix(self.rotation_head(output))
        return pred_trans, pred_rot


if __name__ == '__main__':
    batch_mkpts0 = torch.randn(8, 500, 2)
    batch_mkpts1 = torch.randn(8, 500, 2)
    batch_img0 = torch.randn(8, 3, 256, 256)
    batch_img1 = torch.randn(8, 3, 256, 256)
    model = Mkpts_Reg_Model(num_sample=500, mode='6d')
    pred_trans, pred_rot = model(batch_mkpts0, batch_mkpts1, batch_img0, batch_img1)
    print(pred_trans.shape, pred_rot.shape) # [8, 3], [8, 3, 3]
