import torch
from torch import nn

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


class Mkpts_Reg_Model(nn.Module):
    def __init__(self, num_sample: int, mode: str='matrix'):
        super().__init__()
        self.pts_size = 2
        self.N_freqs = 9
        self.embedding = Embedding(in_channels=self.pts_size, N_freqs=self.N_freqs, logscale=False)
        self.transformerlayer = nn.TransformerEncoderLayer(d_model=2 * self.pts_size * (2 * self.N_freqs + 1), nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformerlayer, num_layers=4)

        self.inner_size = 64
        self.out_channel1 = 9
        self.out_channel2 = 4

        self.translation_head = nn.Linear(in_features=self.inner_size, out_features=3)

        self.mode = mode
        if self.mode == 'matrix':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=9)
        elif self.mode == 'quat':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=4)
        elif self.mode == '6d':
            self.rotation_head = nn.Linear(in_features=self.inner_size, out_features=6)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=num_sample * 2 * self.pts_size * (2 * self.N_freqs + 1),
                      out_features=num_sample * (2 * self.N_freqs + 1)),
            nn.LeakyReLU(),
            nn.Linear(in_features=num_sample * (2 * self.N_freqs + 1),
                      out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.inner_size),
            nn.Linear(in_features=self.inner_size, out_features=self.inner_size),
            nn.LeakyReLU(),
        )

    def convert2matrix(self, x: torch.tensor):
        if self.mode == 'matrix':
            matrix = x.view(x.shape[0], 3, 3)
        elif self.mode == 'quat':
            matrix = qua2mat(x)
        elif self.mode == '6d':
            matrix = o6d2mat(x)
        return matrix

    def forward(self, batch_mkpts0: torch.tensor, batch_mkpts1: torch.tensor):
        x = torch.cat((batch_mkpts0, batch_mkpts1), dim=-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.reshape(x.shape[0], -1)
        y = self.mlp(x)
        pred_trans = self.translation_head(y)
        pred_rot = self.convert2matrix(self.rotation_head(y))
        return pred_trans, pred_rot
