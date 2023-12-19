import os
from dataset import NocsDataset
from torch.utils.data import DataLoader
from test_utils import collate_fn, data_prefetcher, geodesic_distance, o6d2mat, qua2mat
import torch
from torch import nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
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

    def forward(self, x):
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
        return torch.cat(out, -1)

class Model(nn.Module):
    def __init__(self, mode='matrix'):
        super().__init__()
        self.L = 9
        self.N = 3
        self.embedding = Embedding(self.N, self.L)
        self.transformerlayer = nn.TransformerEncoderLayer(self.N * (self.L * 2 + 1), 1)
        self.transformer = nn.TransformerEncoder(self.transformerlayer, 4)

        self.inner_size = 128
        self.out_channel1 = 9
        self.out_channel2 = 4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.N * (self.L * 2 + 1)))
        self.linear1 = nn.Linear(2, 128)
        self.linear2 = nn.Linear(128, 16)
        self.translation_head = nn.Linear(self.inner_size, 3)
        self.mode = mode
        if self.mode == 'matrix':
            self.rotation_head = nn.Linear(self.inner_size, 9)
        elif self.mode == 'quat':
            self.rotation_head = nn.Linear(self.inner_size, 4)
        elif self.mode == '6d':
            self.rotation_head = nn.Linear(self.inner_size, 6)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            nn.Linear(self.N * (self.L * 2 + 1), self.inner_size),
            nn.LeakyReLU(),
            nn.Linear(self.inner_size, self.inner_size)
        )

    def convert2matrix(self, x):
        if self.mode == 'matrix':
            matrix = x.view(-1, 3, 3)
        elif self.mode == 'quat':
            matrix = qua2mat(x)
        elif self.mode == '6d':
            matrix = o6d2mat(x)
        return matrix

    def forward(self, batch_x2d, batch_x3d):
        # [2d, 3d]拼接
        x = torch.cat((batch_x2d, batch_x3d), dim=1)
        # x = batch_x3d
        num_samples = 1024 # 序列太长了，采样
        last_dim_size = x.size(-1)
        samples = torch.randint(last_dim_size, size=(num_samples,))
        x = x[:, :, samples]
        x = x.permute(2, 0, 1)
        x = self.embedding(x)
        # 拼接cls_token,输入transformer
        cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        y = self.transformer(torch.cat((cls_token, x)))
        y = self.mlp(y[0, :, :])
        pred_trans = self.translation_head(y)
        pred_rot = self.convert2matrix(self.rotation_head(y))
        return pred_trans, pred_rot


device = 'cuda'
net = Model(mode='6d').to(device)

dataset_path = '/data1/renjl/nocs/data'
data_path = ['real/train', 'real/test', 'camera/train', 'camera/val']
annpath = '../'
real_train_set = NocsDataset(os.path.join(dataset_path, data_path[0]), annpath=annpath)
real_test_set = NocsDataset(os.path.join(dataset_path, data_path[1]), annpath=annpath)
real_train_loader = DataLoader(real_train_set, batch_size=32, collate_fn=collate_fn, drop_last=True)
real_test_loader = DataLoader(real_test_set, batch_size=32, collate_fn=collate_fn, drop_last=True)

L2 = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)

for epoch in range(3):
    prefetcher = data_prefetcher(real_test_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    for i in range(len(real_test_loader)):
        # segmentation, nocs_map, pose = targets['segnentaion'], targets['nocs_map'], targets['pose']
        batch_x2d = []
        batch_x3d = []
        batch_pose = []
        for target in targets:
            masks, nocs_map, poses, ids = target['segmentation'], target['nocs_map'], target['pose'], target['category_id']
            nocs_map = nocs_map.permute(2, 1, 0)
            for mask, pose, idx in zip(masks, poses, ids):
                if idx in [1, 3, 4, 5]:
                    continue
                rows, cols = torch.where(mask == 1)
                x3d = nocs_map[:, cols, rows]
                x2d = torch.stack((rows, cols))
                batch_x2d.append(x2d)
                batch_x3d.append(x3d)
                batch_pose.append(pose)

        max_l = max(tensor.size(1) for tensor in batch_x2d)
        batch_x2d = [F.pad(tensor, (0, max_l - tensor.size(1)), mode='constant', value=0) for tensor in batch_x2d]
        batch_x3d = [F.pad(tensor, (0, max_l - tensor.size(1)), mode='constant', value=0) for tensor in batch_x3d]
        batch_x2d = torch.stack(batch_x2d, dim=0).to(device)
        batch_x3d = torch.stack(batch_x3d, dim=0).to(device)
        pred_trans, pred_rot = net(batch_x2d, batch_x3d)

        gt_pose = torch.stack(batch_pose, dim=0)
        gt_trans, gt_rot = gt_pose[:, :3, 3], gt_pose[:, :3, :3]
        # loss = L2(gt_trans, pred_trans)
        loss = geodesic_distance(gt_rot, pred_rot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch{epoch + 1}:', loss, i)

        samples, targets = prefetcher.next()

torch.save(net, 'test.pt')
