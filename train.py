# %%
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn

from pose.dataset import pose_dataset
from pose.utils import collate_fn, geodesic_distance
from pose.model import Mkpts_Reg_Model


if os.name == 'nt':
    LM_dataset_path = 'd:/git_project/POPE/data/LM_dataset/'
    LM_dataset_json_path = 'd:/git_project/POPE/data/pairs/LINEMOD-test.json'
    LM_dataset_points_path = 'd:/git_project/POPE/data/LM_dataset-points/'

    onepose_path = 'e:/datasets/OnePose/test_data/'
    onepose_json_path = 'd:/git_project/POPE/data/pairs/Onepose-test.json'
    onepose_points_path = 'd:/git_project/POPE/data/onepose-points/'

    oneposeplusplus_path = 'e:/datasets/OnePose++/lowtexture_test_data/'
    oneposeplusplus_json_path = 'd:/git_project/POPE/data/pairs/OneposePlusPlus-test.json'
    oneposeplusplus_points_path = 'd:/git_project/POPE/data/oneposeplusplus-points/'
elif os.name == 'posix':
    LM_dataset_path = 'data/LM_dataset/'
    LM_dataset_json_path = 'data/pairs/LINEMOD-test.json'
    LM_dataset_points_path = 'data/LM_dataset-points/'

    onepose_path = 'data/onepose/'
    onepose_json_path = 'data/pairs/Onepose-test.json'
    onepose_points_path = 'data/onepose-points/'

    oneposeplusplus_path = 'data/oneposeplusplus/'
    oneposeplusplus_json_path = 'data/pairs/OneposePlusPlus-test.json'
    oneposeplusplus_points_path = 'data/oneposeplusplus-points/'

paths = [
    ('linemod', LM_dataset_path, LM_dataset_json_path, LM_dataset_points_path),
    ('onepose', onepose_path, onepose_json_path, onepose_points_path),
    ('oneposeplusplus', oneposeplusplus_path, oneposeplusplus_json_path, oneposeplusplus_points_path)
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = pose_dataset(paths)
mkpts_max_len, mkpts_sum_len = dataset.get_mkpts_info()


random.seed(20231223)
torch.manual_seed(20231223)
torch.cuda.manual_seed(20231223)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

num_sample = 400
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False, collate_fn=collate_fn(num_sample))
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False, collate_fn=collate_fn(num_sample))

# %%
net = Mkpts_Reg_Model(num_sample=num_sample, mode='6d').to(device)
net.train()

L2 = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6, weight_decay=1e-5)

# %%
num_epochs = 5

for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        # if i == 1: break

        batch_K0 = []
        batch_K1 = []
        batch_pose0 = []
        batch_pose1 = []
        batch_mkpts0 = []
        batch_mkpts1 = []
        batch_pre_K = []
        for data in batch:
            batch_K0.append(data['K0'])
            batch_K1.append(data['K1'])
            # print(data['pose0'].shape)
            if data['pose0'].shape[0] == 3:
                data['pose0'] = np.vstack((data['pose0'], np.array([0, 0, 0, 1])))
            if data['pose1'].shape[0] == 3:
                data['pose1'] = np.vstack((data['pose1'], np.array([0, 0, 0, 1])))
            batch_pose0.append(data['pose0'])
            batch_pose1.append(data['pose1'])
            batch_mkpts0.append(data['mkpts0'])
            batch_mkpts1.append(data['mkpts1'])
            batch_pre_K.append(data['pre_K'])
        batch_K0 = torch.from_numpy(np.stack(batch_K0, axis=0)).float().to(device)
        batch_K1 = torch.from_numpy(np.stack(batch_K1, axis=0)).float().to(device)
        batch_pose0 = torch.from_numpy(np.stack(batch_pose0, axis=0)).float().to(device)
        batch_pose1 = torch.from_numpy(np.stack(batch_pose1, axis=0)).float().to(device)
        batch_mkpts0 = torch.from_numpy(np.stack(batch_mkpts0, axis=0)).float().to(device)
        batch_mkpts1 = torch.from_numpy(np.stack(batch_mkpts1, axis=0)).float().to(device)
        batch_pre_K = torch.from_numpy(np.stack(batch_pre_K, axis=0)).float().to(device)

        batch_relative_pose = torch.matmul(batch_pose1, batch_pose0.permute(0, 2, 1))

        pre_t, pre_rot = net(batch_mkpts0, batch_mkpts1)

        gt_t = batch_relative_pose[:, :3, 3]
        gt_rot = batch_relative_pose[:, :3, :3]

        t_loss = L2(gt_t, pre_t)
        rot_loss = geodesic_distance(gt_rot, pre_rot)

        loss = 15 * t_loss + rot_loss # t_loss与rot_loss的数量级不一样，需要调整一下权重
        # print(t_loss.sum(), rot_loss.sum())
        # loss = t_loss
        # loss = rot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'epoch: {epoch + 1}, batch: {i + 1}, t_loss: {15 * t_loss.item()}, rot_loss: {rot_loss.item()}, loss: {loss.item()}')

# %%
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
torch.save(net, f'./weights/{net.mode}-{dt_string}-{loss:.4f}.pth')

# %%



