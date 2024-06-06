import os
import torch
import random
import numpy as np
import argparse
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader

from pose.dataset import pose_dataset
from pose.utils import collate_fn, geodesic_distance, relative_pose_error
from pose.model0606 import MoCoPE
from pose.animator import Animator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=list)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--num_sample', default=500, type=int, help='mkpts sample num')
parser.add_argument('--rot_mode', default='6d', type=str, help='rotation representation mode')
parser.add_argument('--vim_model_type', default='small', type=str, help='vim kind')
parser.add_argument('--train_type', default='mkpts+vim', type=str, help='net train mode')
parser.add_argument('--gpu_name', default='0', type=str)
args = parser.parse_args()

if os.name == 'nt':
    LM_dataset_path = 'd:/git_project/POPE/data/LM_dataset/'
    LM_dataset_json_path = 'd:/git_project/POPE/data/pairs/LINEMOD-test.json'
    LM_dataset_points_path = 'd:/git_project/POPE/data/LM_dataset-points/'

    onepose_path = 'd:/git_project/POPE/data/onepose/'
    onepose_json_path = 'd:/git_project/POPE/data/pairs/Onepose-test.json'
    onepose_points_path = 'd:/git_project/POPE/data/onepose-points/'

    onepose_plusplus_path = 'd:/git_project/POPE/data/onepose_plusplus/'
    onepose_plusplus_json_path = 'd:/git_project/POPE/data/pairs/OneposePlusPlus-test.json'
    onepose_plusplus_points_path = 'd:/git_project/POPE/data/onepose_plusplus-points/'

    ycbv_path = 'd:/git_project/POPE/data/ycbv/'
    ycbv_json_path = 'd:/git_project/POPE/data/pairs/YCB-VIDEO-test.json'
    ycbv_points_path = 'd:/git_project/POPE/data/ycbv-points'
elif os.name == 'posix':
    LM_dataset_path = 'data/LM_dataset/'
    LM_dataset_json_path = 'data/pairs/LINEMOD-test.json'
    LM_dataset_points_path = 'data/LM_dataset-points/'

    onepose_path = 'data/onepose/'
    onepose_json_path = 'data/pairs/Onepose-test.json'
    onepose_points_path = 'data/onepose-points/'

    onepose_plusplus_path = 'data/onepose_plusplus/'
    onepose_plusplus_json_path = 'data/pairs/OneposePlusPlus-test.json'
    onepose_plusplus_points_path = 'data/onepose_plusplus-points/'

    ycbv_path = 'data/ycbv/'
    ycbv_json_path = 'data/pairs/YCB-VIDEO-test.json'
    ycbv_points_path = 'data/ycbv-points'

paths = [
    ('linemod', LM_dataset_path, LM_dataset_json_path, LM_dataset_points_path),
    ('onepose', onepose_path, onepose_json_path, onepose_points_path),
    ('onepose_plusplus', onepose_plusplus_path, onepose_plusplus_json_path, onepose_plusplus_points_path),
    ('ycbv', ycbv_path, ycbv_json_path, ycbv_points_path),
]

dataset_idx = args.dataset
# print(args.dataset)
num_epochs = args.num_epochs
num_sample = args.num_sample
rot_mode = args.rot_mode
vim_model_type = args.vim_model_type
train_type = args.train_type
device = 'cuda:' + args.gpu_name

path = []
for idx in dataset_idx:
    # print(paths[int(idx)])
    path.append(paths[int(idx)])
# print(path)
dataset = pose_dataset(path)

random.seed(20231223)
torch.manual_seed(20231223)
torch.cuda.manual_seed(20231223)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn(num_sample))
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn(num_sample))

net = MoCoPE(num_sample=num_sample, mode=rot_mode, vim_model_type=vim_model_type, train_type=train_type).to(device)
net.train()

L2 = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6, weight_decay=1e-4)

start_time = datetime.now()

for epoch in range(1, num_epochs + 1):
    for i, batch in enumerate(train_dataloader):
        # if i == 1: break

        batch_K0 = []
        batch_K1 = []
        batch_pose0 = []
        batch_pose1 = []
        batch_mkpts0 = []
        batch_mkpts1 = []
        batch_pre_K = []
        batch_img0 = []
        batch_img1 = []

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
            batch_img0.append(data['img0'])
            batch_img1.append(data['img1'])

        # batch_K0 = torch.from_numpy(np.stack(batch_K0, axis=0)).float().to(device)
        # batch_K1 = torch.from_numpy(np.stack(batch_K1, axis=0)).float().to(device)

        batch_pose0 = np.stack(batch_pose0, axis=0)
        batch_pose1 = np.stack(batch_pose1, axis=0)
        batch_relative_pose = np.matmul(batch_pose1, np.linalg.inv(batch_pose0))
        batch_relative_pose = torch.from_numpy(batch_relative_pose).float()
        batch_pose1 = torch.from_numpy(batch_pose1).float()


        gt_rot = batch_relative_pose[:, :3, :3].to(device)
        gt_t = batch_pose1[:, :3, 3].to(device)

        batch_mkpts0 = torch.from_numpy(np.stack(batch_mkpts0, axis=0)).float().to(device)
        batch_mkpts1 = torch.from_numpy(np.stack(batch_mkpts1, axis=0)).float().to(device)

        # batch_pre_K = torch.from_numpy(np.stack(batch_pre_K, axis=0)).float().to(device)

        batch_img0 = torch.from_numpy(np.stack(batch_img0, axis=0)).float()
        batch_img1 = torch.from_numpy(np.stack(batch_img1, axis=0)).float()
        batch_img0 = batch_img0.permute(0, 3, 2, 1).to(device)
        batch_img1 = batch_img1.permute(0, 3, 2, 1).to(device)

        pre_t, pre_rot = net(batch_mkpts0, batch_mkpts1, batch_img0, batch_img1)

        t_loss = L2(gt_t, pre_t)
        rot_loss = geodesic_distance(gt_rot, pre_rot)

        loss = 10 * t_loss + rot_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataloader) // 5) == 0 or i == len(train_dataloader) - 1:
            # animator.add(epoch + (i + 1) / len(train_dataloader),
            #             (t_loss.item(), rot_loss.item()))
            print(f'epoch: {epoch}, t_loss: {t_loss.item()}, rot_loss: {rot_loss.item()}, loss: {loss.item()}')

end_time = datetime.now()
print(f'cost time: {(end_time - start_time).total_seconds() / 3600} h')

now_time = datetime.now()

ckpts_name = f"{paths[0][0]}-{now_time.strftime('%Y-%m-%d-%H')}-{train_type}-{cnn_model_type}-{num_sample}-{num_epochs}-{loss:.4f}.pth"
torch.save(net, f'./weight/{ckpts_name}')

print(ckpts_name)
