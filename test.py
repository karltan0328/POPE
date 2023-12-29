# %%
import os
import torch
import pprint
import random
import numpy as np
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from pose.dataset import pose_dataset
from pose.utils import collate_fn, geodesic_distance, relative_pose_error, aggregate_metrics
from pose.model import Mkpts_Reg_Model

# %%
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
    # ('linemod', LM_dataset_path, LM_dataset_json_path, LM_dataset_points_path),
    # ('onepose', onepose_path, onepose_json_path, onepose_points_path),
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
net = torch.load('./weights/6d-2023-12-28-20-34-25-0.4200.pth')

net.eval()

L2 = nn.MSELoss()

# %%
metrics = dict()
metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})

res_table = []

# %%
for i, batch in enumerate(test_dataloader):
    if i == 1: break
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
    # print(batch_relative_pose.shape)

    pre_t, pre_rot = net(batch_mkpts0, batch_mkpts1)
    # print(pre_t.shape, pre_rot.shape)

    # 采用下面的代码进行训练
    # gt_t = batch_relative_pose[:, :3, 3]
    # gt_rot = batch_relative_pose[:, :3, :3]

    # t_loss = L2(gt_t, pre_t)
    # rot_loss = geodesic_distance(gt_rot, pre_rot)
    # print(t_loss, rot_loss)

    # loss = t_loss + rot_loss

    # 采用下面的代码进行测试
    t_err, R_err = relative_pose_error(batch_relative_pose, pre_rot, pre_t, ignore_gt_t_thr=0.0)
    # print(t_err.shape, R_err.shape)
    # print(t_err, R_err)
    # numpy写法
    metrics['t_errs'] = metrics['t_errs'] + np.array(t_err.reshape(-1).cpu().detach().numpy()).tolist()
    metrics['R_errs'] = metrics['R_errs'] + np.array(R_err.reshape(-1).cpu().detach().numpy()).tolist()
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)

    # R = pre_rot
    # R_gt = batch_relative_pose[:, :3, :3]
    # bmm = torch.bmm(R.permute(0, 2, 1), R_gt) # (batch, 3, 3)
    # bmm_trace = bmm.diagonal(dim1=1, dim2=2).sum(dim=-1).reshape(-1, 1) # (batch, 1)
    # cos = (bmm_trace - 1) / 2
    # print(cos)

# %%
logger.info('\n' + pprint.pformat(val_metrics_4tb))

# %%



