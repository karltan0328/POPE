import os
import torch
import random
import numpy as np
import pandas as pd
import argparse
from torch import nn
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from torch.utils.data import DataLoader

from pose.dataset import pose_dataset
from pose.utils import (
    collate_fn,
    geodesic_distance,
    relative_pose_error,
    relative_pose_error_np,
    recall_object,
    aggregate_metrics
)
from pose.model0606 import MoCoPE
from pose.animator import Animator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=list)
parser.add_argument('--ckp_name', required=True, type=str)
parser.add_argument('--num_sample', default=500, type=int, help='mkpts sample num')
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
ckp_name = args.ckp_name
num_sample = args.num_sample
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

net = torch.load('./weights/' + ckp_name).to(device)
net.eval()

id2name_dict = {
    1: "ape",
    2: "benchvise",
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

ycbv_dict = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
}

# linemod
ape_data = []
benchvise_data = []
camera_data = []
can_data = []
cat_data = []
driller_data = []
duck_data = []
eggbox_data = []
glue_data = []
holepuncher_data = []
iron_data = []
lamp_data = []
phone_data = []
# onepose
aptamil_data = []
jzhg_data = []
minipuff_data = []
hlyormosiapie_data = []
brownhouse_data = []
oreo_data = []
mfmilkcake_data = []
diycookies_data = []
taipingcookies_data = []
tee_data = []
# onepose++
toyrobot_data = []
yellowduck_data = []
sheep_data = []
fakebanana_data = []
teabox_data = []
orange_data = []
greenteapot_data = []
lecreusetcup_data = []
insta_data = []
# ycbv
one_data = []
two_data = []
three_data = []
four_data = []
five_data = []
six_data = []
seven_data = []
eight_data = []
nine_data = []
ten_data = []

all_data = {
    # linemod
    'ape_data': ape_data,
    'benchvise_data': benchvise_data,
    'camera_data': camera_data,
    'can_data': can_data,
    'cat_data': cat_data,
    'driller_data': driller_data,
    'duck_data': duck_data,
    'eggbox_data': eggbox_data,
    'glue_data': glue_data,
    'holepuncher_data': holepuncher_data,
    'iron_data': iron_data,
    'lamp_data': lamp_data,
    'phone_data': phone_data,
    # onepose
    'aptamil_data': aptamil_data,
    'jzhg_data': jzhg_data,
    'minipuff_data': minipuff_data,
    'hlyormosiapie_data': hlyormosiapie_data,
    'brownhouse_data': brownhouse_data,
    'oreo_data': oreo_data,
    'mfmilkcake_data': mfmilkcake_data,
    'diycookies_data': diycookies_data,
    'taipingcookies_data': taipingcookies_data,
    'tee_data': tee_data,
    # onepose++
    'toyrobot_data': toyrobot_data,
    'yellowduck_data': yellowduck_data,
    'sheep_data': sheep_data,
    'fakebanana_data': fakebanana_data,
    'teabox_data': teabox_data,
    'orange_data': orange_data,
    'greenteapot_data': greenteapot_data,
    'lecreusetcup_data': lecreusetcup_data,
    'insta_data': insta_data,
    # ycbv
    'one_data': one_data,
    'two_data': two_data,
    'three_data': three_data,
    'four_data': four_data,
    'five_data': five_data,
    'six_data': six_data,
    'seven_data': seven_data,
    'eight_data': eight_data,
    'nine_data': nine_data,
    'ten_data': ten_data,
}

linemod_type = ['ape_data', 'benchvise_data', 'camera_data', 'can_data', 'cat_data', 'driller_data', 'duck_data', 'eggbox_data', 'glue_data', 'holepuncher_data', 'iron_data', 'lamp_data', 'phone_data']
onepose_type = ['aptamil_data', 'jzhg_data', 'minipuff_data', 'hlyormosiapie_data', 'brownhouse_data', 'oreo_data', 'mfmilkcake_data', 'diycookies_data', 'taipingcookies_data', 'tee_data']
oneposeplusplus_type = ['toyrobot_data', 'yellowduck_data', 'sheep_data', 'fakebanana_data', 'teabox_data', 'orange_data', 'greenteapot_data', 'lecreusetcup_data', 'insta_data']
ycbv_type = ['one_data', 'two_data', 'three_data', 'four_data', 'five_data', 'six_data', 'seven_data', 'eight_data', 'nine_data', 'ten_data']

for i, batch in enumerate(test_dataloader):
    for data in batch:
        if 'lm' in data['name']:
            all_data[f"{id2name_dict[int(data['name'][2:])]}_data"].append(data)
        else:
            if data['name'] in ('12345678910'):
                all_data[f"{ycbv_dict[int(data['name'])]}_data"].append(data)
            else:
                all_data[f"{data['name']}_data"].append(data)

empty_keys = []
for key in all_data.keys():
    if len(all_data[key]) == 0:
        empty_keys.append(key)

for key in empty_keys:
    all_data.pop(key)

for key in all_data.keys():
    print(key, len(all_data[key]))

print('len(all_data):', len(all_data))

res_table = []

for key in all_data.keys():
    if key in linemod_type:
        logger.info(f"LINEMOD: {key}")
    elif key in onepose_type:
        logger.info(f"ONEPOSE: {key}")
    elif key in oneposeplusplus_type:
        logger.info(f"ONEPOSE++: {key}")
    elif key in ycbv_type:
        logger.info(f"YCBV: {key}")
    metrics = dict()
    metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})
    recall_image, all_image = 0, 0
    for item in all_data[key]:
        all_image += 1
        K0 = item['K0']
        K1 = item['K1']
        pose0 = item['pose0']
        pose1 = item['pose1']
        pre_bbox = item['pre_bbox']
        gt_bbox = item['gt_bbox']
        mkpts0 = item['mkpts0']
        mkpts1 = item['mkpts1']
        pre_K = item['pre_K']
        img0 = item['img0']
        img1 = item['img1']
        name = item['name']
        pair_name = item['pair_name']
        # linemod
        if 'lm' in name:
            name = id2name_dict[int(name[2:])]
        # ycbv
        if name in '12345678910':
            name = ycbv_dict[int(name)]

        if name not in key:
            print(f'name: {name}, key: {key}')
            continue

        is_recalled = recall_object(pre_bbox, gt_bbox)

        recall_image = recall_image + int(is_recalled > 0.5)

        batch_mkpts0 = torch.from_numpy(mkpts0).unsqueeze(0).float().to(device)
        batch_mkpts1 = torch.from_numpy(mkpts1).unsqueeze(0).float().to(device)
        img0 = torch.from_numpy(img0).unsqueeze(0).float().to(device)
        img1 = torch.from_numpy(img1).unsqueeze(0).float().to(device)
        img0 = img0.permute(0, 3, 2, 1)
        img1 = img1.permute(0, 3, 2, 1)
        pre_t, pre_rot = net(batch_mkpts0, batch_mkpts1, img0, img1)
        pre_t = pre_t.squeeze(0).detach().cpu().numpy()
        pre_rot = pre_rot.squeeze(0).detach().cpu().numpy()

        relative_pose = np.matmul(pose1, np.linalg.inv(pose0))
        gt_pose = np.zeros_like(pose1)
        gt_pose[:3, :3] = relative_pose[:3, :3]
        gt_pose[:3, 3] = pose1[:3, 3]
        t_err, R_err = relative_pose_error_np(gt_pose, pre_rot, pre_t, ignore_gt_t_thr=0.0)

        metrics['R_errs'].append(R_err)
        metrics['t_errs'].append(t_err)
        metrics['identifiers'].append(pair_name)

    print(f"Acc: {recall_image}/{all_image}")
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    val_metrics_4tb["AP50"] = recall_image / all_image
    # logger.info('\n' + pprint.pformat(val_metrics_4tb))
    res_table.append([f"{name}"] + list(val_metrics_4tb.values()))

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
for i, key in enumerate(all_data.keys()):
    if i == 1:
        break
    if key in linemod_type:
        print(f"LINEMOD")
    elif key in onepose_type:
        print(f"ONEPOSE")
    elif key in oneposeplusplus_type:
        print(f"ONEPOSE++")
    elif key in ycbv_type:
        print(f"YCBV")

headers = ["Category"] + list(val_metrics_4tb.keys())
all_data = np.array(res_table)[:, 1:].astype(np.float32)
res_table.append(["Avg"] + all_data.mean(0).tolist())
print(tabulate(res_table, headers=headers, tablefmt='fancy_grid'))

df = pd.DataFrame(res_table, columns=headers)
df_rounded = df.round(6)
df_rounded.to_excel(f"{path[0][0]}-res.xlsx", index=False)
