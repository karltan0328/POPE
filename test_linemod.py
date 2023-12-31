# %%
import os
import cv2
import json
import torch
import pprint
import numpy as np
from tqdm import tqdm
from loguru import logger
from tabulate import tabulate

from pose.utils import collate_fn, geodesic_distance, relative_pose_error, aggregate_metrics, recall_object, project_points

# %%
LM_id2name_dict = {
    1: 'ape',
    2: 'benchvise',
    4: 'camera',
    5: 'can',
    6: 'cat',
    8: 'driller',
    9: 'duck',
    10: 'eggbox',
    11: 'glue',
    12: 'holepuncher',
    13: 'iron',
    14: 'lamp',
    15: 'phone',
}

# %%
with open("data/pairs/LINEMOD-test.json") as f:
    dir_list = json.load(f)
len(dir_list)

# %%
ROOT_DIR = 'data/LM_dataset/'

# %%
res_table = []

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

num_sample = 400

net = torch.load('./weights/6d-2023-12-28-20-34-25-0.4200.pth').to(device)

net.eval()

# %%
for label_idx, test_dict in enumerate(dir_list):
    logger.info(f"LINEMOD: {label_idx + 1}/{len(dir_list)}")
    metrics = dict()
    metrics.update({'R_errs':[], 't_errs':[], 'inliers':[], "identifiers":[]})
    sample_data = dir_list[label_idx]["0"][0]
    label = sample_data.split("/")[0]
    name = label.split("-")[1]
    dir_name = os.path.dirname(sample_data)
    FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name)
    recall_image, all_image = 0, 0
    for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
        for pair_idx, pair_name in enumerate(tqdm(rotation_list)):
            all_image = all_image + 1
            base_name = os.path.basename(pair_name)
            idx0_name = base_name.split("-")[0]
            idx1_name = base_name.split("-")[1]
            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color_full"), idx1_name)

            K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
            K1_path = image1_name.replace("color_full", "intrin").replace("png", "txt")
            K0 = np.loadtxt(K0_path)
            K1 = np.loadtxt(K1_path)

            pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
            pose1_path = image1_name.replace("color_full", "poses_ba").replace("png", "txt")
            pose0 = np.loadtxt(pose0_path)
            pose1 = np.loadtxt(pose1_path)
            if pose0.shape[0] == 3:
                pose0 = np.concatenate([pose0, np.array([[0, 0, 0, 1]])], axis=0)
                pose1 = np.concatenate([pose1, np.array([[0, 0, 0, 1]])], axis=0)

            points_file_path = os.path.join('d:/git_project/POPE/data/LM_dataset-points/', pair_name.split("/")[0])
            pre_bbox_path = os.path.join(points_file_path, "pre_bbox")
            mkpts0_path = os.path.join(points_file_path, "mkpts0")
            mkpts1_path = os.path.join(points_file_path, "mkpts1")
            pre_K_path = os.path.join(points_file_path, "pre_K")
            points_name = pair_name.split("/")[-1]
            pre_bbox_path = os.path.join(pre_bbox_path, f'{points_name}.txt')
            mkpts0_path = os.path.join(mkpts0_path, f'{points_name}.txt')
            mkpts1_path = os.path.join(mkpts1_path, f'{points_name}.txt')
            pre_K_path = os.path.join(pre_K_path, f'{points_name}.txt')

            if not os.path.exists(pre_bbox_path):
                continue
            pre_bbox = np.loadtxt(pre_bbox_path)
            mkpts0 = np.loadtxt(mkpts0_path)
            mkpts1 = np.loadtxt(mkpts1_path)
            pre_K = np.loadtxt(pre_K_path)

            if mkpts0.shape[0] > num_sample:
                rand_idx = np.random.choice(mkpts0.shape[0], num_sample, replace=False)
                mkpts0 = mkpts0[rand_idx]
                mkpts1 = mkpts1[rand_idx]
            else:
                mkpts0 = np.concatenate([mkpts0, np.zeros((num_sample - mkpts0.shape[0], 2))], axis=0)
                mkpts1 = np.concatenate([mkpts1, np.zeros((num_sample - mkpts1.shape[0], 2))], axis=0)

            _3d_bbox = np.loadtxt(f"{os.path.join(ROOT_DIR, label)}/box3d_corners.txt")
            bbox_pts_3d, _ = project_points(_3d_bbox, pose1[:3, :4], K1)
            bbox_pts_3d = bbox_pts_3d.astype(np.int32)
            x0, y0, w, h = cv2.boundingRect(bbox_pts_3d)
            x1, y1 = x0 + w, y0 + h
            gt_bbox = np.array([x0, y0, x1, y1])
            is_recalled = recall_object(pre_bbox, gt_bbox)
            recall_image = recall_image + int(is_recalled > 0.5)

            batch_mkpts0 = torch.from_numpy(mkpts0).unsqueeze(0).float().to(device)
            batch_mkpts1 = torch.from_numpy(mkpts1).unsqueeze(0).float().to(device)
            pre_t, pre_rot = net(batch_mkpts0, batch_mkpts1)
            # pre_t = pre_t.cpu()
            # pre_rot = pre_rot.cpu()

            batch_pose0 = torch.from_numpy(pose0).unsqueeze(0).float().to(device)
            batch_pose1 = torch.from_numpy(pose1).unsqueeze(0).float().to(device)
            batch_relative_pose = torch.matmul(batch_pose1, batch_pose0.permute(0, 2, 1))
            t_err, R_err = relative_pose_error(batch_relative_pose, pre_rot, pre_t, ignore_gt_t_thr=0.0)

            metrics['t_errs'] = metrics['t_errs'] + np.array(t_err.reshape(-1).cpu().detach().numpy()).tolist()
            metrics['R_errs'] = metrics['R_errs'] + np.array(R_err.reshape(-1).cpu().detach().numpy()).tolist()
            metrics['identifiers'].append(pair_name)

    print(f'Acc: {recall_image}/{all_image}')
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    val_metrics_4tb['AP50'] = recall_image / all_image
    logger.info('\n' + pprint.pformat(val_metrics_4tb))

    obj_name = int(name[2:])

    res_table.append([f"{LM_id2name_dict[obj_name]}"] + list(val_metrics_4tb.values()))

# %%
headers = ["Category"] + list(val_metrics_4tb.keys())
all_data = np.array(res_table)[:, 1:].astype(np.float32)
res_table.append(["Avg"] + all_data.mean(0).tolist())
print(tabulate(res_table, headers=headers, tablefmt='fancy_grid'))

# %%



