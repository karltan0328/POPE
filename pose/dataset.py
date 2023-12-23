import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class pose_dataset(Dataset):
    def __init__(self, json_paths:list):
        self.data = []
        for dataset_name, dataset_path, json_path, points_path in tqdm(json_paths):
            assert os.path.exists(dataset_path), f'{dataset_path} does not exist'
            assert os.path.exists(json_path), f'{json_path} does not exist'
            with open(json_path) as f:
                dir_list = json.load(f)
            for label_idx, dict in enumerate(dir_list):
                sample_data = dir_list[label_idx]["0"][0]
                label = sample_data.split("/")[0]
                name = label.split("-")[1]
                dir_name = os.path.dirname(sample_data)
                FULL_ROOT_DIR = os.path.join(dataset_path, dir_name)
                for rotation_key, rotation_list in zip(dict.keys(), dict.values()):
                    for pair_idx, pair_name in enumerate(rotation_list):
                        base_name = os.path.basename(pair_name)
                        idx0_name = base_name.split("-")[0]
                        idx1_name = base_name.split("-")[1]

                        if dataset_name == 'linemod':
                            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color_full"), idx1_name)
                        elif dataset_name == 'onepose' or dataset_name == 'oneposeplusplus':
                            image0_name = os.path.join(FULL_ROOT_DIR, idx0_name)
                            image1_name = os.path.join(FULL_ROOT_DIR.replace("color", "color"), idx1_name)

                        if dataset_name == 'linemod':
                            K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                            K1_path = image1_name.replace("color_full", "intrin").replace("png", "txt")
                        elif dataset_name == 'onepose' or dataset_name == 'oneposeplusplus':
                            K0_path = image0_name.replace("color", "intrin_ba").replace("png", "txt")
                            K1_path = image1_name.replace("color", "intrin_ba").replace("png", "txt")

                        if dataset_name == 'linemod':
                            pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                            pose1_path = image1_name.replace("color_full", "poses_ba").replace("png", "txt")
                        elif dataset_name == 'onepose' or dataset_name == 'oneposeplusplus':
                            pose0_path = image0_name.replace("color", "poses_ba").replace("png", "txt")
                            pose1_path = image1_name.replace("color", "poses_ba").replace("png", "txt")

                        points_file_path = os.path.join(points_path, pair_name.split('/')[0])
                        mkpts0_path = os.path.join(points_file_path, "mkpts0")
                        mkpts1_path = os.path.join(points_file_path, "mkpts1")
                        pre_K_path = os.path.join(points_file_path, "pre_K")
                        points_name = pair_name.split("/")[-1]
                        mkpts0_path = os.path.join(mkpts0_path, f'{points_name}.txt')
                        mkpts1_path = os.path.join(mkpts1_path, f'{points_name}.txt')
                        pre_K_path = os.path.join(pre_K_path, f'{points_name}.txt')

                        K0 = np.loadtxt(K0_path)
                        K1 = np.loadtxt(K1_path)
                        pose0 = np.loadtxt(pose0_path)
                        pose1 = np.loadtxt(pose1_path)

                        try:
                            mkpts0 = np.loadtxt(mkpts0_path)
                        except:
                            print(f'{mkpts0_path} does not exist')
                            continue
                        mkpts1 = np.loadtxt(mkpts1_path)
                        pre_K = np.loadtxt(pre_K_path)
                        if mkpts0.shape[0] == 0:
                            print(f'file {mkpts0_path} is empty')
                            continue
                        self.data.append([K0, K1, pose0, pose1, mkpts0, mkpts1, pre_K])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
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
    data = pose_dataset(paths)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j].shape[0] == 0:
                print(i, j)
