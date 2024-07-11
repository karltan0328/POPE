import torch
import random
import numpy as np
from loguru import logger
from collections import OrderedDict
import torch.nn.functional as F


def geodesic_distance(X, X1=None, mode='mean'):
    assert X.dim() in [2, 3]

    if X.dim() == 2:
        X = X.expand(1, -1, -1)

    if X1 is None:
        X1 = torch.eye(3).expand(X.shape[0], 3, 3).to(X.device)

    m = X @ X1.permute(0, 2, 1)
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.clamp(cos, -0.999999, 0.999999) # handle numercial errors
    # cos = torch.min(cos, torch.ones(X.shape[0])).to(device)
    # cos = torch.max(cos, -torch.ones(X.shape[0])).to(device)
    if mode == 'mean':
        return torch.acos(cos).mean()
    return


def normalize_vector(v, return_mag=False):
    """
    将最后一维看作一个向量，对其进行归一化
    """
    v_mag = torch.sqrt(v.pow(2).sum(-1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).to(v.device))
    v_mag = v_mag.view(v.shape[0], 1).expand(v.shape[0], -1)
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:, 0]
    else:
        return v


def qua2mat(quaternion):
    quat = normalize_vector(quaternion).contiguous()

    # (batch, 1)
    qw = quat[..., 0].contiguous().view(quaternion.shape[0], 1)
    qx = quat[..., 1].contiguous().view(quaternion.shape[0], 1)
    qy = quat[..., 2].contiguous().view(quaternion.shape[0], 1)
    qz = quat[..., 3].contiguous().view(quaternion.shape[0], 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx # (batch, 1)
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    # (batch, 3)
    row0 = torch.cat((1 - 2 * yy - 2 * zz,
                      2 * xy - 2 * zw,
                      2 * xz + 2 * yw), dim=-1)
    row1 = torch.cat((2 * xy + 2 * zw,
                      1 - 2 * xx - 2 * zz,
                      2 * yz - 2 * xw), dim=-1)
    row2 = torch.cat((2 * xz - 2 * yw,
                      2 * yz + 2 * xw,
                      1 - 2 * xx - 2 * yy), dim=-1)

    matrix = torch.cat((row0.view(quaternion.shape[0], 1, 3),
                        row1.view(quaternion.shape[0], 1, 3),
                        row2.view(quaternion.shape[0], 1, 3)), dim=-2)

    return matrix # (batch, 3, 3)


def cross_product(u, v):
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    out = torch.cat((i.view(u.shape[0], 1),
                     j.view(u.shape[0], 1),
                     k.view(u.shape[0], 1)), dim=-1)

    return out # (batch, 3)


def o6d2mat(ortho6d):
    # ortho6d - (batch, 6)
    x_raw = ortho6d[..., 0:3] # (batch, 3)
    y_raw = ortho6d[..., 3:6] # (batch, 3)

    x = normalize_vector(x_raw) # (batch, 3)
    z = cross_product(x, y_raw) # (batch, 3)
    z = normalize_vector(z) # (batch, 3)
    y = cross_product(z, x) # (batch, 3)

    x = x.view(x_raw.shape[0], 3, 1)
    y = y.view(x_raw.shape[0], 3, 1)
    z = z.view(x_raw.shape[0], 3, 1)
    matrix = torch.cat((x, y, z), dim=-1) # (batch, 3, 3)
    return matrix


def collate_fn(num_sample):
    random.seed(20231223)
    def collate(batch):
        # print(len(batch), type(batch), type(batch[0]))
        after_process_batch = []
        for data in batch:
            item = {}
            for key in data.keys():
                item[key] = data[key]
                if key == 'mkpts0' or key == 'mkpts1':
                    # 如果mkpts的长度大于num_sample,则随机采样
                    if item[key].shape[0] > num_sample:
                        item[key] = item[key][random.sample(range(item[key].shape[0]), num_sample), :]
                    # 否则补0
                    else:
                        item[key] = np.concatenate((
                            item[key],
                            np.zeros((num_sample - item[key].shape[0], 2), dtype=np.float32)), axis=0)
            after_process_batch.append(item)
        return after_process_batch
    return collate


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    """
    T_0to1: (batch, 4, 4)
    R: (batch, 3, 3)
    t: (batch, 3)
    """
    # angle error between 2 vectors
    t_gt = T_0to1[:, :3, 3].to(t.device) # (batch, 3)

    n = (torch.norm(t, dim=-1) * torch.norm(t_gt, dim=-1)).reshape(-1, 1) # (batch, 1)
    t_err = torch.rad2deg( # (batch, 1)
        torch.arccos(
            torch.clamp(
                torch.bmm(t.unsqueeze(1), t_gt.unsqueeze(2)).reshape(-1, 1) / n.reshape(-1, 1), -1.0, 1.0)))
    t_err = torch.minimum(t_err, 180 - t_err)      # handle E ambiguity
    t_err_0_indices = [i for i in range(len(t_gt)) if torch.norm(t_gt[i]) < ignore_gt_t_thr]
    t_err[t_err_0_indices] = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:, :3, :3].to(R.device) # (batch, 3, 3)

    bmm = torch.bmm(R.permute(0, 2, 1), R_gt) # (batch, 3, 3)
    bmm_trace = bmm.diagonal(dim1=1, dim2=2).sum(dim=-1).reshape(-1, 1) # (batch, 1)
    cos = (bmm_trace - 1) / 2
    cos = torch.clamp(cos, -1., 1.)  # handle numercial errors
    R_err = torch.rad2deg(torch.abs(torch.arccos(cos))) # (batch, 1)
    return t_err, R_err


def relative_pose_error_np(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)      # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, R_err


def project_points(pts, RT, K):
    pts = np.matmul(pts, RT[:, :3].transpose()) + RT[:, 3:].transpose()
    pts = np.matmul(pts, K.transpose())
    dpt = pts[:, 2]
    mask0 = (np.abs(dpt) < 1e-4) & (np.abs(dpt) > 0)
    if np.sum(mask0) > 0:
        dpt[mask0]=  1e-4
    mask1 = (np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1) > 0:
        dpt[mask1] = -1e-4
    pts2d = pts[:, :2] / dpt[:, None]
    return pts2d, dpt


def recall_object(boxA, boxB, thresholded=0.5):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def error_acc(type, errors, thresholds):
    accs = []
    for thr in thresholds:
        accs.append(np.sum(errors < thr) / errors.shape[0])
    res = {f'{type}:ACC{t:2d}': auc for t, auc in zip(thresholds, accs)}
    res[f"{type}:medianErr"] = np.median(errors)
    res[f"{type}:meanErr"] = np.mean(errors)
    return res


def error_auc(type, errors, thresholds):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)
    return {f'{type}:auc@{t:2d}': auc for t, auc in zip(thresholds, aucs)}


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """
    Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """

    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30]
    rotation_aucs = error_auc("R", np.array(metrics['R_errs']), angular_thresholds)
    translation_aucs = error_auc("t", np.array(metrics['t_errs']), angular_thresholds)

    rotation_accs = error_acc("R", np.array(metrics['R_errs']), angular_thresholds)
    translation_accs = error_acc("t", np.array(metrics['t_errs']), angular_thresholds)

    return {**rotation_aucs, **rotation_accs, **translation_aucs, **translation_accs}

def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            k = '.'.join(k.split('.')[1:]) # remove encoder in the name
        if k.endswith('kernel'):
            k = '.'.join(k.split('.')[:-1]) # remove kernel in the name
            new_k = k + '.weight'
            if len(v.shape) == 3: # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).\
                    reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2: # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).\
                    reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif 'ln' in k or 'linear' in k:
            k = k.split('.')
            k.pop(-2) # remove ln and linear in the name
            new_k = '.'.join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
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

    load(model, prefix=prefix)

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
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
