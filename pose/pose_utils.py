import torch
from torch import nn

def geodesic_distance(X, X1=None,mode='mean'):
    assert X.dim() in [2, 3]

    if X.dim() == 2:
        X = X.expand(1, -1, -1)

    if X1 is None:
        X1 = torch.eye(3).expand(X.shape[0], 3, 3).to(X.device)

    m = X @ X1.permute(0, 2, 1)
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    # cos = torch.min(cos, torch.ones(X.shape[0])).to(device)
    # cos = torch.max(cos, -torch.ones(X.shape[0])).to(device)
    if mode == 'mean':
        return torch.acos(cos).mean()
    return


def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).cuda())
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

def qua2mat(quaternion):
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out

def o6d2mat(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])  #Tuple -> Tensor
    return tuple(batch)

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    if targets is not None:
        # Only write targets to cuda device if not None, otherwise return None
        targets = [{k:v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets

class BoundingBoxEmbeddingSine(nn.Module):
    """
    Positional embedding of bounding box coordinates.
    """
    def __init__(self, num_pos_feats=32):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, bboxes: torch.Tensor):
        # Assuming only the bboxes for a single image get passed
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=bboxes.device)
        dim_t = 2 ** dim_t
        x_enc = bboxes[:, 0, None] * dim_t
        y_enc = bboxes[:, 1, None] * dim_t
        w_enc = bboxes[:, 2, None] * dim_t
        h_enc = bboxes[:, 3, None] * dim_t
        x_enc = torch.cat((x_enc.sin(), x_enc.cos()), dim=-1)
        y_enc = torch.cat((y_enc.sin(), y_enc.cos()), dim=-1)
        w_enc = torch.cat((w_enc.sin(), w_enc.cos()), dim=-1)
        h_enc = torch.cat((h_enc.sin(), h_enc.cos()), dim=-1)
        pos_embed = torch.cat((x_enc, y_enc, w_enc, h_enc), dim=-1)
        return pos_embed
