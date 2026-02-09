import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l2_loss(pred, target, vis, eps=1e-6):
    """
    pred/target: (N, J, 2)
    vis: (N, J) boolean or 0/1 mask
    """
    mask = vis.float().unsqueeze(-1)
    diff = (pred - target) ** 2 * mask
    denom = mask.sum() * 2.0 + eps
    return diff.sum() / denom

def soft_argmax_2d(heatmaps, beta=100.0, normalize=True):
    """
    heatmaps: (N, J, H, W)
    returns coords: (N, J, 2) as (x, y)
    """
    n, j, h, w = heatmaps.shape
    flat = heatmaps.view(n, j, -1)
    prob = F.softmax(flat * beta, dim=-1).view(n, j, h, w)

    ys = torch.linspace(0, h - 1, h, device=heatmaps.device)
    xs = torch.linspace(0, w - 1, w, device=heatmaps.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x = (prob * xx).sum(dim=(-2, -1))
    y = (prob * yy).sum(dim=(-2, -1))

    if normalize:
        x = x / (w - 1)
        y = y / (h - 1)

    return torch.stack([x, y], dim=-1)


def heatmap_coord_l2_loss(pred_heatmaps, target_coords, vis, beta=100.0, normalize=True, eps=1e-6):
    pred_coords = soft_argmax_2d(pred_heatmaps, beta=beta, normalize=normalize)
    return masked_l2_loss(pred_coords, target_coords, vis, eps=eps)


class HeatmapCoordLoss(nn.Module):
    def __init__(self, beta=100.0, normalize=True):
        super().__init__()
        self.beta = beta
        self.normalize = normalize

    def forward(self, pred_heatmaps, target_coords, vis):
        return heatmap_coord_l2_loss(
            pred_heatmaps,
            target_coords,
            vis,
            beta=self.beta,
            normalize=self.normalize,
        )

class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        # constant so curve is continuous
        self.C = w - w * torch.log(torch.tensor(1 + w / epsilon))

    def forward(self, pred, target, vis=None):
        # pred, target: (N, 21, 2)
        # vis: (N, 21) optional

        diff = pred - target
        abs_diff = diff.abs()

        w = self.w
        eps = self.epsilon
        C = self.C.to(pred.device)

        # two regions
        small = w * torch.log(1 + abs_diff / eps)
        large = abs_diff - C

        loss = torch.where(abs_diff < w, small, large)

        # apply visibility mask if given
        if vis is not None:
            vis = vis.float().unsqueeze(-1)  # (N,21,1)
            loss = loss * vis
            denom = vis.sum() * 2.0 + 1e-6   # x and y
        else:
            denom = loss.numel()

        return loss.sum() / denom

def coords_to_heatmaps(coords, vis, H=64, W=64, sigma=2.0):
    """
    coords: (N,J,2) in [0,1] normalized
    vis:    (N,J) boolean
    returns (N,J,H,W)
    """
    N, J, _ = coords.shape
    device = coords.device
    yy = torch.arange(H, device=device).view(1, 1, H, 1).float()
    xx = torch.arange(W, device=device).view(1, 1, 1, W).float()

    x = coords[..., 0].clamp(0, 1) * (W - 1)
    y = coords[..., 1].clamp(0, 1) * (H - 1)
    x = x.view(N, J, 1, 1)
    y = y.view(N, J, 1, 1)

    g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    if vis is not None:
        g = g * vis.float().view(N, J, 1, 1)

    return g

class HeatmapMSELoss(nn.Module):
    def __init__(self, H=64, W=64, sigma=2.0):
        super().__init__()
        self.H, self.W, self.sigma = H, W, sigma
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, pred_heatmaps, target_coords, vis):
        gt = coords_to_heatmaps(target_coords, vis, H=self.H, W=self.W, sigma=self.sigma)
        return self.mse(pred_heatmaps, gt)