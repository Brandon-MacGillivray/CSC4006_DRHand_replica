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
