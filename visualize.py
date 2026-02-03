import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Allow running as a script without requiring mk_3 as a package.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from dataset import RHDDatasetCoords
from architecture import Backbone, Heatmap_reg
from losses import soft_argmax_2d


class HandPoseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = Heatmap_reg()

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def visualize_one(model, dataset, index, device, beta=100.0):
    img, coords, vis = dataset[index]
    img_b = img.unsqueeze(0).to(device)

    heatmaps = model(img_b)
    pred = soft_argmax_2d(heatmaps, beta=beta, normalize=True)[0].cpu()

    h, w = img.shape[1], img.shape[2]
    scale = torch.tensor([w - 1, h - 1], dtype=torch.float32)
    gt_xy = coords * scale
    pred_xy = pred * scale

    img_np = img.permute(1, 2, 0).numpy()
    vis_idx = vis.numpy().astype(bool)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_np)
    plt.scatter(gt_xy[vis_idx, 0], gt_xy[vis_idx, 1], c="lime", s=25, label="GT")
    plt.scatter(pred_xy[:, 0], pred_xy[:, 1], c="red", s=25, label="Pred")
    plt.legend()
    plt.title(f"Sample {index}")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/RHD_v1-1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--beta", type=float, default=100.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RHDDatasetCoords(
        args.root,
        split=args.split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
    )

    model = HandPoseNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    visualize_one(model, dataset, args.index, device, beta=args.beta)


if __name__ == "__main__":
    main()
