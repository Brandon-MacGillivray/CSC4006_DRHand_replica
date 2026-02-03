import argparse
import os
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Allow running as a script without requiring mk_3 as a package.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from dataset import RHDDatasetCoords
from losses import HeatmapCoordLoss, soft_argmax_2d
from architecture import Backbone, Heatmap_reg
from utils import save_checkpoint


class HandPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = Heatmap_reg()

    def forward(self, x):
        return self.head(self.backbone(x))


class DatasetWithIds(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if isinstance(self.ds, Subset):
            base = self.ds.dataset
            base_idx = self.ds.indices[idx]
            img, coords, vis = base[base_idx]
            sample_id = base.ids[base_idx] if hasattr(base, "ids") else base_idx
            return img, coords, vis, base_idx, sample_id
        img, coords, vis = self.ds[idx]
        sample_id = self.ds.ids[idx] if hasattr(self.ds, "ids") else idx
        return img, coords, vis, idx, sample_id


@torch.no_grad()
def predict_coords(pred_hm, beta=100.0):
    pred = soft_argmax_2d(pred_hm, beta=beta, normalize=True)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/RHD_v1-1")
    parser.add_argument("--split", default="training", choices=["training", "evaluation"])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hand", default="right", choices=["left", "right", "auto"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--sample-id", type=int, default=-1)
    parser.add_argument("--csv-path", default="mk_3/train_subset_coords.csv")
    parser.add_argument("--csv-mean-path", default="mk_3/train_subset_mean.csv")
    parser.add_argument("--checkpoint-dir", default="mk_3/checkpoints_subset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RHDDatasetCoords(
        args.root,
        split=args.split,
        input_size=args.input_size,
        hand=args.hand,
        normalize=True,
    )

    if args.sample_id >= 0:
        if args.sample_id not in ds.ids:
            raise ValueError(f"sample-id {args.sample_id} not found in dataset")
        idx = ds.ids.index(args.sample_id)
        ds = Subset(ds, [idx])
    elif args.subset_size > 0:
        n = min(args.subset_size, len(ds))
        ds = Subset(ds, list(range(n)))

    ds = DatasetWithIds(ds)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = HandPoseNet().to(device)
    loss_fn = HeatmapCoordLoss(beta=args.beta, normalize=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    csv_file = open(args.csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "epoch",
            "step",
            "ds_idx",
            "sample_id",
            "joint",
            "pred_x",
            "pred_y",
            "gt_x",
            "gt_y",
            "vis",
        ]
    )

    mean_file = open(args.csv_mean_path, "w", newline="")
    mean_writer = csv.writer(mean_file)
    mean_writer.writerow(["epoch", "step", "ds_idx", "sample_id", "mean_err", "vis_points"])

    for epoch in range(args.epochs):
        model.train()
        for imgs, coords, vis, _, _ in loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            vis = vis.to(device)

            pred_hm = model(imgs)
            loss = loss_fn(pred_hm, coords, vis)

            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        for step, (imgs, coords, vis, ds_indices, sample_ids) in enumerate(loader, start=1):
            imgs = imgs.to(device)
            coords = coords.to(device)
            vis = vis.to(device)

            pred_hm = model(imgs)
            pred = predict_coords(pred_hm, beta=args.beta).detach().cpu()
            coords_cpu = coords.detach().cpu()
            vis_cpu = vis.detach().cpu()
            ds_idx_cpu = ds_indices.detach().cpu()
            sample_id_cpu = sample_ids.detach().cpu()

            dist = ((pred - coords_cpu) ** 2).sum(dim=-1).sqrt()
            mask = vis_cpu.float()
            sum_err = (dist * mask).sum(dim=1)
            vis_points = mask.sum(dim=1)
            mean_err = sum_err / (vis_points + 1e-6)

            batch = pred.shape[0]
            for b in range(batch):
                mean_writer.writerow(
                    [
                        epoch,
                        step,
                        int(ds_idx_cpu[b]),
                        int(sample_id_cpu[b]),
                        f"{mean_err[b].item():.6f}",
                        int(vis_points[b].item()),
                    ]
                )
                for j in range(pred.shape[1]):
                    px, py = pred[b, j].tolist()
                    gx, gy = coords_cpu[b, j].tolist()
                    csv_writer.writerow(
                        [
                            epoch,
                            step,
                            int(ds_idx_cpu[b]),
                            int(sample_id_cpu[b]),
                            j,
                            f"{px:.6f}",
                            f"{py:.6f}",
                            f"{gx:.6f}",
                            f"{gy:.6f}",
                            int(vis_cpu[b, j].item()),
                        ]
                    )
        csv_file.flush()
        mean_file.flush()

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
        }
        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

    csv_file.close()
    mean_file.close()


if __name__ == "__main__":
    main()
