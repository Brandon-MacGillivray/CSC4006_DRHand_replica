import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class RHDDatasetCoords(Dataset):
    def __init__(self, root, split="training", input_size=256, hand="right", normalize=True):
        self.root = root
        self.split = split
        self.input_size = input_size
        self.hand = hand
        self.normalize = normalize

        anno_path = os.path.join(root, split, f"anno_{split}.pickle")
        with open(anno_path, "rb") as f:
            self.anno = pickle.load(f)

        self.ids = sorted(self.anno.keys())

    def __len__(self):
        return len(self.ids)

    def _select_hand(self, uv_vis):
        if self.hand == "left":
            return uv_vis[0:21]
        if self.hand == "right":
            return uv_vis[21:42]
        if self.hand == "auto":
            left = uv_vis[0:21]
            right = uv_vis[21:42]
            left_vis = (left[:, 2] == 1).sum()
            right_vis = (right[:, 2] == 1).sum()
            return right if right_vis >= left_vis else left
        raise ValueError("hand must be one of: 'left', 'right', 'auto'")

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        anno = self.anno[sample_id]

        img_path = os.path.join(self.root, self.split, "color", f"{sample_id:05d}.png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size))
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        uv_vis = anno["uv_vis"]  # (42, 3)
        hand = self._select_hand(uv_vis)
        coords = hand[:, :2]  # (21, 2) in 320x320 pixels
        vis = hand[:, 2] == 1  # (21,)

        scale = self.input_size / 320.0
        coords = coords * scale
        if self.normalize:
            coords = coords / self.input_size

        coords = torch.tensor(coords, dtype=torch.float32)
        vis = torch.tensor(vis, dtype=torch.bool)

        return img, coords, vis
