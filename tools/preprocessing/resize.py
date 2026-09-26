"""Reusable NIfTI resize transform and file conversion helpers.

The command-line entry point lives in :mod:`resize_cli`; this module contains
only the transform, dataset adapter and writer used by that entry point.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ResizeTo512:
    """Crop empty borders and resize a ``(C, D, H, W)`` tensor."""

    def __init__(self, target_height=512, target_width=512, target_depth=50):
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.target_depth = int(target_depth)

    def __call__(self, image):
        if image.ndim != 4:
            raise ValueError("input image must be a 4D tensor of shape (channels, depth, height, width)")
        image = self.remove_black_borders(image.to(torch.float32))
        return F.interpolate(
            image.unsqueeze(0),
            size=(self.target_height, self.target_width, self.target_depth),
            mode='trilinear', align_corners=False,
        )[0]

    @staticmethod
    def remove_black_borders(image):
        mask = (image > 0).any(dim=0)
        indices = torch.nonzero(mask)
        if indices.numel() == 0:
            return image
        mins, maxs = indices.min(dim=0).values, indices.max(dim=0).values
        return image[:, mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]


class NiiDataset(Dataset):
    """Read NIfTI files directly under ``root_path`` for preprocessing."""

    def __init__(self, root_path, transform=None):
        self.root_path = Path(root_path)
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.root_path.iterdir()
            if p.is_file() and p.name.endswith(('.nii', '.nii.gz'))
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = torch.from_numpy(nib.load(str(path)).get_fdata()).float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, path.name


def save_processed_images(loader, save_path):
    """Write transformed batches as NIfTI files with an identity affine."""
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    for images, filenames in loader:
        for image, filename in zip(images, filenames):
            stem = filename[:-7] if filename.endswith('.nii.gz') else filename[:-4]
            output_path = output / (stem + '_processed.nii')
            nib.save(nib.Nifti1Image(image.squeeze(0).numpy(), np.eye(4)), str(output_path))
