"""Save a selected axial slice from every NIfTI file in a directory."""
import argparse
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_nii_slices(input_dir, output_dir, slice_index=35):
    source, dest = Path(input_dir), Path(output_dir); dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.nii*')):
        data = nib.load(str(path)).get_fdata()
        if slice_index < data.shape[2]:
            plt.imsave(str(dest / (path.stem + '.png')), data[:, :, slice_index], cmap='gray')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True); p.add_argument('--slice-index', type=int, default=35)
    a = p.parse_args(); visualize_nii_slices(a.input, a.output, a.slice_index)
