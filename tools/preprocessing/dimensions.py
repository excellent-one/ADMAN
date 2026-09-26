"""Print dimensions of all NIfTI files recursively."""
import argparse
from pathlib import Path
import nibabel as nib

def print_dimensions(root):
    for path in sorted(Path(root).rglob('*.nii*')):
        print(path, nib.load(str(path)).shape)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); print_dimensions(p.parse_args().input)
