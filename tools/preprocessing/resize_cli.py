"""Configurable CLI wrapper for the legacy resize transform."""
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from .resize import ResizeTo512, NiiDataset, save_processed_images

def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--height', type=int, default=128); p.add_argument('--width', type=int, default=128); p.add_argument('--depth', type=int, default=24)
    a = p.parse_args(); transform = transforms.Compose([ResizeTo512(a.height, a.width, a.depth)])
    loader = DataLoader(NiiDataset(a.input, transform=transform), batch_size=1, shuffle=False)
    save_processed_images(loader, a.output)

if __name__ == '__main__':
    main()
