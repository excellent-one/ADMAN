
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


def _safe_minmax(image):
    lo, hi = np.min(image), np.max(image)
    if hi == lo:
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo)


def select_split(paths, ratios, part, seed=8, group=None):
    """Split a group of paths into class-stratified train/val/test subsets.

    Args:
        paths (list): File paths, already sorted by the caller.
        ratios (tuple): (train, val, test) ratios that sum to 1.0.
        part (str or None): One of 'train'/'val'/'test'; None returns every path unchanged.
        seed (int): Shuffle seed, so the same seed reproduces exactly the same split
            across separate Dataset objects.
        group (str, optional): Class name used for logging only.
    Returns:
        list: Paths belonging to part.
    """
    if ratios is None or part is None:
        return list(paths)

    import random as _random
    rng = _random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    bounds = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n),
    }
    lo, hi = bounds[part]
    return shuffled[lo:hi]

class NiiDataset2(Dataset):
    def __init__(self, root_path, domain1,domain2, transform=None, return_index=False):
        """
        Args:
            root_path (str): Path to the root directory containing the subdirectories for AD and NC.
            domain (str): The domain ('HR' or 'LR').
            transform (callable, optional): A function/transform to apply to the data.
            return_index (bool, optional): If True, additionally return idx so a batch can be
                traced back to specific files. Defaults to False (original behavior).
        """
        self.root_path = root_path
        self.domain1 = domain1
        self.domain2 = domain2
        self.transform = transform
        self.return_index = return_index
        self.file_paths1 = []
        self.file_paths2 = []
        self.labels1 = []
        self.labels2 = []

        folder_path1 = os.path.join(root_path, domain1)
        folder_path2 = os.path.join(root_path, domain2)
        class_name1 = sorted(os.listdir(folder_path1))
        class_name2 = sorted(os.listdir(folder_path2))

        # Load file paths and labels from the specified domain
        for label, folder in enumerate(class_name1):
            folder_path = os.path.join(root_path, domain1, folder)
            if os.path.exists(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith('.nii'):
                        self.file_paths1.append(os.path.join(folder_path, fname))
                        self.labels1.append(label)


        # Load file paths and labels from the specified domain
        for label, folder in enumerate(class_name2):
            folder_path = os.path.join(root_path, domain2, folder)
            if os.path.exists(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith('.nii'):
                        self.file_paths2.append(os.path.join(folder_path, fname))
                        self.labels2.append(label)

    def __len__(self):
        return len(self.file_paths1)

    # def __len__(self):
    #     return len(self.file_paths1) + len(self.file_paths2)
    #
    # def __getitem__(self, idx):
    #     if idx < len(self.file_paths1):
    #         file_path = self.file_paths1[idx]
    #         label = self.labels1[idx]
    #         nii_file = nib.load(file_path)
    #         image_data = nii_file.get_fdata()
    #         target_shape = (128, 128, 96)
    #         zoom_factors = [target / current for target, current in zip(target_shape, image_data.shape)]
    #         resized_image_data = zoom(image_data, zoom_factors, order=1)
    #         if self.transform:
    #             resized_image_data = self.transform(resized_image_data)
    #         return resized_image_data, label
    #     else:
    #         idx -= len(self.file_paths1)
    #         file_path = self.file_paths2[idx]
    #         label = self.labels2[idx]
    #         nii_file = nib.load(file_path)
    #         image_data = nii_file.get_fdata()
    #         if self.transform:
    #             image_data = self.transform(image_data)
    #         return image_data, label
    def __getitem__(self, idx):
        nii_path1 = self.file_paths1[idx]
        nii_path2 = self.file_paths2[idx]
        nii_img1 = nib.load(nii_path1)
        nii_img2 = nib.load(nii_path2)
        img_data1 = nii_img1.get_fdata()
        img_data2 = nii_img2.get_fdata()

        # Normalize the data to range [0, 1]
        img_data1 = _safe_minmax(img_data1)
        img_data2 = _safe_minmax(img_data2)

        # Convert to a 3D tensor (assume grayscale, single channel)
        img_data1 = torch.tensor(img_data1, dtype=torch.float32).unsqueeze(0)
        img_data2 = torch.tensor(img_data2, dtype=torch.float32).unsqueeze(0)

        img_data1 = torch.nn.functional.interpolate(img_data1.unsqueeze(0), size=(128, 128, 96), mode='trilinear',
                                                    align_corners=False).squeeze(0)

        # Apply transformations if provided
        if self.transform:
            img_data1 = self.transform(img_data1)
            img_data2 = self.transform(img_data2)

        label1 = self.labels1[idx]
        label2 = self.labels2[idx]

        if self.return_index:
            return img_data1, img_data2, label1, label2, idx
        return img_data1,img_data2, label1,label2

class NiiDataset(Dataset):
    def __init__(self, root_path, domain, transform=None, split=None, split_seed=8,
                 return_index=False):
        """
        Args:
            root_path (str): Path to the root directory containing the subdirectories for AD and NC.
            domain (str): The domain ('HR' or 'LR').
            transform (callable, optional): A function/transform to apply to the data.
            split (tuple, optional): (train, val, test) ratios, e.g. (0.7, 0.15, 0.15). When given,
                the domain is divided into three disjoint class-stratified subsets and only the
                requested one is kept. Pass one of 'train'/'val'/'test' to select the part.
                Defaults to None, which keeps the original behavior (use every file).
            split_seed (int): Seed for the split shuffle, so the same seed always reproduces the
                same partition across separate Dataset objects.
        """
        self.root_path = root_path
        self.domain = domain
        self.transform = transform
        self.return_index = return_index
        self.file_paths = []
        self.labels = []
        self.split = split
        self.split_seed = split_seed

        split_ratios, split_part = self._parse_split(split)

        class_to_label = {"AD": 0, "NC": 1}

        # Load file paths and labels from the specified domain
        for folder, label in class_to_label.items():
            folder_path = os.path.join(root_path, domain, folder)
            if not os.path.exists(folder_path):
                continue
            paths = []
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.nii'):
                    paths.append(os.path.join(folder_path, fname))
            paths = select_split(paths, split_ratios, split_part, split_seed, group=folder)
            self.file_paths.extend(paths)
            self.labels.extend([label] * len(paths))

    @staticmethod
    def _parse_split(split):
        """Accept (0.7, 0.15, 0.15) for automatic ratios or ('train', 0.7, 0.15, 0.15) to select a part."""
        if split is None:
            return None, None
        if isinstance(split, str):
            raise ValueError("split must be written as (ratios) or (part, ratios...), e.g. ('val', 0.7, 0.15, 0.15)")
        part = split[0] if isinstance(split[0], str) else None
        ratios = split[1:] if part else split
        ratios = tuple(float(r) for r in ratios)
        if len(ratios) != 3:
            raise ValueError("split ratios need three numbers (train, val, test), got {}".format(ratios))
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0, got {}".format(ratios))
        if part is not None and part not in ('train', 'val', 'test'):
            raise ValueError("split part must be one of 'train'/'val'/'test', got {}".format(part))
        return ratios, part

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nii_path = self.file_paths[idx]
        nii_img = nib.load(nii_path)
        img_data = nii_img.get_fdata()

        # Normalize the data to range [0, 1]
        img_data = _safe_minmax(img_data)

        # Convert to a 3D tensor (assume grayscale, single channel)
        img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0)

        # Apply transformations if provided
        if self.transform:
            img_data = self.transform(img_data)

        label = self.labels[idx]

        if self.return_index:
            return img_data, label, idx
        return img_data, label

# class ResizeTo512:
#     def __init__(self, target_depth=50, target_height=512, target_width=512):
#         '''
#         Args:
#             target_depth:
#             target_height:
#             target_width:
#         '''
#         self.target_depth = target_depth
#         self.target_height = target_height
#         self.target_width = target_width
#
#     def __call__(self, image):
#         # print(image.ndim)
#         if image.ndim != 4:
#             raise ValueError("input image must be a 4D tensor of shape (channels, depth, height, width)")
#
#         image = image.to(torch.float32)
#         image = self.remove_black_borders(image)
#         resized_image = F.interpolate(image.unsqueeze(0), size=(self.target_depth, self.target_height, self.target_width), mode='trilinear', align_corners=False)[0]
#         return resized_image
#
#     def remove_black_borders(self, image):
#         channels, depth, height, width = image.shape
#         image = image.to(torch.float32)
#         non_black_mask = (image > 0).any(dim=0)
#         non_black_indices = torch.nonzero(non_black_mask)
#         if non_black_indices.size(0) == 0:
#             return image
#
#         min_d, max_d = non_black_indices[:, 0].min().item(), non_black_indices[:, 0].max().item()
#         min_h, max_h = non_black_indices[:, 1].min().item(), non_black_indices[:, 1].max().item()
#         min_w, max_w = non_black_indices[:, 2].min().item(), non_black_indices[:, 2].max().item()
#
#         cropped_image = image[:, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
#
#         return cropped_image

# Example function to load data based on domains
def load_data(root_path, domain, batch_size, kwargs, split=None, split_seed=8, shuffle=True,
              drop_last=True):

    transform = None
    dataset = NiiDataset(root_path=root_path, domain=domain, transform=transform,
                         split=split, split_seed=split_seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader

def load_data2(root_path, domain1, domain2,batch_size, kwargs):

    dataset = NiiDataset2(root_path=root_path, domain1=domain1, domain2=domain2,transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs) #drop_last=True, delete the last batch
    return loader

# Test script
if __name__ == '__main__':
    root_path = r'/media/user/BF45460F1FCF00571/Tcy/MRI_Code/DSAN/data/AD_1_PET'  # Update this with the path to your dataset
    train_loader_hr = load_data2(root_path, domain1='AD_128_90', domain2='AD_Monitor_128_24',batch_size=2,kwargs = {'num_workers': 0, 'pin_memory': False})
    for i, (data, labels) in enumerate(train_loader_hr):
        print(f"Data shape: {data.shape}")

    # batch_size = 2
    # kwargs = {'num_workers': 0, 'pin_memory': False}
    #
    # # Test HR domain
    # print("Testing HR domain...")
    # train_loader_hr = load_data2(root_path, domain1='AD_128_90',domain2='AD_Monitor_128_24',batch_size=batch_size, kwargs=kwargs)
    # for i, (data, labels) in enumerate(train_loader_hr):
    #     print(f"HR Batch {i + 1}:")
    #     print(f"Data shape: {data.shape}")  #B C H W D
    #     print(f"Labels: {labels}")
    #     if i == 1:  # Only check first two batches
    #         break

    # # Test LR domain
    # print("\nTesting LR domain...")
    # train_loader_lr = load_data(root_path, 'LR', batch_size, kwargs)
    # for i, (data, labels) in enumerate(train_loader_lr):
    #     print(f"LR Batch {i + 1}:")
    #     print(f"Data shape: {data.shape}")
    #     print(f"Labels: {labels}")
    #     if i == 1:  # Only check first two batches
    #         break
    #
    # # Test LR_test domain
    # print("\nTesting LR_test domain...")
    # test_loader_lr_test = load_data(root_path, 'LR_test', batch_size, kwargs)
    # for i, (data, labels) in enumerate(test_loader_lr_test):
    #     print(f"LR_test Batch {i + 1}:")
    #     print(f"Data shape: {data.shape}")
    #     print(f"Labels: {labels}")
    #     if i == 1:  # Only check first two batches
    #         break
