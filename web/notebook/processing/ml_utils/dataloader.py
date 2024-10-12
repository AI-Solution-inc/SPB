import os
import cv2
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torch

from sklearn.model_selection import train_test_split


class DefectDataset(Dataset):
    def __init__(self, dataset_root) -> None:
        self.root = dataset_root
        self.images = list(sorted(os.listdir(f"{self.root}/data")))
        self.masks = list(sorted(os.listdir(f"{self.root}/masks")))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
            transforms.Resize((640, 640))
        ])

        self.mask_transform = transforms.Resize((640, 640))

    def __getitem__(self, index):
        img_path = f"{self.root}/data/{self.images[index]}"
        mask_path = f"{self.root}/masks/{self.masks[index]}"

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask_init = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((8, *mask_init.shape))
        for i in range(7):
            mask[i] = np.where(mask_init == i, 1, 0)
        mask[7] = np.ones(mask_init.shape) - np.sum(mask[:-1], axis=0)

        img_t = self.transform(img).float()
        mask_t = torch.from_numpy(mask)
        mask_t = self.mask_transform(mask_t)
        return img_t, mask_t
    
    def __len__(self):
        return len(self.masks)


def getDefectDatasetLoaders(init_dataset, batch_size=1, test_size=0.2):

    train_idx_set, test_idx_set = train_test_split(
        np.arange(len(init_dataset)), 
        test_size=test_size, 
        random_state=42, 
        shuffle=True)

    train_dataset = Subset(init_dataset, train_idx_set)
    test_dataset = Subset(init_dataset, test_idx_set)

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    loader_train.__setattr__("len", len(train_dataset))
    loader_test.__setattr__("len", len(test_dataset))
    
    return {'train': loader_train, 'test': loader_test}
