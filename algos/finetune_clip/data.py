from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
import json
import cv2
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import *


def create_dataloader(image_dir, label_dir, batch_size):
    names = list(
        set(
            [
                x.split(".")[0]
                .replace("_post_disaster", "")
                .replace("_pre_disaster", "")
                for x in os.listdir(image_dir)
                if not ".aux.xml" in x
            ]
        )
    )
    random.seed(42)
    random.shuffle(names)
    train_imgs = names[: int(len(names) * 0.8)]
    val_imgs = names[int(len(names) * 0.8) :]
    train_dataset = CLIPDataset(train_imgs, image_dir, label_dir)
    val_dataset = CLIPDataset(val_imgs, image_dir, label_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    return train_loader, val_loader


class CLIPDataset(Dataset):
    def __init__(self, img_list, img_dir, label_dir, transform=None):
        self.img_list = img_list
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.data = []
        for x in tqdm(img_list):
            with open(os.path.join(label_dir, f"{x}_post_disaster.json"), "r") as f:
                text = json.load(f)["metadata"]["disaster_type"]
            self.data.append(
                (
                    f"{img_dir}/{x}_pre_disaster.png",
                    f"{img_dir}/{x}_post_disaster.png",
                    f"{label_dir}/{x}_post_disaster.json",
                    text,
                )
            )  # (pre_path, post_path, label_path, description)

    def load_image(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整图像大小
        transform = ResizeLongestSide(1024)
        image = transform.apply_image(image)

        # 转换为浮点数并归一化
        image = torch.as_tensor(image).permute(2, 0, 1).float() / 255.0

        # 标准化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        image = (image - mean) / std
        return image.unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pre_path, post_path, label_path, description = self.data[idx]
        pre_image = self.load_image(pre_path)
        post_image = self.load_image(post_path)

        return pre_image, post_image, description
