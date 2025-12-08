import numpy as np
import pandas as pd
import cv2
import random
import albumentations as A
import torch
from torch.utils.data import Dataset
class Augmentation(): 
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.scales = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.probability_scale = 0.5
        self.albu_transform = A.Compose([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)])
        
    def __call__(self, image, boxes=None): 
        h_ori, w_ori, _ = image.shape
        short_side = min(h_ori, w_ori)
        # --- 1. Random Crop (Sample Redistribution) ---
        scale = random.choice(self.scales)
        crop_size = int(short_side * scale)

        max_x_offset = w_ori - crop_size
        max_y_offset = h_ori - crop_size
        # Koordinat pojok kiri atas crop
        x = random.randint(min(0, max_x_offset), max(0, max_x_offset))
        y = random.randint(min(0, max_y_offset), max(0, max_y_offset))

        # Buat Canvas
        mean_rgb = np.mean(image, axis=(0, 1))
        canvas = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * mean_rgb.astype(np.uint8)
        
        # Hitung Intersection
        src_x1 = max(0, x)
        src_y1 = max(0, y)
        src_x2 = min(w_ori, x + crop_size)
        src_y2 = min(h_ori, y + crop_size)

        dst_x1 = max(0, -x)
        dst_y1 = max(0, -y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # tempel gambar
        if src_x2 > src_x1 and src_y2 > src_y1: 
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        image = canvas
        # --- UPDATE KOORDINAT (BOX & KPS) ---
        if boxes is not None and len(boxes) > 0: 
            # boxes_aug = boxes.copy()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y

            boxes[:, 4::3] -= x
            boxes[:, 5::3] -= y

            box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

            keep = (box_centers_x >= 0) & (box_centers_x < crop_size) & \
                   (box_centers_y >= 0) & (box_centers_y < crop_size)
            
            boxes = boxes[keep]

            np.clip(boxes[:, 0], 0, crop_size, out=boxes[:, 0])
            np.clip(boxes[:, 1], 0, crop_size, out=boxes[:, 1])
            np.clip(boxes[:, 2], 0, crop_size, out=boxes[:, 2])
            np.clip(boxes[:, 3], 0, crop_size, out=boxes[:, 3])
        resize_scale = self.target_size / crop_size
        image = cv2.resize(image, (self.target_size, self.target_size))

        if len(boxes) > 0: 
            boxes[:, :4] *= resize_scale

            boxes[:, 4::3] *= resize_scale
            boxes[:, 5::3] *= resize_scale

        if random.random() < 0.5: 
            image = cv2.flip(image, 1)

            if len(boxes) > 0: 
                w_curr = self.target_size
                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()

                boxes[:, 0] = w_curr - old_x2
                boxes[:, 2] = w_curr - old_x1

                boxes[:, 4::3] = w_curr - boxes[:, 4::3]

                kps = boxes[:, 4:].copy()

                temp = kps[:, 0:3].copy()
                kps[:, 0:3] = kps[:, 3:6]
                kps[:, 3:6] = temp

                temp = kps[:, 9:12].copy()
                kps[:, 9:12] = kps[:, 12:15]
                kps[:, 12:15] = temp

                boxes[:, 4:] = kps
        transformed = self.albu_transform(image=image)
        image = transformed['image']

        return image, boxes






        

    
class WiderFaceDataset(Dataset):
    def __init__(self, image_paths, targets, is_train=True, target_size=240):
        self.image_paths = image_paths
        self.targets = targets
        self.is_train = is_train
        self.target_size = target_size

        # Initiate Augmentor
        if self.is_train: 
            self.augmentor = Augmentation(target_size=self.target_size)
        else: 
            self.augmentor = None

    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)

        boxes = self.targets[index].copy()
        print(boxes.shape)
        if self.is_train: 
            image, boxes = self.augmentor(image, boxes)
        else: 
            h_ori, w_ori, _ = image.shape
            image = cv2.resize(image, (self.target_size, self.target_size))

            if len(boxes) > 0:
                scale_x = self.target_size / w_ori
                scale_y = self.target_size / h_ori
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                boxes[:, 4::3] *= scale_x
                boxes[:, 5::3] *= scale_y
        
        image = image[:, :, ::-1].transpose(2, 0, 1) 
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255.0 # Normalisasi 0-1
        
        return torch.from_numpy(image), torch.from_numpy(boxes)



  

