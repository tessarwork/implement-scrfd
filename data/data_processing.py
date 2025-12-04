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
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_size, width=target_size, p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    def __call__(self, image, boxes=None): 
        h_ori, w_ori, _ = image.shape
        short_side = min(h_ori, w_ori)
        scale = random.choice(self.scales)
        crop_size = int(short_side * scale)

        max_x_offset = w_ori - crop_size
        max_y_offset = h_ori - crop_size
        x = random.randint(min(0, max_x_offset), max(0, max_x_offset))
        y = random.randint(min(0, max_y_offset), max(0, max_y_offset))

        mean_rgb = np.mean(image, axis=(0, 1))
        canvas = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * mean_rgb.astype(np.uint8)
        
        src_x1 = max(0, x)
        src_y1 = max(0, y)
        src_x2 = min(w_ori, x + crop_size)
        src_y2 = min(h_ori, y + crop_size)

        dst_x1 = max(0, -x)
        dst_y1 = max(0, -y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1: 
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        image_cropped = canvas
        if boxes is not None and len(boxes) > 0: 
            # boxes_aug = boxes.copy()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y

            np.clip(boxes[:, 0], 0, crop_size, out=boxes[:, 0])
            np.clip(boxes[:, 1], 0, crop_size, out=boxes[:, 1])
            np.clip(boxes[:, 2], 0, crop_size, out=boxes[:, 2])
            np.clip(boxes[:, 3], 0, crop_size, out=boxes[:, 3])

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            class_labels = np.ones(len(boxes))
        else: 
            boxes = []
            class_labels = []

        
        
        # Apply color jitter with 0.5 probability

        bboxes_list = boxes.tolist() if len(boxes) > 0 else []

        transformed = self.albu_transform( 
            image=image_cropped,
            bboxes=bboxes_list,
            class_labels=class_labels 
        )
        image_final = transformed['image']
        boxes_final = np.array(transformed['bboxes'], dtype=np.float32)

        if len(boxes_final) == 0: 
            boxes_final = np.zeros((0, 4), dtype=np.float32)
        return image_final, boxes_final
    
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
        
        image = image[:, :, ::-1].transpose(2, 0, 1) 
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255.0 # Normalisasi 0-1
        
        return torch.from_numpy(image), torch.from_numpy(boxes)



  

