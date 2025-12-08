from data_processing import WiderFaceDataset
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import os
from data_parsing import parse_wider_face


def visualize_sample(tensor_img, tensor_boxes): 
    img = tensor_img.numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = img.copy()

    boxes = tensor_boxes.numpy()
    h, w, _ = img.shape

    for box in boxes: 
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(img, (x1, y1), (x2,y2), (0, 255, 0), 2)
        landmarks = box[4:]

        for i in range(0, 15, 3):
            lx = int(landmarks[i])
            ly = int(landmarks[i+1])
            viz = landmarks[i+2] # Visibility
            
            # Gambar hanya jika valid (koordinat > 0)
            if lx > 0 and ly > 0:
                # Mata (Titik 1 & 2) -> Merah
                if i < 6: 
                    color = (0, 0, 255) 
                # Hidung (Titik 3) -> Biru
                elif i < 9:
                    color = (255, 0, 0)
                # Mulut (Titik 4 & 5) -> Kuning
                else:
                    color = (0, 255, 255)
                    
                cv2.circle(img, (lx, ly), 3, color, -1)

        # cv2.circle(img, (x1,y1), 5, (255,0,0), -1)
    return img

if __name__ == "__main__": 
    txt_path = "data/WIDER_train/labelv2_test.txt"
    img_root = "data/WIDER_train/images"

    print("--- START INTEGRATION TEST ---")

    try: 
        paths, boxes = parse_wider_face(txt_path, img_root)
        if len(paths) == 0:
            print("Error: Tidak ada data yang di-parse. Cek file txt Anda.")
            exit()
        print("\n[INFO] Inisialisasi Dataset...")
        dataset = WiderFaceDataset(paths, boxes, is_train=True, target_size=240)

        idx_test = 0
        print(f"[INFO] Menguji Index ke-{idx_test}: {paths[idx_test]}")

        plt.figure(figsize=(15, 10))

        for i in range(6): 
            tensor_img, tensor_boxes = dataset[idx_test]
            vis_img = visualize_sample(tensor_img, tensor_boxes)
            plt.subplot(2, 3, i+1)
            plt.imshow(vis_img)
            plt.title(f"Augment #{i+1}\nBoxes: {len(tensor_boxes)}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        print("\n[SUCCESS] Pipeline berjalan lancar!")
    except Exception as e: 
        print(f"\n[ERROR] Terjadi kesalahan: {e}")

