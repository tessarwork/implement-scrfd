from data_processing_kps import WiderFaceDataset # Sesuaikan nama file dataset Anda
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import os
from data_parsing import parse_wider_face # Sesuaikan nama file parser Anda

def visualize_sample(tensor_img, tensor_targets): 
    img = tensor_img.numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = img.copy()

    targets = tensor_targets.numpy()

    for ann in targets: 
        # 1. Gambar Box (Index 0-3)
        x1, y1, x2, y2 = ann[:4].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. Gambar Keypoints (Index 4-18)
        landmarks = ann[4:]
        # Reshape jadi 5 titik (x, y, v) biar gampang loop
        # Jika padding -1 semua, loop ini aman karena ada if lx > 0
        kps = landmarks.reshape(-1, 3) 
        
        colors = [
            (0, 0, 255),   # Mata Kiri (Merah)
            (0, 0, 255),   # Mata Kanan (Merah)
            (255, 0, 0),   # Hidung (Biru)
            (0, 255, 255), # Mulut Kiri (Kuning)
            (0, 255, 255)  # Mulut Kanan (Kuning)
        ]

        for i, point in enumerate(kps):
            lx, ly, v = point
            if lx > 0 and ly > 0: # Cek validitas
                cv2.circle(img, (int(lx), int(ly)), 4, colors[i], -1)

    return img

if __name__ == "__main__": 
    # Pastikan path ini benar
    txt_path = "data/WIDER_train/labelv2_test.txt"
    img_root = "data/WIDER_train/images"

    print("--- START INTEGRATION TEST WITH KPS ---")

    try: 
        paths, anns = parse_wider_face(txt_path, img_root)
        
        if len(paths) == 0:
            print("Error: Tidak ada data yang di-parse. Cek file txt Anda.")
            exit()
            
        print("\n[INFO] Inisialisasi Dataset...")
        # Target size 640 sesuai paper (240 kekecilan untuk lihat detail landmark)
        dataset = WiderFaceDataset(paths, anns, is_train=True, target_size=640)

        # Ambil sampel acak atau index 0
        idx_test = 0 
        print(f"[INFO] Menguji Index ke-{idx_test}: {paths[idx_test]}")
        
        # Cek shape data mentah
        dummy_img, dummy_ann = dataset[idx_test]
        print(f"[DEBUG] Annotation Shape: {dummy_ann.shape}") # Harusnya [N, 19]

        plt.figure(figsize=(15, 10))

        for i in range(6): 
            tensor_img, tensor_anns = dataset[idx_test]
            vis_img = visualize_sample(tensor_img, tensor_anns)
            
            plt.subplot(2, 3, i+1)
            plt.imshow(vis_img)
            plt.title(f"Augment #{i+1}\nTargets: {len(tensor_anns)}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
        print("\n[SUCCESS] Pipeline Visualisasi Landmark Berjalan!")
        
    except Exception as e: 
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        import traceback
        traceback.print_exc() # Print detail error