import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_iou_matrix(boxes1, boxes2):
    """
    Menghitung Pairwise IoU.
    Args:
        boxes1: Tensor (N, 4) -> Anchors [x, y, w, h] atau [x1, y1, x2, y2]
        boxes2: Tensor (M, 4) -> GT Boxes [x1, y1, x2, y2]
    Returns:
        iou_matrix: (N, M)
    """
    if boxes1.shape[-1] == 4: 
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    b2_x1 = boxes2[:, 0]
    b2_y1 = boxes2[:, 1]
    b2_x2 = boxes2[:, 2]
    b2_y2 = boxes2[:, 3]

    # Intersection
    # (N, 1) vs (1, M) -> Broadcasting (N, M)
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    return inter_area / union_area.clamp(min=1e-6)

class ATSSAssigner(nn.Module): 
    def __init__(self, topk=9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk

    def forward(self, anchors, num_anchors_per_level, gt_boxes): 
            """
        Args:
            anchors: Tensor (N, 4) [cx, cy, w, h] - Semua anchor gabungan
            num_anchors_per_level: List [int] - Jumlah anchor per stride [12800, 3200, 800]
            gt_boxes: Tensor (M, 4) [x1, y1, x2, y2] - Ground truth pada image ini
        
        Returns:
            assigned_gt_inds: Tensor (N,) 
                - 0: Background/Ignore
                - 1...M: Index GT yang cocok (1-based index)
            max_iou_per_anchor: Tensor (N,) - Nilai IoU anchor tsb dengan GT pasangannya
        """
            num_anchors = anchors.size(0)
            num_gt = gt_boxes.size(0)

            if num_gt == 0: 
                return torch.zeros(num_anchors, dtype=torch.long, device=anchors.device), torch.zeros(num_anchors, dtype=torch.float, device=anchors.device)
            
            # Hitung IoU Matrix
            overlaps = compute_iou_matrix(anchors, gt_boxes)

            # 2. Hitung L2 Distance Matrix (N, M)
            # Anchor Center
            anchor_cx = anchors[:, 0]
            anchor_cy = anchors[:, 1]

            # GT Center
            gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
            gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

            distances = (anchor_cx.unsqueeze(1) - gt_cx.unsqueeze(0)).pow(2) + (anchor_cy.unsqueeze(1) - gt_cy.unsqueeze(0)).pow(2)
            distances = distances.sqrt()

            candidate_idxs = []
            start_idx = 0

            for num_in_level in num_anchors_per_level:
                end_idx = start_idx + num_in_level
                
                # Ambil distance pada level ini saja
                distances_per_level = distances[start_idx:end_idx, :] # (Num_Level, M)
                
                # Cari top-k terdekat (Smallest distance)
                # topk_idxs: (k, M)
                k = min(self.topk, num_in_level)
                _, topk_idxs_in_level = distances_per_level.topk(k, dim=0, largest=False)
                
                # Convert ke global index
                candidate_idxs.append(start_idx + topk_idxs_in_level)
                
                start_idx = end_idx

            # Gabungkan kandidat dari semua level
            # candidate_idxs: Tensor (K*L, M) -> Indeks anchor kandidat untuk setiap GT
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # 4. Hitung Adaptive Threshold (Mean + Std)
            # Pseudocode Baris 7-10
            
            # Ambil IoU milik para kandidat saja
            # gather dim=0 artinya ambil baris sesuai indeks di candidate_idxs
            candidate_ious = overlaps.gather(0, candidate_idxs) # (K*L, M)

            # Hitung Mean & Std per GT (dim=0)
            iou_mean = candidate_ious.mean(dim=0)
            iou_std = candidate_ious.std(dim=0)
            
            # Threshold: t = m + v
            iou_thresh = iou_mean + iou_std
            
            # 5. Final Selection: Check IoU > Threshold
            # Pseudocode Baris 12
            
            # Buat mask kosong seukuran matriks IoU (N, M)
            is_pos = torch.zeros_like(overlaps, dtype=torch.bool)
            
            # Kita hanya perlu set True pada baris-baris yang jadi kandidat
            # Loop per GT (kolom) untuk efisiensi mapping index
            for gt_idx in range(num_gt):
                cand_for_this_gt = candidate_idxs[:, gt_idx] # Index anchor kandidat
                thresh_for_this_gt = iou_thresh[gt_idx]
                
                # Ambil IoU aslinya
                ious_for_this_gt = overlaps[cand_for_this_gt, gt_idx]
                
                # Cek syarat IoU
                valid_mask = ious_for_this_gt >= thresh_for_this_gt
                
                # Tandai di matriks besar
                # Hanya anchor kandidat yang lolos threshold yang jadi True
                is_pos[cand_for_this_gt[valid_mask], gt_idx] = True

            # 6. Center Sampling: Check Center Inside GT (Wajib di ATSS)
            # Pseudocode Baris 12 ("and center of c in g")
            
            # Matriks logika (N, M): Apakah anchor i ada di dalam GT j?
            # Expand anchor centers (N, 1)
            acx = anchor_cx.unsqueeze(1)
            acy = anchor_cy.unsqueeze(1)
            
            # Expand GT (1, M)
            g_x1 = gt_boxes[:, 0].unsqueeze(0)
            g_y1 = gt_boxes[:, 1].unsqueeze(0)
            g_x2 = gt_boxes[:, 2].unsqueeze(0)
            g_y2 = gt_boxes[:, 3].unsqueeze(0)

            is_in_gts = (acx > g_x1) & (acx < g_x2) & (acy > g_y1) & (acy < g_y2)

            # Gabungkan Syarat: (IoU > Thresh) DAN (Center Inside)
            is_pos = is_pos & is_in_gts

            # 7. Ambiguity Handling (Satu Anchor untuk Banyak GT?)
            # Pseudocode tidak detail, tapi standar praktiknya: Ambil Max IoU
            
            # Hitung berapa GT yang nge-klaim satu anchor
            # Sum horizontal
            num_gt_per_anchor = is_pos.sum(dim=1)
            
            # Jika ada anchor yang diklaim > 1 GT, pilih yang IoU-nya paling besar
            # Kita buat mask baru
            assigned_gt_inds = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
            
            # Cari max IoU per baris (per anchor)
            max_iou_per_anchor, max_iou_col_idx = overlaps.max(dim=1)
            
            # Filter hanya yang is_pos True
            # Di baris mana saja is_pos ada True?
            pos_anchor_idxs = torch.where(is_pos.any(dim=1))[0]
            
            # Untuk anchor positif, assigned_gt adalah index GT dengan IoU terbesar
            # (Tambahkan +1 karena 0 dipakai untuk background)
            assigned_gt_inds[pos_anchor_idxs] = max_iou_col_idx[pos_anchor_idxs] + 1
            
            return assigned_gt_inds, max_iou_per_anchor


if __name__=="__main__":
     import cv2
     import numpy as np
     import os
     
     try: 
        from architecture.test_architecture import SCRFD
        from anchor_generator import SCRFDAnchorGenerator
        from data.data_parsing import parse_wider_face

     except ImportError as e: 
         print("Error Import! Pastikan path file sudah benar.")
         print(f"Detail: {e}")
         exit()
     print("Initialized Model SCRFD2.5G")
     model = SCRFD(variant="2.5GF")
     model.eval()

     input_gambar = "data/WIDER_train/images/0--Parade/0_Parade_marchingband_1_5.jpg"
     input_bounding_boxes = "data/WIDER_train/labelv2_test.txt"
     target_size = 640

     image = cv2.imread(input_gambar)
     if image is None: 
         raise FileNotFoundError(f"Gambar tidak ditemukan: {input_gambar}")
     h_ori, w_ori, _ = image.shape
     image_resize = cv2.resize(image, (target_size, target_size))

     image_tensor = image_resize[:, :, ::-1].transpose(2, 0, 1)
     image_tensor = np.ascontiguousarray(image_tensor, dtype=np.float32)
     image_tensor /= 255.0

     image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)

     scale_x = target_size/ w_ori
     scale_y = target_size / h_ori

    #  parse bboxes
     all_paths, all_boxes = parse_wider_face(input_bounding_boxes, input_gambar)
     target_idx = -1
     for i, path in enumerate(all_paths): 
         if input_gambar in path: 
             target_idx = i
             break
     if target_idx == -1: 
         raise ValueError("Gambar tidak ditemukan di file label txt!")
     gt_boxes = all_boxes[target_idx]

     if len(gt_boxes) > 0: 
         gt_boxes = gt_boxes.copy() # Biar aman
         gt_boxes[:, [0, 2]] *= scale_x # Scale X
         gt_boxes[:, [1, 3]] *= scale_y # Scale Y

     ground_truth = torch.from_numpy(gt_boxes)
     print(f"Ground Truth Loaded: {len(ground_truth)} faces.")

     cls_outs, reg_outs = model(image_tensor)
     feature_maps_sizes = [out.shape[-2:] for out in cls_outs]

     anchor_gen = SCRFDAnchorGenerator()
     anchors = anchor_gen.generate_anchors(feature_maps_sizes)

     num_anchors_per_level = [size[0] * size[1] * 2 for size in feature_maps_sizes]

     print("Running ATSS")
     assigner = ATSSAssigner(topk=9)
     assigned_gt_inds, max_ious = assigner(anchors, num_anchors_per_level, ground_truth)

     # --- 6. VISUALISASI HASIL ---
     num_positives = (assigned_gt_inds > 0).sum().item()
     print(f"Total Anchor Positif (Matched): {num_positives}")

    # Visualisasi
     vis_img = image_resize.copy()

    # Gambar GT (Hijau)
     for box in ground_truth:
         x1, y1, x2, y2 = box.int().numpy()
         cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Gambar Anchor Positif (Merah)
     pos_idxs = torch.where(assigned_gt_inds > 0)[0]
     pos_anchors = anchors[pos_idxs]

     for anc in pos_anchors:
         cx, cy, w, h = anc.int().numpy()
        # Gambar titik pusat anchor
         cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1)

     cv2.imshow("ATSS Result", vis_img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()









    



                
        