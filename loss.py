import torch 
import torch.nn as nn
import torch.nn.functional as F

def compute_diou_loss(pred_boxes, gt_boxes):
    """
    Menghitung Distance-IoU (DIoU) Loss.
    Args:
        pred_boxes: Tensor (N, 4) [x1, y1, x2, y2]
        gt_boxes:   Tensor (N, 4) [x1, y1, x2, y2]
    """
    # 1. Hitung IoU biasa
    lt = torch.max(pred_boxes[:, :2], gt_boxes[:, :2])
    rb = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:])
    wh = (rb-lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area2 = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    # 2. Hitung DIoU term (jarak center)
    # Center Pred
    pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    # euclidean distance antar center
    center_dist_sq = (pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2

    # Diagonal kotak pembungkus terkecil (c^2)
    lt_outer = torch.min(pred_boxes[:, :2], gt_boxes[:, :2])
    rb_outer = torch.max(pred_boxes[:, 2:], gt_boxes[:, 2:])
    wh_outer = (rb_outer - lt_outer).clamp(min=0)
    diag_dist_sq = wh_outer[:, 0]**2 + wh_outer[:, 1]**2 + 1e-6

    diou = iou - (center_dist_sq / diag_dist_sq)

    return 1 - diou

class SCRFDLoss(nn.Module): 
    def __init__(self): 
        super(SCRFDLoss, self).__init__()
        self.cls_loss_weight = 1.0
        self.box_loss_weight = 2.0

    def forward(self, cls_preds, reg_preds, anchors, assigned_gt_inds, gt_boxes): 
        """
        Args:
            cls_preds: (N, 1) Logits prediksi kelas
            reg_preds: (N, 4) Prediksi offset regresi
            anchors:   (N, 4) Koordinat anchor [cx, cy, w, h]
            assigned_gt_inds: (N, ) Label target dari ATSS (0=Bg, >0=FG)
            gt_boxes:  (M, 4) Ground Truth Boxes asli [x1, y1, x2, y2]
        """
        device = cls_preds.device
        # --- 1. SIAPKAN TARGET CLASSIFICATION ---
        cls_targets = (assigned_gt_inds > 0).float().unsqueeze(1)
        pos_mask = (assigned_gt_inds > 0)
        num_pos = pos_mask.sum()

        # --- 2. CLASSIFICATION LOSS (Focal Loss) ---
        probs = torch.sigmoid(cls_preds)
        bce_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_targets, reduction='none')

        alpha = 0.25
        gamma = 2.0

        p_t = probs * cls_targets + (1 - probs) * (1 - cls_targets)
        focal_weight = alpha * cls_targets + (1 - alpha) * (1 - cls_targets)
        focal_weight = focal_weight * ((1 - p_t) ** gamma)

        cls_loss = focal_weight * bce_loss

        cls_loss = cls_loss.sum() / max(1.0, num_pos.item())

        # --- 3. REGRESSION LOSS (DIoU) ---
        box_loss = torch.tensor(0.0, device=device)
        if num_pos > 0: 
            pos_anchors = anchors[pos_mask]
            pos_reg_preds = reg_preds[pos_mask]
            pos_assigned_gt_inds = assigned_gt_inds[pos_mask]

            matched_gt_boxes = gt_boxes[pos_assigned_gt_inds - 1]
            # Decode Center
            pred_cx = pos_anchors[:, 0] + pos_reg_preds[:, 0] * pos_anchors[:, 2]
            pred_cy = pos_anchors[:, 1] + pos_reg_preds[:, 1] * pos_anchors[:, 3]
            # Decode Size (Exp)
            pred_w = pos_anchors[:, 2] * torch.exp(pos_reg_preds[:, 2])
            pred_h = pos_anchors[:, 3] * torch.exp(pos_reg_preds[:, 3])
            
            # Convert Center-WH ke Corner (x1, y1, x2, y2) untuk DIoU
            pred_x1 = pred_cx - pred_w / 2
            pred_y1 = pred_cy - pred_h / 2
            pred_x2 = pred_cx + pred_w / 2
            pred_y2 = pred_cy + pred_h / 2

            decoded_pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
            diou = compute_diou_loss(decoded_pred_boxes, matched_gt_boxes)
            box_loss = diou.sum() / num_pos.item()

        total_loss = self.cls_loss_weight * cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss
    