import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging
import sys
from datetime import datetime

try: 
    from architecture.scrfd import SCRFD
    from anchor_generator import SCRFDAnchorGenerator
    from atss import ATSSAssigner
    from loss import SCRFDLoss
    from data.data_parsing import parse_wider_face
    from data.data_processing import WiderFaceDataset
except ImportError as e: 
    print(f"CRITICAL ERROR: Modul tidak ditemukan. {e}")
    sys.exit(1)

def get_current_lr(epoch, batch_idx, num_batches, args): 
    """
    Menghitung Learning Rate sesuai Paper SCRFD:
    - Initial: 0.00001
    - Warmup: Linear sampai Target LR (Epoch 0-3)
    - Decay: /10 di epoch 440 dan 544
    - Total Epoch: 640
    """
    target_lr = 0.01 * (args.batch_size / 32.0)
    if epoch < 3: 
        start_lr = 0.00001

        total_warmup_iters = 3 * num_batches
        current_iter = (epoch * num_batches) + batch_idx

        lr = start_lr + (target_lr - start_lr) * (current_iter / total_warmup_iters)
        return lr
    lr = target_lr
    if epoch >= 544: 
        lr *= 0.01
    elif epoch >= 440: 
        lr *= 0.1
    return lr



def setup_logger(log_dir): 
    """
    Menyiapkan sistem logging: Output ke Terminal dan File.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("SCRFD_Training")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler 1: File (Simpan ke disk)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler 2: Stream (Tampil di Terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger, log_filename

def detection_collate(batch):
    imgs = []
    targets = []
    for sample in batch: 
        imgs.append(sample[0])
        targets.append(sample[1])
    imgs = torch.stack(imgs, 0)
    return imgs, targets

def process_model_output(cls_outs, reg_outs): 
    final_cls_pred = []
    final_reg_pred = []

    for stride_idx in range(len(cls_outs)): 
        cls = cls_outs[stride_idx]
        reg = reg_outs[stride_idx]

        B, _, H, W = cls.shape

        cls = cls.permute(0, 2, 3, 1).contiguous().view(B, -1, 1) # Pakai 'u' setelah 'g'
        reg = reg.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)


        
        cls = cls.view(B, -1, 1)
        reg = reg.view(B, -1, 4)

        final_cls_pred.append(cls)
        final_reg_pred.append(reg)
    final_cls_pred = torch.cat(final_cls_pred, dim=1)
    final_reg_pred = torch.cat(final_reg_pred, dim=1)

    return final_cls_pred, final_reg_pred
def evaluate(model, val_loader, device, anchor_gen, assigner, criterion):
    model.eval()
    val_loss = 0.0
    val_cls_loss = 0.0
    val_reg_loss = 0.0
    
    with torch.no_grad():
        # Gunakan tqdm simple untuk validasi
        for imgs, targets in tqdm(val_loader, desc="Validating", leave=False):
            imgs = imgs.to(device)
            
            # Forward
            cls_outs, reg_outs = model(imgs)
            flat_cls, flat_reg = process_model_output(cls_outs, reg_outs)
            
            # Anchor & Assignment (Sama seperti train)
            feat_sizes = [out.shape[-2:] for out in cls_outs]
            anchors = anchor_gen.generate_anchors(feat_sizes, device=device)
            num_anchors_per_level = [s[0]*s[1]*2 for s in feat_sizes]
            
            batch_loss = 0.0
            batch_cls = 0.0
            batch_reg = 0.0
            
            for b in range(imgs.size(0)):
                curr_cls = flat_cls[b]
                curr_reg = flat_reg[b]
                curr_gt = targets[b].to(device)
                
                assigned_gt_inds, _ = assigner(anchors, num_anchors_per_level, curr_gt)
                total, l_cls, l_reg = criterion(curr_cls, curr_reg, anchors, assigned_gt_inds, curr_gt)
                
                batch_loss += total.item()
                batch_cls += l_cls.item()
                batch_reg += l_reg.item()
            
            val_loss += (batch_loss / imgs.size(0))
            val_cls_loss += (batch_cls / imgs.size(0))
            val_reg_loss += (batch_reg / imgs.size(0))
            
    avg_loss = val_loss / len(val_loader)
    avg_cls = val_cls_loss / len(val_loader)
    avg_reg = val_reg_loss / len(val_loader)
    
    model.train() # Kembalikan ke mode train
    return avg_loss, avg_cls, avg_reg

def main(args): 
    logger, log_file = setup_logger(args.log_dir)
    logger.info("=== SCRFD TRAINING STARTED ===")
    logger.info(f"Log file saved to: {log_file}")

    logger.info(f"Config: Batch={args.batch_size}, Epochs={args.epochs}, LR={args.lr}")
    # logger.info(f"Data Root: {args.img_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    logger.info("Parsing dataset annotation file...")
    train_paths, train_boxes = parse_wider_face(args.label_path_train, args.img_root_train)

    logger.info("parsing Validation Data")
    val_paths, val_boxes = parse_wider_face(args.label_path_val, args.img_root_val)

    train_dataset = WiderFaceDataset(train_paths, train_boxes, is_train=True, target_size=640)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate, pin_memory=True)

    val_dataset = WiderFaceDataset(val_paths, val_boxes, is_train=False, target_size=640)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate, pin_memory=True)

    logging.info(f"Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    logger.info("Initializing Model SCRFD-2.5GF...")
    model = SCRFD(variant='2.5GF').to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[440, 544])

    anchor_gen = SCRFDAnchorGenerator()
    assigner = ATSSAssigner(topk=9)
    criterion = SCRFDLoss().to(device)

    logger.info("Initialization Complete. Starting Training Loop...")

    start_epoch = 0
    best_val_loss = float('inf')
    epoch_multi = 8
    total_epochs = args.epochs * epoch_multi


    for epoch in range(total_epochs): 
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_reg = 0.0

        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch_idx, (imgs, targets) in enumerate(prog_bar): 
            try: 
                imgs = imgs.to(device)

                cls_outs, reg_outs = model(imgs)
                flat_cls, flat_reg = process_model_output(cls_outs, reg_outs)

                feat_sizes = [out.shape[-2:] for out in cls_outs]
                anchors = anchor_gen.generate_anchors(feat_sizes, devices=device)
                num_anchors_per_level = [s[0] * s[1]*2 for s in feat_sizes]

                batch_loss = 0.0
                batch_cls = 0.0
                batch_reg = 0.0

                for b in range(imgs.size(0)):
                    curr_cls = flat_cls[b]
                    curr_reg = flat_reg[b]
                    curr_gt = targets[b].to(device)

                    with torch.no_grad(): 
                        assigned_gt_inds, _ = assigner(anchors, num_anchors_per_level, curr_gt)

                    total, l_cls, l_reg = criterion(curr_cls, curr_reg, anchors, assigned_gt_inds, curr_gt)

                    batch_loss += total
                    batch_cls += l_cls
                    batch_reg += l_reg

                batch_loss /= imgs.size(0)
                batch_cls /= imgs.size(0)
                batch_reg /= imgs.size(0)

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
                optimizer.step()

                epoch_loss += batch_loss.item()
                epoch_cls += batch_cls.item()
                epoch_reg += batch_reg.item()

                prog_bar.set_postfix(loss=f"{batch_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.5f}")
            except Exception as e: 
                logger.error(f"error in batch {batch_idx}: {e}")
                continue
        avg_train_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.5f} (LR: {current_lr:.6f})")
        scheduler.step()

        if (epoch + 1) % 80 == 0:
            val_loss, val_cls, val_reg = evaluate(model, val_loader, device, anchor_gen, assigner, criterion)
            logger.info(f"Epoch {epoch+1} Validation | Loss: {val_loss:.5f} | Cls: {val_cls:.5f} | Reg: {val_reg:.5f}")
            save_path = f"weights/scrfd_epoch_{epoch+1}.pth"
            os.makedirs("weights", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, save_path)
            logger.info(f"Checkpoint saved: {save_path}")

            if val_loss < best_val_loss: 
                best_val_loss = val_loss
                best_path = "weights/scrfd_best.pth"
                torch.save(model.state_dict(), best_path)
                logger.info(f"New Best Model Saved! (Loss: {val_loss:.5f})")
    logger.info("======Training FINISHED=====")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root_train', type=str, default='data/WIDER_train/images')
    parser.add_argument('--label_path_train', type=str, default='data/WIDER_train/labelv2.txt') # Ganti ke label asli nanti
    parser.add_argument('--img_root_val', type=str, default='data/WIDER_val/images')
    parser.add_argument('--label_path_val', type=str, default='data/WIDER_val/labelv2.txt')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (Default 8 untuk GPU 8GB-an)')
    parser.add_argument('--epochs', type=int, default=80, help='Total epochs (Paper: 640)')
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning Rate (0.0025 untuk Batch 8)')
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--resume', type=str, default=None, help='Path ke file .pth untuk resume training')
    parser.add_argument('--log_dir', type=str, default='logs')

    args = parser.parse_args()
    main(args)



                               

        


