from architecture.resnet import BasicBlock, ResnetBackbone, PAFPN, SCRFDHead
import torch
import torch.nn as nn
import torch.nn.functional as F
class SCRFD(nn.Module):
    def __init__(self, variant='2.5GF'):
        super(SCRFD, self).__init__()
        
        # Konfigurasi sesuai Tabel 2 Paper 
        if variant == '2.5GF':
            # ResNet-34 structure (layers=[3,4,6,3]), width multiplier 0.25
            backbone_width = 0.25
            backbone_layers = [3, 4, 6, 3]
            neck_channels = 96 # 
        elif variant == '10GF':
             # ResNet-34 structure, width multiplier 0.5
            backbone_width = 0.5
            backbone_layers = [3, 4, 6, 3]
            neck_channels = 128
        else:
            raise ValueError("Varian belum disupport di contoh ini.")

        # 1. Init Backbone
        self.backbone = ResnetBackbone(BasicBlock, backbone_layers, width_mult=backbone_width)
        
        # 2. Init Neck (Input channels diambil dari output backbone)
        # ResNet channels: [64, 128, 256, 512] * width_mult
        c3_c4_c5_channels = self.backbone.out_channels[1:] # Ambil C3, C4, C5
        self.neck = PAFPN(c3_c4_c5_channels, out_channels=neck_channels)
        
        # 3. Init Head
        # Paper: anchor {16,32} @ S8, {64,128} @ S16, {256,512} @ S32 -> 2 anchor/loc 
        self.head = SCRFDHead(in_channels=neck_channels, num_anchors=2)

    def forward(self, x):
        # Forward pass
        features = self.backbone(x)      # [C3, C4, C5]
        fpn_feats = self.neck(features)  # [P3, P4, P5]
        cls_logits, bbox_preds = self.head(fpn_feats)
        
        return cls_logits, bbox_preds

# --- TESTING MODEL ---
if __name__ == "__main__":
    # Buat model SCRFD-2.5GF
    model = SCRFD(variant='2.5GF')
    
    # Dummy Input (Batch 1, RGB, 640x640)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Forward
    cls_outs, reg_outs = model(dummy_input)
    
    print("Model SCRFD-2.5GF Initialized.")
    for i, stride in enumerate([8, 16, 32]):
        print(f"\nStride {stride} Output:")
        print(f"  Class Shape: {cls_outs[i].shape}") # Expect: [1, 2, H/s, W/s]
        print(f"  Reg Shape:   {reg_outs[i].shape}") # Expect: [1, 8, H/s, W/s] (4 coords * 2 anchors)