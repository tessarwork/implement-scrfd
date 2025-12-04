import torch
# from architecture.resnet import *
from architecture.test_architecture import SCRFD

class SCRFDAnchorGenerator: 
    def __init__(self, strides=[8, 16, 32]):
        self.strides = strides
        self.anchors_per_stride = { 
            8:  [16, 32],
            16: [64, 128],
            32: [256, 512]
        }
    def generate_anchors(self, feature_maps_sizes, devices='cpu'): 
        all_anchors = []
        for stride, size in zip(self.strides, feature_maps_sizes): 
            height, width = size
            # print(f"[INFO]Size: {size}")
           
            scales = self.anchors_per_stride[stride]
            # print(f"[INFO]scales: {scales}")

            ys = torch.arange(0, height, device=devices) * stride
            xs = torch.arange(0, width, device=devices) * stride

            # print(f"xs_before: {xs}, ys_before: {ys}")

            ys = ys+stride/2
            xs = xs+stride/2
            # print(f"xs_after: {xs}, ys_after: {ys}")



            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid_x = grid_x.reshape(-1)
            grid_y = grid_y.reshape(-1)
            # print(f"coordiante grid_x and grid_y: {grid_x}, {grid_y}")


            num_anchors = len(scales)
            # print(f"[INFO]num_anchors: {num_anchors}")

            center_x = grid_x.view(-1, 1).repeat(1, num_anchors).view(-1)
            center_y = grid_y.view(-1, 1).repeat(1, num_anchors).view(-1)

            # print(f"[INFO] center_x: {center_x}, center_y: {center_y}")

            scale_tensor =  torch.tensor(scales, device=devices).float()
            wh = scale_tensor.view(1, -1).repeat(height * width, 1).view(-1)
            # print(f"[INFO] scale tensor: {scale_tensor.shape}")
            # print(f"[INFO] wh: {wh.shape}")

            stride_anchors = torch.stack([center_x, center_y, wh, wh], dim=-1)
            # print(f"[INFO] stride anchors: {stride_anchors.shape}")
            all_anchors.append(stride_anchors)
            real_anchors = torch.cat(all_anchors, dim=0)
            # print(f"[INFO] real anchors: {real_anchors.shape}")


        return real_anchors
    
if __name__ == "__main__": 
    model = SCRFD(variant='2.5GF')
    dummy_input = torch.randn(1, 3, 224, 224)

    cls_outs, reg_outs = model(dummy_input)

    features_maps_sizes = [out.shape[-2:] for out in cls_outs]

    # print(f"Feature Map Sizes dari Model: {features_maps_sizes}")

    anchor_gen = SCRFDAnchorGenerator()
    anchors = anchor_gen.generate_anchors(features_maps_sizes)
    total_predictions = 0
    for out in cls_outs: 
        total_predictions += (out.shape[2] * out.shape[3] * 2)
    print(f"Total Prediksi Model (Logits): {total_predictions}")
    print(f"Total Anchors Generated:       {anchors.shape[0]}")

    if total_predictions == anchors.shape[0]:
        print(">> MATCH! Struktur Model dan Anchor sudah sinkron.")
    else:
        print(">> ERROR! Jumlah prediksi dan anchor tidak sama.")
    
    # print("Model SCRFD-2.5GF Initialized.")
    # for i, stride in enumerate([8, 16, 32]):
    #     print(f"\nStride {stride} Output:")
    #     print(f"  Class Shape: {cls_outs[i].shape}") # Expect: [1, 2, H/s, W/s]
    #     print(f"  Reg Shape:   {reg_outs[i].shape}") # Expect: [1, 8, H/s, W/s] (4 coords * 2 anchors)

