import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1): 
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module): 
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: 
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResnetBackbone(nn.Module): 
    def __init__(self, block, layers, width_mult=1.0): 
        super(ResnetBackbone, self).__init__()
        self.inplanes = int(64 * width_mult)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        base_channels = [64, 128, 256, 512]
        filters = [int(c * width_mult) for c in base_channels]

        self.layer1 = self._make_layer(block, filters[0], layers[0])
        self.layer2 = self._make_layer(block, filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters[3], layers[3], stride=2)

        self.out_channels = filters

    def _make_layer(self, block, planes, blocks, stride=1): 
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( 
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): 
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x) # Stride 4
        c3 = self.layer2(c2) # Stride 8 (Feature map penting untuk wajah kecil [cite: 176])
        c4 = self.layer3(c3) # Stride 16
        c5 = self.layer4(c4) # Stride 32

        return [c3, c4, c5]


class PAFPN(nn.Module): 
    def __init__(self, in_channels_list, out_channels=96):
        super(PAFPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        for in_c in in_channels_list: 
            self.lateral_convs.append(nn.Conv2d(in_c, out_channels, 1))
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list: 
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.downsample_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1): 
            self.downsample_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))
    def forward(self, inputs): 
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): 
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')
        inter_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        outs = [inter_outs[0]]
        for i in range(used_backbone_levels - 1):
            prev = outs[-1]
            downsampled = self.downsample_convs[i](prev)
            current = inter_outs[i+1] + downsampled
            outs.append(current)
            
        return outs # [P3, P4, P5]
    
class SCRFDHead(nn.Module): 
    def __init__(self, in_channels=96, num_anchors = 2, num_layers=2): 
        super(SCRFDHead, self).__init__()
        self.num_anchors = num_anchors

        cls_tower = []
        reg_tower = []
        for _ in range(num_layers): 
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, in_channels)) # GN 
            cls_tower.append(nn.ReLU(inplace=True))

            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_pred = nn.Conv2d(in_channels, num_anchors * 1, 3, padding=1) 
        
        # Reg: 4 koordinat (x, y, w, h) per anchor
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        
        # Scale initialization (Optional tapi bagus untuk training stabil)
        self.scales = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(3)])

    def forward(self, inputs):
        # inputs: [P3, P4, P5] dari Neck
        cls_logits = []
        bbox_preds = []
        
        for x, scale in zip(inputs, self.scales):
            # Weight sharing: Tower yang sama dipanggil untuk setiap input feature map
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)
            
            # Prediksi
            cls = self.cls_pred(cls_feat)
            reg = self.reg_pred(reg_feat)
            
            # Apply learnable scale per level (biar regression stabil antar stride)
            reg = reg * scale 
            
            cls_logits.append(cls)
            bbox_preds.append(reg)
            
        return cls_logits, bbox_preds
    

