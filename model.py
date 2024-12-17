import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(GatedConv2d, self).__init__()
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = self.conv_gate(x)
        gate = torch.sigmoid(gate)
        return feat * gate

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.out_channels = out_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.gated_conv = GatedConv2d(len(in_channels_list)*out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        ref_size = features[0].shape[-2:]
        fused = []
        for i, f in enumerate(features):
            f = self.convs[i](f)
            if f.shape[-2:] != ref_size:
                f = F.interpolate(f, size=ref_size, mode='bilinear', align_corners=False)
            fused.append(f)
        fused = torch.cat(fused, dim=1)
        fused = self.gated_conv(fused)
        return fused

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            GatedConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            GatedConv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        in_block_ch = out_ch + skip_ch
        self.block = nn.Sequential(
            GatedConv2d(in_block_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            GatedConv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class GatedMultiScaleSegmentationNet(nn.Module):
    def __init__(self, num_classes=2, base_ch=64):
        super(GatedMultiScaleSegmentationNet, self).__init__()
        self.down1 = DownBlock(3, base_ch)
        self.down2 = DownBlock(base_ch, base_ch*2)
        self.down3 = DownBlock(base_ch*2, base_ch*4)
        self.down4 = DownBlock(base_ch*4, base_ch*8)
        self.bottom = DownBlock(base_ch*8, base_ch*8)
        self.msf = MultiScaleFusion([base_ch*2, base_ch*4, base_ch*8], base_ch*4)
        self.post_fusion_gated_conv = GatedConv2d(base_ch*8 + base_ch*4, base_ch*4, kernel_size=3, padding=1)

        self.up3 = UpBlock(in_ch=256, skip_ch=512, out_ch=256)
        self.up2 = UpBlock(in_ch=256, skip_ch=256, out_ch=128)
        self.up1 = UpBlock(in_ch=128, skip_ch=128, out_ch=64)
        
        self.final_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        self.final_gated_conv = GatedConv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)

        x5 = self.pool(x4)
        x5 = self.bottom(x5)

        fused = self.msf([x2, x3, x4])
        fused_up = F.interpolate(fused, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x_mid = torch.cat([x5, fused_up], dim=1)
        x_mid = self.post_fusion_gated_conv(x_mid)
        
        x = self.up3(x_mid, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)

        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, x1], dim=1)
        x = self.final_gated_conv(x)
        x = self.final_conv(x)
        return x

def get_model(num_classes=2, device='cuda'):
    model = GatedMultiScaleSegmentationNet(num_classes=num_classes)
    model = model.to(device)
    return model
