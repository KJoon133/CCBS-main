import torch.nn as nn
import torch
import sys

sys.path.append("/home/railab/Workspace/CCBS/code/")

from core.model.unet_parts import *

class UNet_2D_256(nn.Module):
    def __init__(self, n_channels, n_classes, coord_dim, bilinear=False):
        super(UNet_2D_256, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))
        self.down3 = (Down(16, 32))
        factor = 2 if bilinear else 1
        self.down4 = (Down(32, 64 // factor))
        
        self.up1 = (Up(64, 32 // factor, bilinear))
        self.up2 = (Up(32, 16 // factor, bilinear))
        self.up3 = (Up(16, 8 // factor, bilinear))
        self.up4 = (Up(8, 4, bilinear))
        self.outc = (OutConv(4, n_classes))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16384 + coord_dim, 16384)

    def forward(self, x, coord_data):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        flat_x = self.flatten(x5)
        # print(flat_x.shape) 

        cat_x = torch.cat((flat_x, coord_data), dim=1)
        # print(cat_x.shape)

        x5 = self.linear(cat_x)
        # print(x5.shape)


        x5 = x5.reshape(x5.shape[0], -1, 16, 16)

        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        out = self.outc(x)
        # print(out.shape)
        return out



class UNet_2D_128(nn.Module):
    def __init__(self, n_channels, n_classes, coord_dim, bilinear=False):
        super(UNet_2D_128, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))
        self.down3 = (Down(16, 32))
        factor = 2 if bilinear else 1
        self.down4 = (Down(32, 64 // factor))
        
        self.up1 = (Up(64, 32 // factor, bilinear))
        self.up2 = (Up(32, 16 // factor, bilinear))
        self.up3 = (Up(16, 8 // factor, bilinear))
        self.up4 = (Up(8, 4, bilinear))
        self.outc = (OutConv(4, n_classes))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4096 + coord_dim, 4096)

    def forward(self, x, coord_data):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        flat_x = self.flatten(x5)
        # print(flat_x.shape) 

        cat_x = torch.cat((flat_x, coord_data), dim=1)
        # print(cat_x.shape)

        x5 = self.linear(cat_x)
        # print(x5.shape)

        x5 = x5.reshape(x5.shape[0], -1, 8, 8)

        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        out = self.outc(x)
        # print(out.shape)
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_2D_64(nn.Module):
    def __init__(self, n_channels, n_classes, coord_dim, bilinear=False):
        super(UNet_2D_64, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (Down(4, 8))
        self.down2 = (Down(8, 16))
        self.down3 = (Down(16, 32))
        factor = 2 if bilinear else 1
        self.down4 = (Down(32, 64 // factor))
        
        self.up1 = (Up(64, 32 // factor, bilinear))
        self.up2 = (Up(32, 16 // factor, bilinear))
        self.up3 = (Up(16, 8 // factor, bilinear))
        self.up4 = (Up(8, 4, bilinear))
        self.outc = (OutConv(4, n_classes))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024 + coord_dim, 1024)

    def forward(self, x, coord_data):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        flat_x = self.flatten(x5)
        # print(flat_x.shape) 

        cat_x = torch.cat((flat_x, coord_data), dim=1)
        # print(cat_x.shape)

        x5 = self.linear(cat_x)
        # print(x5.shape)


        x5 = x5.reshape(x5.shape[0], -1, 4, 4)

        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        out = self.outc(x)
        # print(out.shape)
        return out


class UNet_2D_32(nn.Module):
    def __init__(self, n_channels, n_classes, coord_dim, bilinear=False):
        super(UNet_2D_32, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 32))
        self.down2 = (Down(32, 64))
        self.down3 = (Down(64, 64))
        factor = 2 if bilinear else 1
        self.down4 = (Down(64, 64 // factor))
        
        self.up1 = (Up(64, 64 // factor, bilinear))
        self.up2 = (Up(64, 64 // factor, bilinear))
        self.up3 = (Up(64, 32 // factor, bilinear))
        self.up4 = (Up(32, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024 + coord_dim, 1024)

    def forward(self, x, coord_data):
        
        print(x.shape)
        x1 = self.inc(x)
        print(x1.shape)
        x2 = self.down1(x1)
        print(x2.shape)
        x3 = self.down2(x2)
        print(x3.shape)
        x4 = self.down3(x3)
        print(x4.shape)

        flat_x = self.flatten(x4)
        print(flat_x.shape)

        cat_x = torch.cat((flat_x, coord_data), dim=1)
        print(cat_x.shape)

        x4 = self.linear(cat_x)
        print(x4.shape)

        x4 = x4.reshape(x4.shape[0], -1, 4, 4)
        print(x4.shape)

        x = self.up2(x4, x3)
        print(x.shape)
        x = self.up3(x, x2)
        print(x.shape)
        x = self.up4(x, x1)
        print(x.shape)
        out = self.outc(x)
        print(out.shape)
        return out


class UNet_2D_Sigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_2D_Sigmoid, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 64))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.up2(x, x3)
        # print(x.shape)
        x = self.up3(x, x2)
        # print(x.shape)
        x = self.up4(x, x1)
        # print(x.shape)
        out = self.outc(x)
        return nn.Sigmoid()(out)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

if __name__ == "__main__":
    import numpy as np

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    model = UNet_2D_128(n_channels=2, n_classes=1, coord_dim=102).to(device)

    conv_dummy = torch.Tensor(np.random.rand(20, 2, 128, 128)).to(device)
    coord_dummy = torch.Tensor(np.random.rand(20, 102)).to(device)

    pred = model(conv_dummy, coord_dummy)