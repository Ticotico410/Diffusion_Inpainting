from unet.modules import *
from torch import nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, time_dim=256):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.time_dim = time_dim

        self.time_mlp = TimeMLP(time_dim)

        self.inc = ResBlock(n_channels, 64, time_dim=time_dim)
        self.down1 = Down(64, 128, time_dim=time_dim)
        self.down2 = Down(128, 256, time_dim=time_dim)
        self.down3 = Down(256, 512, time_dim=time_dim)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, time_dim=time_dim)
        self.up1 = Up(1024, 512 // factor, bilinear, time_dim=time_dim)
        self.up2 = Up(512, 256 // factor, bilinear, time_dim=time_dim)
        self.up3 = Up(256, 128 // factor, bilinear, time_dim=time_dim)
        self.up4 = Up(128, 64, bilinear, time_dim=time_dim)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)   # [B, time_dim]

        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)

        logits = self.outc(x)
        return logits