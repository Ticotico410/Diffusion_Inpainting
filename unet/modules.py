import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, time_dim=256, num_groups=32):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        # group 数不能大于 channel 数
        g1 = min(num_groups, mid_channels)
        g2 = min(num_groups, out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(g1, mid_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(g2, out_channels),
            nn.SiLU(inplace=True),
        )

        self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_channels)
            )

    def forward(self, x, t_emb=None):
        x = self.double_conv(x)

        if t_emb is not None:
            t_out = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
            x = x + t_out

        return x

    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_dim=time_dim)

    def forward(self, x, t_emb=None):
        x = self.maxpool(x)
        x = self.conv(x, t_emb)
        return x
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, time_dim=256):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, time_dim=time_dim)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, time_dim=time_dim)

    def forward(self, x1, x2, t_emb=None):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_emb)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    


# for the timestep embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        return: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb
    

class TimeMLP(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        return self.net(t)