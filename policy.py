import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from unet.model import UNet


class DiffusionInpaintPolicy(nn.Module):
    def __init__(
        self,
        pred_type: str = "x0",
        image_channels: int = 3,
        mask_channels: int = 1,
        bilinear: bool = False,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        no_skip: bool = False
    ):
        super().__init__()

        if pred_type not in ["x0", "eps"]:
            raise ValueError("pred_type must be 'x0' or 'eps'")

        self.pred_type = pred_type
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.timesteps = timesteps

        in_channels = image_channels + image_channels + mask_channels   # xt + masked_image + mask
        out_channels = image_channels

        self.model = UNet(
            n_channels=in_channels,
            n_classes=out_channels,
            bilinear=bilinear,
            no_skip=no_skip,
        )

        # diffusion schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)                                   # [T]
        self.register_buffer("alphas", alphas)                                 # [T]
        self.register_buffer("alpha_bars", alpha_bars)                         # [T]
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))        # [T]
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))  # [T]

        # perceptual network: VGG16 intermediate features
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.perceptual_net = models.vgg16(weights=weights).features[:16].eval()
        for p in self.perceptual_net.parameters():
            p.requires_grad = False

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )


    def forward(self, xt, masked_image, mask, t):
        x = torch.cat([xt, masked_image, mask], dim=1)
        pred = self.model(x, t)
        return pred
    

    def extract(self, a, t, x_shape):
        """ Extract a[t] and reshape to [B, 1, 1, 1] for broadcasting """
        b = t.shape[0]
        out = a.gather(0, t)
        return out.view(b, *((1,) * (len(x_shape) - 1)))

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()

    def q_sample(self, x0, t, mask, noise=None):
        """
        Forward diffusion:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        noise = noise * mask

        sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def _to_imagenet_input(self, x):
        """ Convert image from [-1, 1] to ImageNet-normalized input """
        x = (x + 1.0) / 2.0
        x = (x - self.imagenet_mean) / self.imagenet_std
        return x

    def perceptual_loss(self, x_pred, x0):
        x_pred_feat = self.perceptual_net(self._to_imagenet_input(x_pred))
        x0_feat = self.perceptual_net(self._to_imagenet_input(x0))
        return F.l1_loss(x_pred_feat, x0_feat)

    def compute_loss(self, masked_image, mask, x0):
        b = x0.shape[0]
        device = x0.device

        t = self.sample_timesteps(b, device)
        xt, noise = self.q_sample(x0, t, mask)

        pred = self.forward(xt, masked_image, mask, t)

        if self.pred_type == "x0":
            pred_img = pred
            # l1 loss
            diff = torch.abs(pred_img - x0) * mask
            hole_loss = diff.sum() / (mask.sum() * x0.shape[1] + 1e-8)
            l1 = F.l1_loss(pred_img, x0) + 6 * hole_loss
            # perceptual loss
            perc = self.perceptual_loss(pred_img, x0)
        elif self.pred_type == "eps":
            alpha_bar_t = self.extract(self.alpha_bars, t, xt.shape)
            pred_img = (xt - torch.sqrt(1.0 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
            pred_img = pred_img.clamp(-1.0, 1.0)
            # directly use mse loss of noise
            mask_expand = mask.expand_as(pred)
            eps_loss = ((pred - noise) ** 2 * mask_expand).sum() / (mask_expand.sum() + 1e-8)
            recon_loss = (torch.abs(pred_img - x0) * mask_expand).sum() / (mask_expand.sum() + 1e-8)
            l1 = 6 * eps_loss + recon_loss         # name it l1 loss for code clean
            # perceptual loss
            # perc = torch.tensor(0.0, device=x0.device)
            perc = self.perceptual_loss(pred_img, x0)
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        return {"l1": l1, "perc": perc}

    def model_predict(self, xt, masked_image, mask, t):
        """ Model prediction """
        pred = self.forward(xt, masked_image, mask, t)

        alpha_bar_t = self.extract(self.alpha_bars, t, xt.shape)

        if self.pred_type == "x0":
            x0_pred = pred.clamp(-1.0, 1.0)
            eps_pred = (xt - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1.0 - alpha_bar_t)
        elif self.pred_type == "eps":
            eps_pred = pred
            x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        return x0_pred, eps_pred

    @torch.no_grad()
    def p_sample_step(self, xt, masked_image, mask, t):
        """ DDPM sampling step """
        b = xt.shape[0]
        device = xt.device

        x0_pred, eps_pred = self.model_predict(xt, masked_image, mask, t)

        alpha_t = self.extract(self.alphas, t, xt.shape)
        alpha_bar_t = self.extract(self.alpha_bars, t, xt.shape)
        beta_t = self.extract(self.betas, t, xt.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        noise = torch.randn_like(xt)
        # try to use no noise added
        # noise = torch.zeros_like(xt)
        nonzero_mask = (t > 0).float().view(b, 1, 1, 1)
        xt_prev = mean + nonzero_mask * torch.sqrt(beta_t) * noise

        xt_prev = masked_image + mask * xt_prev

        return xt_prev, x0_pred, eps_pred

    @torch.no_grad()
    def predict_x0(self, masked_image, mask, xT=None, return_trajectory=False):
        """
        Full iterative reverse diffusion process and return final reconstructed image.

        Args:
            masked_image: [B, 3, H, W]
            mask:         [B, 1, H, W]
            xT:           optional initial noise, [B, 3, H, W]
            return_trajectory: whether to return intermediate x0 predictions

        Returns:
            final_x0 or (final_x0, trajectory)
        """
        device = masked_image.device
        b, c, h, w = masked_image.shape

        if xT is None:
            xt = torch.randn((b, c, h, w), device=device)
        else:
            xt = xT.to(device)

        xt = masked_image + mask * xt

        trajectory = []
        input_image = []

        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            xt, x0_pred, _ = self.p_sample_step(xt, masked_image, mask, t)

            if return_trajectory:
                trajectory.append(x0_pred.clamp(-1.0, 1.0).detach().cpu())
                input_image.append(xt.detach().cpu())

        final_x0 = masked_image + mask * x0_pred
        final_x0 = final_x0.clamp(-1.0, 1.0)

        if return_trajectory:
            return final_x0, trajectory, input_image
        return final_x0


class DirectPredictPolicy(nn.Module):
    def __init__(
            self,
            image_channels: int = 3,
            mask_channels: int = 1,
            bilinear: bool = False,
            no_skip: bool = False,
        ):
        super().__init__()
        in_channels = image_channels + mask_channels
        out_channels = image_channels
        self.model = UNet(
            n_channels=in_channels,
            n_classes=out_channels,
            bilinear=bilinear,
            time_dim=0,
            no_skip=no_skip,
        )

        # perceptual network: VGG16 intermediate features
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.perceptual_net = models.vgg16(weights=weights).features[:16].eval()
        for p in self.perceptual_net.parameters():
            p.requires_grad = False
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, masked_image, mask):
        x = torch.cat([masked_image, mask], dim=1)
        x0_pred = self.model(x)
        return x0_pred
    
    def _to_imagenet_input(self, x):
        """ Convert image from [-1, 1] to ImageNet-normalized input """
        x = (x + 1.0) / 2.0
        x = (x - self.imagenet_mean) / self.imagenet_std
        return x
    
    def perceptual_loss(self, x_pred, x0):
        x_pred_feat = self.perceptual_net(self._to_imagenet_input(x_pred))
        x0_feat = self.perceptual_net(self._to_imagenet_input(x0))
        return F.l1_loss(x_pred_feat, x0_feat)
    
    def compute_loss(self, masked_image, mask, x0):
        x_pred = self.forward(masked_image, mask)
        l1 = F.l1_loss(x_pred, x0)
        # perceptual loss
        perc = self.perceptual_loss(x_pred, x0)

        return {"l1": l1, "perc": perc} 

    @torch.no_grad()
    def predict_x0(self, masked_image, mask, xT=None, return_trajectory=False):
        x_pred = self.forward(masked_image, mask)

        x_pred = mask * x_pred + masked_image * (1 - mask)
        return x_pred