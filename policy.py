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

    def forward(self, xt, masked_image, mask):
        x = torch.cat([xt, masked_image, mask], dim=1)
        pred = self.model(x)
        return pred

    def extract(self, a, t, x_shape):
        """ Extract a[t] and reshape to [B, 1, 1, 1] for broadcasting """
        b = t.shape[0]
        out = a.gather(0, t)
        return out.view(b, *((1,) * (len(x_shape) - 1)))

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def get_target(self, x0, noise):
        if self.pred_type == "x0":
            return x0
        elif self.pred_type == "eps":
            return noise
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

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
        xt, noise = self.q_sample(x0, t)

        pred = self.forward(xt, masked_image, mask)
        target = self.get_target(x0, noise)

        if self.pred_type == "x0":
            pred_img = pred
        elif self.pred_type == "eps":
            alpha_bar_t = self.extract(self.alpha_bars, t, xt.shape)
            pred_img = (xt - torch.sqrt(1.0 - alpha_bar_t) * pred) / torch.sqrt(alpha_bar_t)
        else:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        l1 = F.l1_loss(pred_img, x0)
        perc = self.perceptual_loss(pred_img, x0)

        return {"l1": l1, "perc": perc}

    def model_predict(self, xt, masked_image, mask, t):
        """ Model prediction """
        pred = self.forward(xt, masked_image, mask)

        alpha_bar_t = self.extract(self.alpha_bars, t, xt.shape)

        if self.pred_type == "x0":
            x0_pred = pred
            eps_pred = (xt - torch.sqrt(alpha_bar_t) * x0_pred) / torch.sqrt(1.0 - alpha_bar_t)
        elif self.pred_type == "eps":
            eps_pred = pred
            x0_pred = (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
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

        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            xt, x0_pred, _ = self.p_sample_step(xt, masked_image, mask, t)

            if return_trajectory:
                trajectory.append(x0_pred.detach().cpu())

        final_x0 = xt

        if return_trajectory:
            return final_x0, trajectory
        return final_x0

    @torch.no_grad()
    def predict_eps(self, masked_image, mask, xT=None, return_x0=False):
        """
        Full iterative reverse diffusion process and return final epsilon prediction.

        Args:
            masked_image: [B, 3, H, W]
            mask:         [B, 1, H, W]
            xT:           optional initial noise
            return_x0:    if True, also return final reconstructed image

        Returns:
            final_eps
            or (final_eps, final_x0)
        """
        device = masked_image.device
        b, c, h, w = masked_image.shape

        if xT is None:
            xt = torch.randn((b, c, h, w), device=device)
        else:
            xt = xT.to(device)

        xt = masked_image + mask * xt

        final_eps = None

        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            xt, _, eps_pred = self.p_sample_step(xt, masked_image, mask, t)
            final_eps = eps_pred

        final_x0 = xt

        if return_x0:
            return final_eps, final_x0
        return final_eps