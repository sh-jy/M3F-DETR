import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class fdgd(nn.Module):
    def __init__(
        self,
        dim: int,
        kappa: float = 12.0,
        spectral_eps: float = 1e-6,
        sparsity_threshold: float = 1e-4,
        alpha_init: float = 0.5,
        learnable_alpha: bool = True,
        learnable_kappa: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if not (0.0 <= alpha_init <= 1.0):
            raise ValueError("alpha_init must be in [0, 1].")

        self.dim = dim
        self.spectral_eps = spectral_eps
        self.sparsity_threshold = sparsity_threshold

        self.global_norm = nn.LayerNorm(dim)
        self.tau_proj = nn.Linear(dim, 1, bias=True)

        alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.logit(alpha_tensor))
        else:
            self.register_buffer("alpha_logit", torch.logit(alpha_tensor), persistent=True)

        kappa_tensor = torch.tensor(kappa, dtype=torch.float32)
        if learnable_kappa:
            self.log_kappa = nn.Parameter(torch.log(kappa_tensor))
        else:
            self.register_buffer("log_kappa", torch.log(kappa_tensor), persistent=True)

        self.noise_gate_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.edge_gate_conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.output_proj = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=use_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.tau_proj.weight)
        nn.init.zeros_(self.tau_proj.bias)

        nn.init.kaiming_normal_(self.noise_gate_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.edge_gate_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.output_proj.weight)

        if self.noise_gate_conv.bias is not None:
            nn.init.zeros_(self.noise_gate_conv.bias)
        if self.edge_gate_conv.bias is not None:
            nn.init.zeros_(self.edge_gate_conv.bias)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    @property
    def kappa(self) -> torch.Tensor:
        return torch.exp(self.log_kappa)

    @staticmethod
    def _check_input(x: torch.Tensor, dim: int) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B, C, H, W], but got {tuple(x.shape)}.")
        if x.shape[1] != dim:
            raise ValueError(f"Expected channel dimension {dim}, but got {x.shape[1]}.")

    @staticmethod
    def _compute_observation_map(x: torch.Tensor) -> torch.Tensor:
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        fg = avg_pool + max_pool
        return fg

    @staticmethod
    def _fft2_shift(x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        return x_fft

    @staticmethod
    def _ifft2_from_shifted(x_fft_shifted: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.ifftshift(x_fft_shifted, dim=(-2, -1))
        x = torch.fft.ifft2(x_fft, dim=(-2, -1), norm="ortho")
        return x

    def _compute_dynamic_tau(self, x: torch.Tensor) -> torch.Tensor:
        z = x.mean(dim=(2, 3))
        z = self.global_norm(z)
        tau = torch.sigmoid(self.tau_proj(z))
        return tau

    @staticmethod
    def _build_radial_frequency_map(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        cy = height / 2.0
        cx = width / 2.0
        denom = math.sqrt((height / 2.0) ** 2 + (width / 2.0) ** 2) + 1e-12

        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / denom
        r = r.unsqueeze(0).unsqueeze(0)
        return r

    def _compute_frequency_masks(
        self,
        tau: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self._build_radial_frequency_map(height, width, device, dtype)
        tau = tau.view(-1, 1, 1, 1)
        kappa = self.kappa.to(device=device, dtype=dtype)
        m_low = torch.sigmoid(kappa * (tau - r))
        m_high = 1.0 - m_low
        return m_low, m_high, r

    def _reconstruct_spatial_prior(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        complex_spectrum = torch.polar(amplitude, phase)
        spatial = self._ifft2_from_shifted(complex_spectrum).real
        spatial = self._minmax_normalize(spatial)
        return spatial

    def _compute_channel_spectral_attention(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_fft = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        energy = x_fft.abs().pow(2)

        sparsity = (energy > self.sparsity_threshold).to(x.dtype).mean(dim=(-2, -1))
        mean_energy = energy.mean(dim=(-2, -1))
        variance = ((energy - mean_energy.unsqueeze(-1).unsqueeze(-1)) ** 2).mean(dim=(-2, -1))

        sparsity_norm = sparsity / (sparsity.sum(dim=1, keepdim=True) + self.spectral_eps)
        variance_norm = variance / (variance.sum(dim=1, keepdim=True) + self.spectral_eps)

        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        spectral_score = alpha * sparsity_norm + (1.0 - alpha) * variance_norm
        omega = torch.softmax(spectral_score, dim=1)

        x_reweighted = x * omega.unsqueeze(-1).unsqueeze(-1)

        aux = {
            "energy": energy,
            "sparsity": sparsity,
            "mean_energy": mean_energy,
            "variance": variance,
            "sparsity_norm": sparsity_norm,
            "variance_norm": variance_norm,
            "spectral_score": spectral_score,
            "omega": omega,
        }
        return x_reweighted, aux

    def _directional_interaction(
        self,
        fg: torch.Tensor,
        m_low: torch.Tensor,
        m_high: torch.Tensor,
        x_reweighted: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        f_high = m_high * x_reweighted
        f_low = m_low * x_reweighted

        g_n = torch.sigmoid(self.noise_gate_conv(torch.cat([m_low, fg], dim=1)))
        g_e = torch.sigmoid(self.edge_gate_conv(torch.cat([m_high, fg], dim=1)))

        f_high_tilde = g_n * f_high
        f_low_tilde = g_e * f_low

        fused = self.output_proj(torch.cat([f_high_tilde, f_low_tilde], dim=1))

        aux = {
            "f_high": f_high,
            "f_low": f_low,
            "g_n": g_n,
            "g_e": g_e,
            "f_high_tilde": f_high_tilde,
            "f_low_tilde": f_low_tilde,
            "fused": fused,
        }
        return fused, aux

    def _minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
        x_max = x.amax(dim=(-3, -2, -1), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + self.spectral_eps)
        x_norm = x_norm.clamp(0.0, 1.0)
        return x_norm

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self._check_input(x, self.dim)

        _, _, h, w = x.shape

        fg = self._compute_observation_map(x)

        fg_fft = self._fft2_shift(fg)
        amplitude = fg_fft.abs()
        phase = torch.angle(fg_fft)

        tau = self._compute_dynamic_tau(x)

        m_low_freq, m_high_freq, radial_map = self._compute_frequency_masks(
            tau=tau,
            height=h,
            width=w,
            device=x.device,
            dtype=x.dtype,
        )

        amplitude_low = m_low_freq * amplitude
        amplitude_high = m_high_freq * amplitude

        m_low = self._reconstruct_spatial_prior(amplitude_low, phase)
        m_high = self._reconstruct_spatial_prior(amplitude_high, phase)

        x_reweighted, spectral_aux = self._compute_channel_spectral_attention(x)

        fused_residual, interaction_aux = self._directional_interaction(
            fg=fg,
            m_low=m_low,
            m_high=m_high,
            x_reweighted=x_reweighted,
        )
        out = x + fused_residual

        if not return_intermediates:
            return out

        aux: Dict[str, torch.Tensor] = {
            "fg": fg,
            "fg_fft_real": fg_fft.real,
            "fg_fft_imag": fg_fft.imag,
            "amplitude": amplitude,
            "phase": phase,
            "tau": tau,
            "radial_map": radial_map,
            "m_low_freq": m_low_freq,
            "m_high_freq": m_high_freq,
            "amplitude_low": amplitude_low,
            "amplitude_high": amplitude_high,
            "m_low": m_low,
            "m_high": m_high,
            "x_reweighted": x_reweighted,
            "alpha": self.alpha.detach().view(1),
            "kappa": self.kappa.detach().view(1),
        }
        aux.update(spectral_aux)
        aux.update(interaction_aux)
        return out, aux