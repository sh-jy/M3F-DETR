import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Learnable2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int, base_size: Tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        self.dim = dim
        self.base_size = base_size
        self.pos = nn.Parameter(torch.zeros(1, dim, base_size[0], base_size[1]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(
        self,
        height: int,
        width: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        pos = F.interpolate(
            self.pos,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        pos = pos.permute(0, 2, 3, 1).contiguous().view(1, height * width, self.dim)
        if device is not None or dtype is not None:
            pos = pos.to(
                device=device if device is not None else pos.device,
                dtype=dtype if dtype is not None else pos.dtype,
            )
        return pos


class ldem(nn.Module):
    def __init__(
        self,
        dim: int,
        global_downsample_ratio: int = 4,
        pos_base_size: Tuple[int, int] = (32, 32),
        upsample_mode: str = "bilinear",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if global_downsample_ratio <= 0:
            raise ValueError("global_downsample_ratio must be a positive integer.")

        self.dim = dim
        self.global_downsample_ratio = global_downsample_ratio
        self.upsample_mode = upsample_mode

        self.token_proj = nn.Linear(dim, dim, bias=use_bias)
        self.fine_pos_embed = Learnable2DPositionalEncoding(dim, base_size=pos_base_size)
        self.global_pos_embed = Learnable2DPositionalEncoding(dim, base_size=pos_base_size)

        self.context_norm = nn.LayerNorm(dim)

        self.forget_proj = nn.Linear(2 * dim, dim, bias=use_bias)
        self.write_proj = nn.Linear(2 * dim, dim, bias=use_bias)
        self.readout_gate_proj = nn.Linear(2 * dim, dim, bias=use_bias)

        self.bidir_fusion_proj = nn.Linear(3 * dim, dim, bias=use_bias)
        self.global_inject_proj = nn.Linear(3 * dim, dim, bias=use_bias)

        self.local_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=use_bias
        )
        self.local_proj = nn.Linear(dim, dim, bias=use_bias)
        self.local_fusion_proj = nn.Linear(2 * dim, dim, bias=use_bias)

        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=use_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.token_proj.weight)
        if self.token_proj.bias is not None:
            nn.init.zeros_(self.token_proj.bias)

        nn.init.xavier_uniform_(self.forget_proj.weight)
        nn.init.xavier_uniform_(self.write_proj.weight)
        nn.init.xavier_uniform_(self.readout_gate_proj.weight)
        nn.init.xavier_uniform_(self.bidir_fusion_proj.weight)
        nn.init.xavier_uniform_(self.global_inject_proj.weight)
        nn.init.xavier_uniform_(self.local_proj.weight)
        nn.init.xavier_uniform_(self.local_fusion_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

        if self.forget_proj.bias is not None:
            nn.init.zeros_(self.forget_proj.bias)
        if self.write_proj.bias is not None:
            nn.init.zeros_(self.write_proj.bias)
        if self.readout_gate_proj.bias is not None:
            nn.init.zeros_(self.readout_gate_proj.bias)
        if self.bidir_fusion_proj.bias is not None:
            nn.init.zeros_(self.bidir_fusion_proj.bias)
        if self.global_inject_proj.bias is not None:
            nn.init.zeros_(self.global_inject_proj.bias)
        if self.local_proj.bias is not None:
            nn.init.zeros_(self.local_proj.bias)
        if self.local_fusion_proj.bias is not None:
            nn.init.zeros_(self.local_fusion_proj.bias)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

        nn.init.kaiming_normal_(self.local_dwconv.weight, mode="fan_out", nonlinearity="relu")
        if self.local_dwconv.bias is not None:
            nn.init.zeros_(self.local_dwconv.bias)

    @staticmethod
    def _flatten_hw(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, C, H, W], but got shape {tuple(x.shape)}.")
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return seq, h, w

    @staticmethod
    def _reshape_to_bchw(seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if seq.ndim != 3:
            raise ValueError(f"Expected a 3D tensor [B, N, C], but got shape {tuple(seq.shape)}.")
        b, n, c = seq.shape
        if n != height * width:
            raise ValueError(
                f"Sequence length {n} does not match target spatial size {height}x{width}={height * width}."
            )
        x = seq.view(b, height, width, c).permute(0, 3, 1, 2).contiguous()
        return x

    def _interpolate(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if self.upsample_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(x, size=size, mode=self.upsample_mode, align_corners=False)
        return F.interpolate(x, size=size, mode=self.upsample_mode)

    def _build_sequence(
        self,
        x: torch.Tensor,
        pos_encoder: Learnable2DPositionalEncoding,
    ) -> Tuple[torch.Tensor, int, int]:
        seq, h, w = self._flatten_hw(x)
        pos = pos_encoder(h, w, dtype=seq.dtype, device=seq.device)
        u = self.token_proj(seq) + pos
        return u, h, w

    def _compute_gates(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        c = self.context_norm(u.mean(dim=1))
        c_bar = c.unsqueeze(1).expand(-1, u.size(1), -1)

        z = torch.cat([u, c_bar], dim=-1)
        d = F.softplus(self.forget_proj(z))
        g = torch.sigmoid(self.write_proj(z))
        o = torch.sigmoid(self.readout_gate_proj(z))

        a = torch.exp(-d)
        b = (1.0 - a) * (g * u)

        return {
            "context": c,
            "context_broadcast": c_bar,
            "d": d,
            "g": g,
            "o": o,
            "a": a,
            "b": b,
        }

    def _scan_from_gates(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        o: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = u.shape
        state = u.new_zeros(batch_size, channels)
        states = torch.empty_like(u)
        outputs = torch.empty_like(u)

        for t in range(seq_len):
            state = a[:, t, :] * state + b[:, t, :]
            y_t = o[:, t, :] * state + (1.0 - o[:, t, :]) * u[:, t, :]
            states[:, t, :] = state
            outputs[:, t, :] = y_t

        return outputs, states

    def _scan(
        self,
        u: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gate_dict = self._compute_gates(u)
        y, s = self._scan_from_gates(
            u=u,
            a=gate_dict["a"],
            b=gate_dict["b"],
            o=gate_dict["o"],
        )
        gate_dict["states"] = s
        gate_dict["y"] = y
        return gate_dict

    def _bidirectional_scan(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        forward_dict = self._scan(u)
        y_forward = forward_dict["y"]

        u_rev = torch.flip(u, dims=[1])
        backward_rev_dict = self._scan(u_rev)
        y_backward = torch.flip(backward_rev_dict["y"], dims=[1])

        rho = torch.sigmoid(self.bidir_fusion_proj(torch.cat([u, y_forward, y_backward], dim=-1)))
        y = rho * y_forward + (1.0 - rho) * y_backward

        return {
            "forward": forward_dict,
            "backward_reversed": backward_rev_dict,
            "y_forward": y_forward,
            "y_backward": y_backward,
            "rho": rho,
            "y": y,
        }

    def _global_branch(
        self,
        x: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        _, _, h, w = x.shape
        gh = max(1, math.ceil(h / self.global_downsample_ratio))
        gw = max(1, math.ceil(w / self.global_downsample_ratio))

        x_g = F.adaptive_avg_pool2d(x, output_size=(gh, gw))
        u_g, _, _ = self._build_sequence(x_g, self.global_pos_embed)

        global_scan = self._scan(u_g)
        y_g_seq = global_scan["y"]
        y_g_map = self._reshape_to_bchw(y_g_seq, gh, gw)
        y_g_up = self._interpolate(y_g_map, size=target_hw)
        y_g_up_seq, _, _ = self._flatten_hw(y_g_up)

        return {
            "x_g": x_g,
            "u_g": u_g,
            "scan": global_scan,
            "y_g_seq": y_g_seq,
            "y_g_map": y_g_map,
            "y_g_up": y_g_up,
            "y_g_up_seq": y_g_up_seq,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, C, H, W], but got {tuple(x.shape)}.")
        if x.size(1) != self.dim:
            raise ValueError(
                f"Input channel dimension {x.size(1)} does not match module dim {self.dim}."
            )

        residual = x
        _, _, h, w = x.shape

        u, _, _ = self._build_sequence(x, self.fine_pos_embed)
        long_range_dict = self._bidirectional_scan(u)
        y = long_range_dict["y"]

        global_dict = self._global_branch(x, target_hw=(h, w))
        y_g_up_seq = global_dict["y_g_up_seq"]

        lambda_gate = torch.sigmoid(
            self.global_inject_proj(torch.cat([y, y_g_up_seq, u], dim=-1))
        )
        z = y + lambda_gate * y_g_up_seq

        v_map = self.local_dwconv(x)
        v_seq, _, _ = self._flatten_hw(v_map)
        v = self.local_proj(v_seq)

        eta = torch.sigmoid(self.local_fusion_proj(torch.cat([v, z], dim=-1)))
        u_out = eta * v + (1.0 - eta) * z

        u_out_map = self._reshape_to_bchw(u_out, h, w)
        out = residual + self.output_proj(u_out_map)

        if not return_intermediates:
            return out

        aux = {
            "u": u,
            "y_forward": long_range_dict["y_forward"],
            "y_backward": long_range_dict["y_backward"],
            "rho": long_range_dict["rho"],
            "y": y,
            "lambda_gate": lambda_gate,
            "z": z,
            "v": v,
            "eta": eta,
            "u_out": u_out,
            "global_x": global_dict["x_g"],
            "global_u": global_dict["u_g"],
            "global_y_seq": global_dict["y_g_seq"],
            "global_y_up": global_dict["y_g_up"],
            "forward_gates": {
                "context": long_range_dict["forward"]["context"],
                "d": long_range_dict["forward"]["d"],
                "g": long_range_dict["forward"]["g"],
                "o": long_range_dict["forward"]["o"],
                "a": long_range_dict["forward"]["a"],
                "b": long_range_dict["forward"]["b"],
                "states": long_range_dict["forward"]["states"],
            },
            "backward_gates_reversed_order": {
                "context": long_range_dict["backward_reversed"]["context"],
                "d": long_range_dict["backward_reversed"]["d"],
                "g": long_range_dict["backward_reversed"]["g"],
                "o": long_range_dict["backward_reversed"]["o"],
                "a": long_range_dict["backward_reversed"]["a"],
                "b": long_range_dict["backward_reversed"]["b"],
                "states": long_range_dict["backward_reversed"]["states"],
            },
            "global_gates": {
                "context": global_dict["scan"]["context"],
                "d": global_dict["scan"]["d"],
                "g": global_dict["scan"]["g"],
                "o": global_dict["scan"]["o"],
                "a": global_dict["scan"]["a"],
                "b": global_dict["scan"]["b"],
                "states": global_dict["scan"]["states"],
            },
        }
        return out, aux