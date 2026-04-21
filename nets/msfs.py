import math
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {tuple(x.shape)}.")
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        norm: bool = False,
        act: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class msfs(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int,
        align_mode: str = "bilinear",
        downsample_with_pool: bool = True,
        use_bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not isinstance(in_channels, (list, tuple)) or len(in_channels) == 0:
            raise ValueError("in_channels must be a non-empty list or tuple.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")

        self.in_channels = list(in_channels)
        self.num_scales = len(in_channels)
        self.embed_dim = embed_dim
        self.align_mode = align_mode
        self.downsample_with_pool = downsample_with_pool

        self.input_projs = nn.ModuleList(
            [
                ConvBNAct(
                    in_channels=ch,
                    out_channels=embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=use_bias,
                    norm=False,
                    act=False,
                )
                for ch in self.in_channels
            ]
        )

        self.anchor_norm = nn.LayerNorm(embed_dim, eps=eps)

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale_relation_bias = nn.Parameter(torch.zeros(self.num_scales, self.num_scales))

        self.semantic_transform = ConvBNAct(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=False,
            act=False,
        )

        self.spatial_norm = ChannelLayerNorm2d(embed_dim, eps=eps)

        self.gate_conv = ConvBNAct(
            in_channels=2 * embed_dim + 1,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=False,
            act=False,
        )

        self.memory_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.zeros_(self.scale_relation_bias)

    def _check_inputs(self, inputs: Sequence[torch.Tensor]) -> None:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("inputs must be a list or tuple of feature maps.")
        if len(inputs) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature maps, but got {len(inputs)}.")

        batch_size = inputs[0].shape[0]
        for idx, (x, in_ch) in enumerate(zip(inputs, self.in_channels)):
            if x.ndim != 4:
                raise ValueError(
                    f"Input at scale {idx} must be 4D [B, C, H, W], but got shape {tuple(x.shape)}."
                )
            if x.shape[0] != batch_size:
                raise ValueError("All input feature maps must have the same batch size.")
            if x.shape[1] != in_ch:
                raise ValueError(
                    f"Input at scale {idx} has channel {x.shape[1]}, expected {in_ch}."
                )

    def _resize_to(
        self,
        x: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        target_h, target_w = size
        _, _, h, w = x.shape

        if h == target_h and w == target_w:
            return x

        if self.downsample_with_pool and (h > target_h or w > target_w):
            return F.adaptive_avg_pool2d(x, output_size=(target_h, target_w))

        if self.align_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            return F.interpolate(
                x,
                size=(target_h, target_w),
                mode=self.align_mode,
                align_corners=False,
            )

        return F.interpolate(
            x,
            size=(target_h, target_w),
            mode=self.align_mode,
        )

    def _compute_semantic_anchors(
        self,
        projected_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        anchors: List[torch.Tensor] = []
        for feat in projected_features:
            gap = feat.mean(dim=(2, 3))
            s = self.anchor_norm(gap)
            anchors.append(s)
        anchors = torch.stack(anchors, dim=1)
        return anchors

    def _compute_scale_affinity(
        self,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(anchors)
        k = self.key_proj(anchors)

        affinity = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        affinity = affinity + self.scale_relation_bias.unsqueeze(0)
        alpha = torch.softmax(affinity, dim=-1)
        return alpha

    def _build_cross_scale_candidate(
        self,
        reference_index: int,
        projected_features: Sequence[torch.Tensor],
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ref_feat = projected_features[reference_index]
        _, _, ref_h, ref_w = ref_feat.shape

        aligned_and_transformed: List[torch.Tensor] = []
        for feat in projected_features:
            aligned = self._resize_to(feat, size=(ref_h, ref_w))
            aligned = self.semantic_transform(aligned)
            aligned_and_transformed.append(aligned)

        candidate = torch.zeros_like(ref_feat)
        for src_index, feat in enumerate(aligned_and_transformed):
            weight = alpha[:, reference_index, src_index].view(-1, 1, 1, 1)
            candidate = candidate + weight * feat

        return candidate, aligned_and_transformed

    def _difference_aware_fusion(
        self,
        reference_feat: torch.Tensor,
        candidate_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_norm = self.spatial_norm(reference_feat)
        cand_norm = self.spatial_norm(candidate_feat)

        delta = torch.mean(torch.abs(ref_norm - cand_norm), dim=1, keepdim=True)

        gate_input = torch.cat([reference_feat, candidate_feat, delta], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        fused = reference_feat + gate * candidate_feat
        return fused, delta, gate

    @staticmethod
    def _flatten_to_tokens(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], but got shape {tuple(x.shape)}.")
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        return_intermediates: bool = False,
    ) -> Union[
        Tuple[List[torch.Tensor], torch.Tensor],
        Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]],
    ]:
        self._check_inputs(inputs)

        projected_features = [proj(x) for proj, x in zip(self.input_projs, inputs)]
        anchors = self._compute_semantic_anchors(projected_features)
        alpha = self._compute_scale_affinity(anchors)

        fused_features: List[torch.Tensor] = []
        candidate_features: List[torch.Tensor] = []
        difference_maps: List[torch.Tensor] = []
        fusion_gates: List[torch.Tensor] = []
        aligned_feature_bank: List[List[torch.Tensor]] = []

        for ref_index in range(self.num_scales):
            reference_feat = projected_features[ref_index]

            candidate_feat, aligned_sources = self._build_cross_scale_candidate(
                reference_index=ref_index,
                projected_features=projected_features,
                alpha=alpha,
            )

            fused_feat, delta, gate = self._difference_aware_fusion(
                reference_feat=reference_feat,
                candidate_feat=candidate_feat,
            )

            fused_features.append(fused_feat)
            candidate_features.append(candidate_feat)
            difference_maps.append(delta)
            fusion_gates.append(gate)
            aligned_feature_bank.append(aligned_sources)

        memory_tokens = [self._flatten_to_tokens(feat) for feat in fused_features]
        memory = torch.cat(memory_tokens, dim=1)
        memory = self.memory_proj(memory)

        if not return_intermediates:
            return fused_features, memory

        aux: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "projected_features": projected_features,
            "anchors": anchors,
            "alpha": alpha,
            "candidate_features": candidate_features,
            "difference_maps": difference_maps,
            "fusion_gates": fusion_gates,
            "aligned_feature_bank": aligned_feature_bank,
            "memory_tokens_before_proj": memory_tokens,
        }
        return fused_features, memory, aux