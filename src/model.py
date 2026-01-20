import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        with torch.no_grad():
            self.embed.weight[:, :num_features].fill_(1.0)
            self.embed.weight[:, num_features:].zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype != torch.long:
            y = y.long()
        h = self.norm(x)
        gamma_beta = self.embed(y)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return h * gamma + beta


class Conv3x3(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, dilation: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, num_classes: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = ConditionalInstanceNorm2d(channels, num_classes)
        self.conv1 = Conv3x3(channels, channels, dilation=dilation)
        self.norm2 = ConditionalInstanceNorm2d(channels, num_classes)
        self.conv2 = Conv3x3(channels, channels, dilation=dilation)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x, y)))
        h = self.conv2(self.act(self.norm2(h, y)))
        return x + h


class RCUBlock(nn.Module):
    def __init__(
        self, channels: int, num_classes: int, num_units: int = 2, dilation: int = 1
    ) -> None:
        super().__init__()
        self.units = nn.ModuleList(
            [
                ResidualConvUnit(channels, num_classes, dilation=dilation)
                for _ in range(num_units)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = x
        for unit in self.units:
            h = unit(h, y)
        return h


class CondAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_classes: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if in_ch == out_ch:
            self.norm = None
            self.conv = None
        else:
            self.norm = ConditionalInstanceNorm2d(in_ch, num_classes)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.in_ch == self.out_ch:
            return x
        return self.conv(self.act(self.norm(x, y)))


class MultiResolutionFusion(nn.Module):
    def __init__(
        self, in_channels: Sequence[int], out_ch: int, num_classes: int
    ) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.norms = nn.ModuleList(
            [ConditionalInstanceNorm2d(c, num_classes) for c in in_channels]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(c, out_ch, kernel_size=3, stride=1, padding=1)
                for c in in_channels
            ]
        )
        self.act = nn.ELU()

    def forward(self, xs: Sequence[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        if len(xs) != len(self.convs):
            raise ValueError(f"Expected {len(self.convs)} inputs, got {len(xs)}")

        target_h = max(x.shape[-2] for x in xs)
        target_w = max(x.shape[-1] for x in xs)
        fused = None
        for x, norm, conv in zip(xs, self.norms, self.convs):
            h = conv(self.act(norm(x, y)))
            if h.shape[-2:] != (target_h, target_w):
                h = F.interpolate(h, size=(target_h, target_w), mode="nearest")
            fused = h if fused is None else fused + h
        return fused


class ChainedResidualPooling(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_stages: int = 4) -> None:
        super().__init__()
        self.norms = nn.ModuleList(
            [
                ConditionalInstanceNorm2d(channels, num_classes)
                for _ in range(num_stages)
            ]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                for _ in range(num_stages)
            ]
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = x
        out = x
        for norm, conv in zip(self.norms, self.convs):
            h = self.act(norm(h, y))
            h = F.max_pool2d(h, kernel_size=5, stride=1, padding=2)
            h = conv(h)
            out = out + h
        return out


class RefineBlock(nn.Module):
    def __init__(
        self, in_channels: Sequence[int], out_ch: int, num_classes: int
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [CondAdapter(c, out_ch, num_classes) for c in in_channels]
        )
        self.rcu_in = nn.ModuleList(
            [RCUBlock(out_ch, num_classes, num_units=2) for _ in in_channels]
        )
        self.msf = MultiResolutionFusion(
            [out_ch] * len(in_channels), out_ch, num_classes
        )
        self.crp = ChainedResidualPooling(out_ch, num_classes, num_stages=4)
        self.rcu_out = RCUBlock(out_ch, num_classes, num_units=2)

    def forward(self, xs: Sequence[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        if len(xs) != len(self.adapters):
            raise ValueError(f"Expected {len(self.adapters)} inputs, got {len(xs)}")

        hs: List[torch.Tensor] = []
        for x, adapter, rcu in zip(xs, self.adapters, self.rcu_in):
            h = adapter(x, y)
            h = rcu(h, y)
            hs.append(h)

        h = hs[0] if len(hs) == 1 else self.msf(hs, y)
        h = self.crp(h, y)
        h = self.rcu_out(h, y)
        return h


class NCSN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        nf: int = 128,
        num_classes: int = 10,
        dilations: Sequence[int] = (1, 2, 4, 8),
        scale_by_sigma: bool = True,
    ) -> None:
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("Expected 4 dilation values (for 4 RefineNet stages).")

        self.in_channels = in_channels
        self.nf = nf
        self.num_classes = num_classes
        self.scale_by_sigma = bool(scale_by_sigma)

        self.register_buffer("sigmas", torch.empty(num_classes), persistent=False)

        self.begin_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)

        self.stage1 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[0])
        self.stage2 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[1])
        self.stage3 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[2])
        self.stage4 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[3])

        self.refine4 = RefineBlock([nf], nf, num_classes)
        self.refine3 = RefineBlock([nf, nf], nf, num_classes)
        self.refine2 = RefineBlock([nf, nf], nf, num_classes)
        self.refine1 = RefineBlock([nf, nf], nf, num_classes)

        self.end_norm = ConditionalInstanceNorm2d(nf, num_classes)
        self.end_act = nn.ELU()
        self.end_conv = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() != 1:
            labels = labels.view(-1)

        h = self.begin_conv(x)
        h1 = self.stage1(h, labels)
        h2 = self.stage2(h1, labels)
        h3 = self.stage3(h2, labels)
        h4 = self.stage4(h3, labels)

        r4 = self.refine4([h4], labels)
        r3 = self.refine3([h3, r4], labels)
        r2 = self.refine2([h2, r3], labels)
        r1 = self.refine1([h1, r2], labels)

        out = self.end_conv(self.end_act(self.end_norm(r1, labels)))
        if self.scale_by_sigma:
            if self.sigmas.numel() != self.num_classes:
                raise RuntimeError(
                    f"`sigmas` buffer has shape {tuple(self.sigmas.shape)}; "
                    f"expected ({self.num_classes},). Call `set_sigmas(...)`."
                )
            used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)
            out = out / used_sigmas
        return out

    @torch.no_grad()
    def set_sigmas(self, sigmas: torch.Tensor) -> None:
        if sigmas.dim() != 1:
            raise ValueError("sigmas must be 1-D.")
        if sigmas.numel() != self.num_classes:
            raise ValueError(
                f"sigmas length ({sigmas.numel()}) must match num_classes ({self.num_classes})."
            )
        self.sigmas.copy_(sigmas.to(self.sigmas.device, dtype=self.sigmas.dtype))


def make_sigmas(
    sigma_begin: float,
    sigma_end: float,
    num_scales: int,
) -> torch.Tensor:
    if sigma_begin <= 0 or sigma_end <= 0:
        raise ValueError("sigmas must be positive.")
    if sigma_begin <= sigma_end:
        raise ValueError(
            "Expected sigma_begin > sigma_end (descending noise schedule)."
        )
    if num_scales < 2:
        raise ValueError("num_scales must be >= 2.")

    return torch.exp(
        torch.linspace(math.log(sigma_begin), math.log(sigma_end), num_scales)
    )
