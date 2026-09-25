"""Minimal UnMarker/quality attack backend used by the public evaluator.

The paper's released configuration does not optimize decoder bit loss or a
reference-residual frequency loss.  This module therefore contains only the
low-frequency UnMarker objective, the image-quality objective, and the
selection diagnostics needed by ``eval_tps_decoder_curve.py``.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def parse_kernel_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not sizes:
        raise ValueError("UnMarker kernel sizes cannot be empty")
    if any(size < 3 or size % 2 == 0 for size in sizes):
        raise ValueError("UnMarker kernel sizes must be odd integers >= 3")
    return sizes


class UnMarkerFilterBank(torch.nn.Module):
    """Learned edge-aware smoothing filters used by the low-frequency attack."""

    def __init__(self, height: int, width: int, kernel_sizes=(3, 5, 7), color_sigma=0.15):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.color_sigma = float(color_sigma)
        self.logits = torch.nn.ParameterList()
        for kernel_size in self.kernel_sizes:
            if kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError("UnMarker filter kernel sizes must be odd and >= 3")
            values = torch.full((1, kernel_size * kernel_size, height, width), -4.0)
            values[:, (kernel_size * kernel_size) // 2].fill_(4.0)
            self.logits.append(torch.nn.Parameter(values))

    def forward(self, image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        output = image
        for kernel_size, logits in zip(self.kernel_sizes, self.logits):
            pad = kernel_size // 2
            batch, channels, height, width = output.shape
            patches = F.unfold(output, kernel_size, padding=pad).view(
                batch, channels, kernel_size * kernel_size, height * width
            )
            ref_patches = F.unfold(reference, kernel_size, padding=pad).view(
                batch, channels, kernel_size * kernel_size, height * width
            )
            center = (kernel_size * kernel_size) // 2
            ref_center = ref_patches[:, :, center : center + 1]
            color_gate = torch.exp(
                -(ref_patches - ref_center).abs().mean(dim=1)
                / max(self.color_sigma, 1e-6)
            )
            weights = torch.softmax(logits.view(1, -1, height * width), dim=1)
            weights = weights * color_gate
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            filtered = (patches * weights.unsqueeze(1)).sum(dim=2)
            output = filtered.view(batch, channels, height, width)
        return output


def materialize_adversarial_from_delta(
    watermarked: torch.Tensor,
    raw_delta: torch.Tensor,
    unmarker_filter_bank: UnMarkerFilterBank | None = None,
    max_perturbation: float | None = None,
) -> torch.Tensor:
    adversarial = (watermarked + raw_delta).clamp(-1.0, 1.0)
    if unmarker_filter_bank is not None:
        adversarial = unmarker_filter_bank(adversarial, watermarked).clamp(-1.0, 1.0)
    if max_perturbation is not None and max_perturbation > 0.0:
        adversarial = torch.max(
            torch.min(adversarial, watermarked + max_perturbation),
            watermarked - max_perturbation,
        ).clamp(-1.0, 1.0)
    return adversarial


def unmarker_low_losses(
    image: torch.Tensor, reference: torch.Tensor, mpl_size: int = 5, frl_size: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return multi-scale perturbation and local-deviation terms."""
    mpl_size = max(int(mpl_size), 2)
    frl_size = max(int(frl_size), 3)
    if frl_size % 2 == 0:
        frl_size += 1
    image_mean = F.avg_pool2d(image, kernel_size=mpl_size, stride=mpl_size)
    reference_mean = F.avg_pool2d(reference, kernel_size=mpl_size, stride=mpl_size)
    mpl = (image_mean - reference_mean).abs().mean()
    patches = F.unfold(image, frl_size, padding=frl_size // 2)
    channels = image.shape[1]
    patches = patches.view(
        image.shape[0], channels, frl_size * frl_size, image.shape[-2] * image.shape[-1]
    )
    local_median = patches.median(dim=2).values
    local_deviation = (patches - local_median.unsqueeze(2)).abs().mean()
    return mpl, local_deviation


def unmarker_visual_proxy(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    loss = F.l1_loss(image, reference)
    for size in (2, 4, 8):
        if min(image.shape[-2:]) >= size:
            loss = loss + F.l1_loss(
                F.avg_pool2d(image, size, size),
                F.avg_pool2d(reference, size, size),
            )
    return loss / 4.0


def _global_grad_norm(grads, eps: float = 1e-12) -> torch.Tensor:
    squared = None
    for grad in grads:
        if grad is None:
            continue
        value = grad.detach().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return torch.tensor(eps)
    return squared.sqrt().clamp_min(eps)


def apply_gradnorm_update(
    adversarial: torch.Tensor,
    watermarked: torch.Tensor,
    optimizer_params: list[torch.Tensor],
    args: Any,
    step: int = 0,
) -> dict[str, float]:
    """Mix only UnMarker and image-quality gradients.

    Decoder outputs are intentionally absent from this function: the public
    attack is not optimized with decoder bit loss.
    """
    mpl, local_deviation = unmarker_low_losses(
        adversarial,
        watermarked,
        getattr(args, "unmarker_low_mpl_size", 5),
        getattr(args, "unmarker_low_frl_size", 5),
    )
    unmarker_loss = -mpl + 0.25 * local_deviation
    quality_loss = (
        args.watermark_weight * F.mse_loss(adversarial, watermarked)
        + getattr(args, "unmarker_low_weight", 0.5)
        * getattr(args, "unmarker_visual_weight", 0.25)
        * unmarker_visual_proxy(adversarial, watermarked)
    )
    unmarker_grads = torch.autograd.grad(
        unmarker_loss, optimizer_params, retain_graph=True, allow_unused=True
    )
    quality_grads = torch.autograd.grad(quality_loss, optimizer_params, allow_unused=True)
    unmarker_norm = _global_grad_norm(unmarker_grads).to(adversarial.device)
    quality_norm = _global_grad_norm(quality_grads).to(adversarial.device)
    unmarker_ratio = float(getattr(args, "gradnorm_unmarker_ratio", 1.0))
    for parameter, unmarker_grad, quality_grad in zip(
        optimizer_params, unmarker_grads, quality_grads
    ):
        combined = torch.zeros_like(parameter)
        if unmarker_grad is not None:
            combined = combined + unmarker_ratio * unmarker_grad / unmarker_norm
        if quality_grad is not None:
            combined = combined + float(args.gradnorm_quality_scale) * quality_grad
        parameter.grad = combined
    return {
        "unmarker_loss": float(unmarker_loss.detach().item()),
        "unmarker_mpl": float(mpl.detach().item()),
        "unmarker_frl": float(local_deviation.detach().item()),
        "quality_loss": float(quality_loss.detach().item()),
        "unmarker_grad_norm": float(unmarker_norm.detach().item()),
        "quality_grad_norm": float(quality_norm.detach().item()),
        "unmarker_ratio": unmarker_ratio,
        "step": float(step),
    }


def _psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    mse = max(float(mse), 1e-12)
    return 10.0 * math.log10((data_range * data_range) / mse)


def evaluate_attack_state(
    step: int,
    start_time: float,
    adversarial: torch.Tensor,
    watermarked: torch.Tensor,
    decoders: list[Any],
    decoder_weights: dict[str, float],
    args: Any,
    lpips_quality_evaluator=None,
    unmarker_filter_bank=None,
):
    """Evaluate quality and BER diagnostics without constructing attack loss."""
    watermark_loss = F.mse_loss(adversarial, watermarked)
    mpl, local_deviation = unmarker_low_losses(
        adversarial,
        watermarked,
        getattr(args, "unmarker_low_mpl_size", 5),
        getattr(args, "unmarker_low_frl_size", 5),
    )
    unmarker_loss = -mpl + 0.25 * local_deviation
    loss = args.watermark_weight * watermark_loss + unmarker_loss
    decoder_metrics = {}
    with torch.no_grad():
        for decoder in decoders:
            decoded = decoder.decode(adversarial)
            ber = decoder.bit_error_rate(decoded)
            decoder_metrics[decoder.name] = {
                "loss_weight": float(decoder_weights.get(decoder.name, 0.0)),
                "bit_error_rate_vs_clean": ber,
                "bit_error_distance_to_random": abs(ber - 0.5),
                "decoded_shape": list(decoded.shape),
            }
    weighted_gap = float(
        sum(
            item["loss_weight"] * item["bit_error_distance_to_random"]
            for item in decoder_metrics.values()
        )
    )
    perturb_mse = float(watermark_loss.detach().item())
    psnr_to_watermarked = _psnr_from_mse(perturb_mse)
    lpips_to_watermarked = (
        None if lpips_quality_evaluator is None else lpips_quality_evaluator(adversarial)
    )
    record = {
        "step": int(step),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "loss": float(loss.detach().item()),
        "watermark_loss": perturb_mse,
        "unmarker_loss": float(unmarker_loss.detach().item()),
        "unmarker_mpl": float(mpl.detach().item()),
        "unmarker_frl": float(local_deviation.detach().item()),
        "psnr_to_watermarked": psnr_to_watermarked,
        "lpips_to_watermarked": lpips_to_watermarked,
        "mean_bit_error_distance_to_random": float(
            np.mean([item["bit_error_distance_to_random"] for item in decoder_metrics.values()])
        ) if decoder_metrics else 0.0,
        "weighted_bit_error_distance_to_random": weighted_gap,
        "decoder_weights": decoder_weights,
        "decoder_metrics": decoder_metrics,
        "residual_linf": float((adversarial - watermarked).detach().abs().max().item()),
        "score": -weighted_gap + 0.001 * psnr_to_watermarked,
    }
    return loss, record, adversarial.detach().clone(), (adversarial - watermarked).detach().clone()


def quality_constraints_satisfied(record: dict[str, Any], args: Any) -> bool:
    if record["psnr_to_watermarked"] < args.selection_min_psnr:
        return False
    if args.quality_constraint and args.target_lpips is not None:
        value = record.get("lpips_to_watermarked")
        if value is None or value > args.target_lpips:
            return False
    return True


def tensor_image_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    return ((image + 1.0) * 127.5).round().byte().squeeze(0).permute(1, 2, 0).numpy()
