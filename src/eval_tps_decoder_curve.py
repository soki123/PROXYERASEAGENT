"""Evaluate candidate decoders under TPS, JPEG, VAE, and joint adversarial attacks."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODER_CLASSIFI_ROOT = Path(
    os.environ.get("ENCODER_CLASSIFI_ROOT", PROJECT_ROOT / "third_party" / "encoder_classifi")
)
if str(ENCODER_CLASSIFI_ROOT) not in sys.path:
    sys.path.insert(0, str(ENCODER_CLASSIFI_ROOT))

from attack_topk_decoders import (  # noqa: E402
    add_decoded_margin_diagnostics,
    build_argparser as build_decoder_argparser,
    build_decoder,
    canonical_method,
    load_image_tensor,
    set_seed,
    tensor_image_to_uint8,
)
from attack_topk_decoders_gradnorm import (  # noqa: E402
    UnMarkerFilterBank as GradNormUnMarkerFilterBank,
    apply_gradnorm_update,
    evaluate_attack_state as evaluate_gradnorm_attack_state,
    materialize_adversarial_from_delta as materialize_gradnorm_adversarial,
    parse_kernel_sizes as parse_gradnorm_kernel_sizes,
    quality_constraints_satisfied as gradnorm_quality_constraints_satisfied,
)


DEFAULT_IMAGE = os.environ.get("DEFAULT_WATERMARKED_IMAGE", "")
DEFAULT_OUTPUT_DIR = os.environ.get(
    "DEFAULT_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "single")
)
FIN_FAMILY = frozenset({"fin", "fin_heavy", "fin_jpeg"})
SEQUENTIAL_ATTACK_LIMITS = {
    "tps": 2,
    "jpeg": 3,
    "vae": 2,
    "adversarial": 1,
}
ENTROPY_SCRIPT = Path(
    os.environ.get(
        "ENTROPY_RANKING_SCRIPT", str(PROJECT_ROOT / "src" / "decoder_entropy_ranking.py")
    )
)
METRIC_COLUMNS = [
    "attack",
    "attack_value",
    "scale",
    "quality",
    "vae_quality",
    "setting",
    "trial",
    "seed",
    "image_path",
    "decoder",
    "psnr",
    "lpips",
    "bit_error_rate",
    "bit_acc",
    "bit_error_distance_to_random",
    "decoded_delta_from_clean_mean",
    "decoded_delta_from_clean_max",
    "decoded_margin_to_0_5_mean",
    "decoded_margin_to_0_5_min",
    "decoded_margin_to_0_5_max",
    "clean_decoded_margin_to_0_5_mean",
    "clean_decoded_margin_to_0_5_min",
    "clean_decoded_margin_to_0_5_max",
    "decoded_shape",
]
SUMMARY_METRICS = [
    "psnr",
    "lpips",
    "bit_error_rate",
    "bit_acc",
    "decoded_delta_from_clean_mean",
    "decoded_delta_from_clean_max",
    "decoded_margin_to_0_5_mean",
]
# Fallback PyTorch/CompressAI cache used when TORCH_HOME is not provided by the
# caller. The launch script exports the same local directory explicitly.
TORCH_HOME_IN_SCRIPT = os.environ.get(
    "TORCH_HOME", str(PROJECT_ROOT / "cache" / "torch")
)
OFFICIAL_VAE_INPUT_SIZE = 512
VAE_PROTOCOL_VERSION = "watermarkattacker_512_v1"
_LPIPS_MODELS: dict[str, Any] = {}
_DECODER_MODELS: dict[str, Any] = {}
_VAE_MODELS: dict[tuple[str, int, str], Any] = {}
ATTACK_DECISION_CONTEXT = {
    "tps": {
        "parameter": "scale",
        "description": "generic stochastic thin-plate-spline geometric distortion",
        "surrogate_optimized": False,
    },
    "jpeg": {
        "parameter": "quality",
        "description": "generic JPEG compression distortion",
        "surrogate_optimized": False,
    },
    "vae": {
        "parameter": "quality",
        "description": "generic learned-compression reconstruction distortion",
        "surrogate_optimized": False,
    },
    "adversarial": {
        "parameter": "fixed default configuration (no parameter sweep)",
        "description": "joint gradient-based pixel perturbation optimized against the weighted candidate decoders",
        "surrogate_optimized": True,
    },
}


def parse_scales(value: str) -> list[float]:
    scales = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        scale = float(item)
        if scale < 0.0:
            raise ValueError("TPS scales must be non-negative.")
        scales.append(scale)
    if not scales:
        raise ValueError("At least one TPS scale is required.")
    return scales


def parse_int_list(
    value: str,
    minimum: int | None = None,
    maximum: int | None = None,
    label: str = "value",
) -> list[int]:
    values = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parsed = int(item)
        if minimum is not None and parsed < minimum or maximum is not None and parsed > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        values.append(parsed)
    if not values:
        raise ValueError(f"At least one {label} is required.")
    return values


def normalize_decoder_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("At least one decoder weight is required.")
    normalized_input = {str(name): float(weight) for name, weight in weights.items()}
    if any(not math.isfinite(weight) or weight < 0.0 for weight in normalized_input.values()):
        raise ValueError("Decoder weights must be finite and non-negative.")
    total = sum(normalized_input.values())
    if total <= 0.0:
        raise ValueError("At least one decoder weight must be positive.")
    return {name: weight / total for name, weight in normalized_input.items()}


def parse_decoder_weights(value: str) -> dict[str, float]:
    """Parse comma-separated `decoder=weight` pairs."""
    weights: dict[str, float] = {}
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("--decoder-weights must use decoder=weight pairs separated by commas.")
        raw_name, raw_weight = item.split("=", 1)
        name = canonical_method(raw_name.strip())
        if not name:
            raise ValueError("Decoder name in --decoder-weights cannot be empty.")
        if name in weights:
            raise ValueError(f"Duplicate decoder weight: {name}")
        weights[name] = float(raw_weight.strip())
    return normalize_decoder_weights(weights)


def psnr_between_neg1_tensors(image: torch.Tensor, reference: torch.Tensor) -> float:
    if image.shape[-2:] != reference.shape[-2:]:
        reference = F.interpolate(reference, size=image.shape[-2:], mode="bilinear", align_corners=False)
    mse = F.mse_loss(image, reference)
    mse_value = max(float(mse.detach().item()), 1e-12)
    return 10.0 * math.log10(4.0 / mse_value)


def lpips_between_neg1_tensors(
    image: torch.Tensor,
    reference: torch.Tensor,
    net: str = "alex",
) -> float:
    """Return cached LPIPS for tensors already normalized to [-1, 1]."""
    if image.shape[-2:] != reference.shape[-2:]:
        reference = F.interpolate(reference, size=image.shape[-2:], mode="bilinear", align_corners=False)
    model_key = f"{image.device}:{net}"
    model = _LPIPS_MODELS.get(model_key)
    if model is None:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS decision quality requires the `lpips` package. Install it or use "
                "--decision-quality-metric psnr."
            ) from exc
        model = lpips.LPIPS(net=net, verbose=False).to(image.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _LPIPS_MODELS[model_key] = model
    with torch.no_grad():
        value = model(image, reference, normalize=False)
    return float(value.detach().mean().item())


def build_tps_attack(scale: float, p: float):
    import kornia.augmentation as K

    # Kornia 0.7 does not expose a padding_mode constructor argument. Its TPS
    # implementation calls grid_sample without overriding padding_mode, whose
    # PyTorch default is already "zeros".
    return K.RandomThinPlateSpline(
        scale=scale,
        align_corners=False,
        same_on_batch=False,
        p=p,
        keepdim=False,
    )


def apply_kjpeg_attack(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply deterministic Pillow JPEG compression to an NCHW RGB tensor.

    Kornia 0.7 does not provide ``RandomJPEG``. Pillow is also the JPEG codec
    used by the official WatermarkAttacker baseline, so using it explicitly
    avoids environment-dependent behavior while retaining the same quality
    factor convention.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            "JPEG attack expects an NCHW RGB tensor, got "
            f"shape={tuple(image.shape)}."
        )
    quality = int(quality)
    if quality < 1 or quality > 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}.")
    device = image.device
    dtype = image.dtype
    attacked_samples: list[torch.Tensor] = []
    for sample in image.detach():
        sample_uint8 = (
            ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        buffer = io.BytesIO()
        Image.fromarray(sample_uint8).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            decoded = np.asarray(compressed.convert("RGB"), dtype=np.uint8).copy()
        decoded_tensor = (
            torch.from_numpy(decoded)
            .permute(2, 0, 1)
            .to(device=device, dtype=dtype)
            / 127.5
            - 1.0
        )
        attacked_samples.append(decoded_tensor)
    return torch.stack(attacked_samples, dim=0).clamp(-1.0, 1.0)


def configure_torch_cache() -> None:
    # Respect a path selected by the caller or launch script.
    os.environ.setdefault("TORCH_HOME", TORCH_HOME_IN_SCRIPT)


def build_vae_model(model_name: str, quality: int, device: torch.device):
    configure_torch_cache()
    try:
        from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, cheng2020_anchor, mbt2018, mbt2018_mean
    except ImportError as exc:
        raise RuntimeError(
            "The VAE attack requires compressai. Install the WatermarkAttacker requirements "
            "or run this script in an environment where `import compressai` works."
        ) from exc

    builders = {
        "bmshj2018-factorized": bmshj2018_factorized,
        "bmshj2018-hyperprior": bmshj2018_hyperprior,
        "mbt2018-mean": mbt2018_mean,
        "mbt2018": mbt2018,
        "cheng2020-anchor": cheng2020_anchor,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported VAE model: {model_name}")
    return builders[model_name](quality=quality, pretrained=True).eval().to(device)


def get_cached_vae_model(model_name: str, quality: int, device: torch.device):
    """Return a process-wide VAE instance so batch jobs do not reload weights."""
    key = (model_name, int(quality), str(device))
    if key not in _VAE_MODELS:
        print(f"Loading VAE model {model_name} quality={quality} (batch cache miss)", flush=True)
        _VAE_MODELS[key] = build_vae_model(model_name, int(quality), device)
    return _VAE_MODELS[key]


def get_cached_decoder(method: str, args: argparse.Namespace):
    """Return a decoder shared by sequential images in the current process."""
    key = canonical_method(method)
    if key not in _DECODER_MODELS:
        print(f"Loading decoder {key} (batch cache miss)", flush=True)
        _DECODER_MODELS[key] = build_decoder(key, args)
    return _DECODER_MODELS[key]


def apply_vae_attack(
    image: torch.Tensor,
    model,
    input_size: int = OFFICIAL_VAE_INPUT_SIZE,
) -> torch.Tensor:
    """Apply the official WatermarkAttacker CompressAI spatial protocol.

    WatermarkAttacker resizes every RGB input to 512 x 512 before CompressAI.
    This evaluator keeps attack states at the original resolution, so the
    reconstruction is resized back after the model.  The latter is the same
    bicubic adaptation used by the baseline evaluator before decoding and
    LPIPS measurement.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            "VAE attack expects an NCHW RGB tensor, got "
            f"shape={tuple(image.shape)}."
        )
    input_size = int(input_size)
    if input_size != OFFICIAL_VAE_INPUT_SIZE:
        raise ValueError(
            "The official WatermarkAttacker VAE protocol requires "
            f"input_size={OFFICIAL_VAE_INPUT_SIZE}, got {input_size}."
        )

    original_size = image.shape[-2:]
    # Reproduce Image.open(...).convert("RGB").resize((512, 512)) exactly.
    # Sequential states are quantized as PNG would be before the Pillow resize.
    resized_inputs = []
    for sample in image.detach():
        sample_uint8 = (
            ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        resized = Image.fromarray(sample_uint8).resize((input_size, input_size))
        resized_array = np.asarray(resized, dtype=np.uint8).copy()
        resized_tensor = (
            torch.from_numpy(resized_array)
            .permute(2, 0, 1)
            .to(device=image.device, dtype=image.dtype)
            / 255.0
        )
        resized_inputs.append(resized_tensor)
    image_01 = torch.stack(resized_inputs, dim=0)

    out = model(image_01)
    reconstructed = out["x_hat"].clamp(0.0, 1.0)
    # ``rec.save(out_path)`` in WatermarkAttacker quantizes the reconstruction
    # before the evaluator reloads it.
    reconstructed = torch.floor(reconstructed * 255.0) / 255.0
    if reconstructed.shape[-2:] != original_size:
        reconstructed = F.interpolate(
            reconstructed,
            size=original_size,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
    return (reconstructed * 2.0 - 1.0).clamp(-1.0, 1.0)


def apply_adversarial_attack(
    image: torch.Tensor,
    decoders: list[Any],
    decoder_weights: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the sole public adversarial objective (low-frequency UnMarker + quality)."""
    return apply_gradnorm_attack_in_process(
        image, decoders, decoder_weights, args, output_dir
    )


class CachedLpipsQualityEvaluator:
    """LPIPS quality gate backed by the evaluator's process-wide model cache."""

    def __init__(self, reference: torch.Tensor, net: str = "alex"):
        self.reference = reference.detach()
        self.net = net

    def __call__(self, image: torch.Tensor) -> float:
        return lpips_between_neg1_tensors(image, self.reference, net=self.net)


def build_in_process_gradnorm_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    """Map evaluator options to the standalone GradNorm implementation contract."""
    backend_args = copy.copy(args)
    backend_args.output_dir = str(output_dir)
    backend_args.max_perturbation = float(args.adversarial_max_perturbation)
    backend_args.watermark_weight = float(args.adversarial_watermark_weight)
    backend_args.unmarker_mode = "low"
    backend_args.unmarker_implementation = "custom"
    backend_args.gradnorm_unmarker_ratio = 1.0
    backend_args.gradnorm_quality_scale = float(args.adversarial_gradnorm_quality_scale)
    backend_args.selection_min_step = int(args.adversarial_selection_min_step)
    backend_args.selection_min_psnr = float(args.adversarial_selection_min_psnr)
    backend_args.target_lpips = args.adversarial_target_lpips
    backend_args.lpips_net = str(args.adversarial_lpips_net)
    backend_args.quality_constraint = bool(args.adversarial_quality_constraint)

    # Fixed low-frequency UnMarker defaults used by the public profile.
    backend_args.unmarker_low_weight = 0.5
    backend_args.unmarker_low_kernel_sizes = "3,5,7"
    backend_args.unmarker_low_mpl_size = 5
    backend_args.unmarker_low_frl_size = 5
    backend_args.unmarker_visual_weight = 0.25
    return backend_args


def append_json_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_gradnorm_attack_in_process(
    image: torch.Tensor,
    decoders: list[Any],
    decoder_weights: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run GradNorm in the current worker, reusing already-loaded decoders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = output_dir / "gradnorm_backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = output_dir / "sequential_source.png"
    source_uint8 = tensor_image_to_uint8(image)
    Image.fromarray(source_uint8).save(source_image_path)
    source = (
        torch.from_numpy(np.asarray(source_uint8).copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=args.device, dtype=image.dtype)
        / 127.5
        - 1.0
    )
    backend_args = build_in_process_gradnorm_args(args, backend_dir)
    normalized_weights = normalize_decoder_weights(decoder_weights)
    lpips_evaluator = (
        CachedLpipsQualityEvaluator(source, net=backend_args.lpips_net)
        if backend_args.target_lpips is not None
        else None
    )
    unmarker_filter_bank = None
    if backend_args.unmarker_mode == "low":
        unmarker_filter_bank = GradNormUnMarkerFilterBank(
            source.shape[-2],
            source.shape[-1],
            parse_gradnorm_kernel_sizes(backend_args.unmarker_low_kernel_sizes),
        ).to(backend_args.device)

    if backend_args.unmarker_mode == "high":
        raw_delta = (1e-4 * torch.randn_like(source)).requires_grad_(True)
    else:
        raw_delta = torch.zeros_like(source, requires_grad=True)
    optimizer_params = [raw_delta]
    if unmarker_filter_bank is not None:
        optimizer_params.extend(unmarker_filter_bank.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=backend_args.lr)

    optimization_log_path = backend_dir / "optimization_log.jsonl"
    optimization_log_path.write_text("", encoding="utf-8")
    run_log_path = backend_dir / "run.log"
    run_log_path.write_text(
        "GradNorm backend executed in the evaluator process.\n",
        encoding="utf-8",
    )

    best_score = None
    best_any_score = None
    best_payload = None
    best_any_payload = None
    start_time = time.time()
    convergence_patience = int(getattr(args, "adversarial_convergence_patience", 0) or 0)
    convergence_min_delta = float(
        getattr(args, "adversarial_convergence_min_delta", 0.0) or 0.0
    )
    convergence_warmup = int(getattr(args, "adversarial_convergence_warmup", 0) or 0)
    convergence_best_loss = math.inf
    convergence_stale_steps = 0
    converged = False
    convergence_reason = "maximum_steps_reached"
    executed_steps = 0
    try:
        for step in range(1, backend_args.steps + 1):
            executed_steps = step
            adversarial = materialize_gradnorm_adversarial(
                source,
                raw_delta,
                unmarker_filter_bank,
                backend_args.max_perturbation,
            )
            optimizer.zero_grad(set_to_none=True)
            gradnorm_diagnostics = apply_gradnorm_update(
                adversarial,
                source,
                optimizer_params,
                backend_args,
                step,
            )
            optimizer.step()
            if backend_args.max_perturbation > 0.0:
                with torch.no_grad():
                    raw_delta.clamp_(
                        -backend_args.max_perturbation,
                        backend_args.max_perturbation,
                    )

            adversarial = materialize_gradnorm_adversarial(
                source,
                raw_delta,
                unmarker_filter_bank,
                backend_args.max_perturbation,
            )
            _, record, current_adversarial, current_residual = evaluate_gradnorm_attack_state(
                step,
                start_time,
                adversarial,
                source,
                decoders,
                normalized_weights,
                backend_args,
                lpips_evaluator,
                unmarker_filter_bank,
            )
            record.update(gradnorm_diagnostics)
            if step == 1 or step % backend_args.log_interval == 0 or step == backend_args.steps:
                append_json_record(optimization_log_path, record)
                progress = (
                    f"[in-process GradNorm {step}/{backend_args.steps}] "
                    f"loss={record['loss']:.6f} "
                    f"psnr={record['psnr_to_watermarked']:.2f} "
                    f"weighted_ber_gap={record['weighted_bit_error_distance_to_random']:.4f}"
                )
                with run_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(progress + "\n")
                print(progress, flush=True)

            payload = (record, current_adversarial, current_residual)
            selectable = step > backend_args.selection_min_step
            if selectable and (best_any_score is None or record["score"] > best_any_score):
                best_any_score = float(record["score"])
                best_any_payload = payload
            if (
                selectable
                and gradnorm_quality_constraints_satisfied(record, backend_args)
                and (best_score is None or record["score"] > best_score)
            ):
                best_score = float(record["score"])
                best_payload = payload

            current_loss = float(record["loss"])
            if convergence_best_loss - current_loss >= convergence_min_delta:
                convergence_best_loss = current_loss
                convergence_stale_steps = 0
            else:
                convergence_stale_steps += 1
            if (
                convergence_patience > 0
                and step >= convergence_warmup
                and convergence_stale_steps >= convergence_patience
            ):
                converged = True
                convergence_reason = "loss_plateau"
                record["converged"] = True
                record["convergence_reason"] = convergence_reason
                record["convergence_stale_steps"] = convergence_stale_steps
                if not (step == 1 or step % backend_args.log_interval == 0):
                    append_json_record(optimization_log_path, record)
                break
        if best_payload is None:
            best_payload = best_any_payload
            if best_payload is not None:
                best_payload[0]["selection_quality_fallback"] = True
        if best_payload is None:
            adversarial = materialize_gradnorm_adversarial(
                source,
                raw_delta,
                unmarker_filter_bank,
                backend_args.max_perturbation,
            )
            _, record, current_adversarial, current_residual = evaluate_gradnorm_attack_state(
                executed_steps,
                start_time,
                adversarial,
                source,
                decoders,
                normalized_weights,
                backend_args,
                lpips_evaluator,
                unmarker_filter_bank,
            )
            best_payload = (record, current_adversarial, current_residual)
    finally:
        pass

    best_record, attacked, residual = best_payload
    attacked_path = backend_dir / "adversarial.png"
    Image.fromarray(tensor_image_to_uint8(source)).save(backend_dir / "watermarked.png")
    Image.fromarray(tensor_image_to_uint8(attacked)).save(attacked_path)
    summary_path = backend_dir / "summary.json"
    summary = {
        "backend": "in_process_gradnorm",
        "watermarked_image": str(source_image_path),
        "ranking_csv": getattr(args, "ranking_csv", None),
        "steps": int(backend_args.steps),
        "lr": float(backend_args.lr),
        "decoder_weights": normalized_weights,
        "train_methods": [decoder.name for decoder in decoders],
        "final": best_record,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "setting": "gradnorm",
        "backend": "in_process_gradnorm",
        "steps": int(args.steps),
        "requested_steps": int(args.steps),
        "executed_steps": executed_steps,
        "converged": converged,
        "convergence_reason": convergence_reason,
        "convergence_patience": convergence_patience,
        "convergence_min_delta": convergence_min_delta,
        "convergence_warmup": convergence_warmup,
        "convergence_stale_steps": convergence_stale_steps,
        "lr": float(args.lr),
        "decoder_weights": normalized_weights,
        "train_methods": [decoder.name for decoder in decoders],
        "output_dir": str(backend_dir),
        "source_image": str(source_image_path),
        "optimization_log": str(optimization_log_path),
        "best_record": best_record,
        "summary": summary,
    }
    return attacked.detach(), metadata


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_rows_by(rows, ["scale"])


def summarize_rows_by(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped[key].append(row)

    summary = []
    for key in sorted(grouped):
        scale_rows = grouped[key]
        item: dict[str, Any] = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        item["count"] = len(scale_rows)
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in scale_rows if row.get(metric) not in ("", None)]
            if not values:
                item[f"{metric}_mean"] = ""
                item[f"{metric}_std"] = ""
                continue
            array = np.asarray(values, dtype=np.float64)
            item[f"{metric}_mean"] = float(array.mean())
            item[f"{metric}_std"] = float(array.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(path: Path, rows: list[dict[str, Any]], group_keys: list[str]) -> None:
    fieldnames = [*group_keys, "count"]
    for metric in SUMMARY_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    save_csv(path, rows, fieldnames)


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_attack_value(attack: str, row: dict[str, Any]) -> Any:
    attack_value = row.get(
        "scale",
        row.get("quality", row.get("vae_quality", row.get("setting", row.get("attack_value", "")))),
    )
    if attack == "jpeg":
        return row.get("quality", attack_value)
    if attack == "vae":
        return row.get("vae_quality", attack_value)
    if attack == "tps":
        return row.get("scale", attack_value)
    if attack == "adversarial":
        return row.get("setting", attack_value)
    return attack_value


def build_decision_candidates(attack_payloads: list[dict[str, Any]], ber_precision: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for payload in attack_payloads:
        attack = payload["attack"]
        for row in payload.get("summary", []):
            ber = as_float(row.get("bit_error_rate_mean"))
            ber_std = as_float(row.get("bit_error_rate_std"), 0.0) or 0.0
            psnr = as_float(row.get("psnr_mean"))
            lpips_value = as_float(row.get("lpips_mean"))
            if ber is None or psnr is None:
                continue

            attack_value = infer_attack_value(attack, row)

            candidate_id = f"{attack}:{row.get('decoder')}:{attack_value}:ber={ber:.{ber_precision}f}:psnr={psnr:.6f}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "attack": attack,
                    "decoder": row.get("decoder", ""),
                    "attack_value": attack_value,
                    "ber": ber,
                    "ber_std": ber_std,
                    "ber_key": round(ber, ber_precision),
                    "psnr": psnr,
                    "lpips": lpips_value,
                    "bit_acc": 1.0 - ber,
                    "count": row.get("count", ""),
                }
            )
    return candidates


def build_global_attack_candidates(
    candidates: list[dict[str, Any]],
    decoder_weights: dict[str, float],
    success_ber: float,
    std_penalty: float,
    decoder_weights_by_attack: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate one physical attack candidate across uncertain decoder identities.

    Decoder weights are treated as the probability/importance of each decoder being
    the relevant one. A conservative BER (mean - std_penalty * std) prevents a noisy
    TPS point from winning only because of a high mean over a few trials.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    attack_values: dict[tuple[str, str], Any] = {}
    for candidate in candidates:
        key = (str(candidate["attack"]), json.dumps(candidate["attack_value"], sort_keys=True))
        grouped[key].append(candidate)
        attack_values[key] = candidate["attack_value"]

    global_candidates = []
    for key, decoder_candidates in sorted(grouped.items()):
        attack, _ = key
        attack_decoder_weights = (
            (decoder_weights_by_attack or {}).get(attack, decoder_weights)
        )
        normalized_weights = normalize_decoder_weights(attack_decoder_weights)
        attack_value = attack_values[key]
        by_decoder = {str(candidate["decoder"]): candidate for candidate in decoder_candidates}
        decoder_results = []
        weighted_expected_ber = 0.0
        weighted_conservative_ber = 0.0
        weighted_success_probability = 0.0
        weighted_success_score = 0.0
        coverage_weight = 0.0
        psnr_values = []
        lpips_values = []

        for decoder, weight in normalized_weights.items():
            candidate = by_decoder.get(decoder)
            if candidate is None:
                decoder_results.append(
                    {
                        "decoder": decoder,
                        "weight": weight,
                        "missing": True,
                        "ber": None,
                        "ber_std": None,
                        "evaluation_count": 0,
                        "conservative_ber": 0.0,
                        "success": False,
                    }
                )
                continue
            ber = float(candidate["ber"])
            ber_std = max(float(candidate.get("ber_std", 0.0)), 0.0)
            conservative_ber = max(0.0, ber - std_penalty * ber_std)
            success = conservative_ber >= success_ber
            soft_success = min(conservative_ber / success_ber, 1.0) if success_ber > 0.0 else 1.0
            coverage_weight += weight
            weighted_expected_ber += weight * ber
            weighted_conservative_ber += weight * conservative_ber
            weighted_success_probability += weight * float(success)
            weighted_success_score += weight * soft_success
            psnr_values.append(float(candidate["psnr"]))
            if candidate.get("lpips") is not None:
                lpips_values.append(float(candidate["lpips"]))
            decoder_results.append(
                {
                    "decoder": decoder,
                    "weight": weight,
                    "missing": False,
                    "ber": ber,
                    "ber_std": ber_std,
                    "evaluation_count": int(candidate.get("count") or 0),
                    "conservative_ber": conservative_ber,
                    "effective_ber": conservative_ber,
                    "success": success,
                    "soft_success": soft_success,
                }
            )

        global_candidates.append(
            {
                "candidate_id": f"{attack}:{attack_value}",
                "attack": attack,
                "attack_value": attack_value,
                "attack_context": ATTACK_DECISION_CONTEXT.get(attack, {}),
                "decoder_weights": normalized_weights,
                "psnr": float(np.mean(psnr_values)) if psnr_values else 0.0,
                "lpips": float(np.mean(lpips_values)) if lpips_values else None,
                "weighted_success_probability": weighted_success_probability,
                "weighted_success_score": weighted_success_score,
                "weighted_expected_ber": weighted_expected_ber,
                "weighted_conservative_ber": weighted_conservative_ber,
                "weighted_effective_ber": weighted_conservative_ber,
                "decoder_coverage_weight": coverage_weight,
                "evaluation_count_per_decoder": min(
                    (int(candidate.get("count") or 0) for candidate in decoder_candidates), default=0
                ),
                "decoder_results": decoder_results,
            }
        )
    return global_candidates


def choose_global_attack_locally(
    global_candidates: list[dict[str, Any]],
    min_psnr: float | None,
    target_ber: float | None,
    ber_weight: float = 0.5,
    quality_metric: str = "psnr",
) -> dict[str, Any]:
    """Choose the BER/quality Pareto knee nearest to the normalized ideal point."""
    if ber_weight < 0.0 or ber_weight > 1.0:
        raise ValueError("ber_weight must be in [0, 1].")
    if quality_metric not in {"psnr", "lpips"}:
        raise ValueError(f"Unsupported decision quality metric: {quality_metric}")
    if not global_candidates:
        raise ValueError("No global attack candidates are available for decision.")
    decision_candidates = [candidate for candidate in global_candidates if not is_noop_decision_candidate(candidate)]
    if not decision_candidates:
        raise ValueError("No non-baseline attack candidates are available for decision.")
    eligible = [
        candidate for candidate in decision_candidates if min_psnr is None or float(candidate["psnr"]) >= min_psnr
    ]
    psnr_constraint_relaxed = not eligible
    if psnr_constraint_relaxed:
        eligible = decision_candidates

    def effective_ber(candidate: dict[str, Any]) -> float:
        if "weighted_effective_ber" in candidate:
            return float(candidate["weighted_effective_ber"])
        results = candidate.get("decoder_results") or []
        if results:
            return sum(
                float(item.get("weight", 0.0)) * float(item.get("conservative_ber", 0.0))
                for item in results
                if not item.get("missing", False)
            )
        return float(candidate["weighted_conservative_ber"])

    target_candidates = (
        [candidate for candidate in eligible if effective_ber(candidate) >= target_ber]
        if target_ber is not None
        else []
    )
    target_relaxed = target_ber is not None and not target_candidates
    decision_pool = target_candidates if target_candidates else eligible

    def quality_value(candidate: dict[str, Any]) -> float:
        if quality_metric == "psnr":
            return float(candidate["psnr"])
        lpips_value = candidate.get("lpips")
        if lpips_value is None or not math.isfinite(float(lpips_value)):
            raise ValueError(
                f"Candidate {candidate['candidate_id']} has no LPIPS value; rerun attacks with "
                "--decision-quality-metric lpips."
            )
        # The Pareto implementation maximizes both axes; lower LPIPS means higher quality.
        return -float(lpips_value)

    quality_weight = 1.0 - ber_weight

    def pareto_frontier_for(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frontier = []
        for candidate in candidates:
            candidate_ber = effective_ber(candidate)
            candidate_quality = quality_value(candidate)
            dominated = False
            for other in candidates:
                if other is candidate:
                    continue
                other_ber = effective_ber(other)
                other_quality = quality_value(other)
                if other_ber >= candidate_ber and other_quality >= candidate_quality and (
                    other_ber > candidate_ber or other_quality > candidate_quality
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        return frontier

    frontier = pareto_frontier_for(decision_pool)

    ber_values = [effective_ber(candidate) for candidate in frontier]
    quality_values = [quality_value(candidate) for candidate in frontier]
    ber_min, ber_max = min(ber_values), max(ber_values)
    quality_min, quality_max = min(quality_values), max(quality_values)

    scored_frontier = []
    for candidate in frontier:
        candidate_ber = effective_ber(candidate)
        candidate_quality = quality_value(candidate)
        normalized_ber = (candidate_ber - ber_min) / (ber_max - ber_min) if ber_max > ber_min else 1.0
        normalized_quality = (
            (candidate_quality - quality_min) / (quality_max - quality_min)
            if quality_max > quality_min
            else 1.0
        )
        ideal_distance = math.sqrt(
            ber_weight * (1.0 - normalized_ber) ** 2
            + quality_weight * (1.0 - normalized_quality) ** 2
        )
        scored_frontier.append(
            {
                "candidate": candidate,
                "weighted_effective_ber": candidate_ber,
                "normalized_ber": normalized_ber,
                "normalized_quality": normalized_quality,
                "ideal_distance": ideal_distance,
            }
        )

    best_score = min(
        scored_frontier,
        key=lambda item: (
            float(item["ideal_distance"]),
            -quality_value(item["candidate"]),
            -float(item["weighted_effective_ber"]),
        ),
    )
    best = best_score["candidate"]
    reason = (
        f"selects the BER-{quality_metric.upper()} Pareto knee nearest to the normalized ideal point "
        f"using BER weight={ber_weight} and quality weight={quality_weight}"
    )
    if target_ber is not None and not target_relaxed:
        reason += f" among attacks reaching weighted effective BER target={target_ber}"
    if target_relaxed:
        reason += f"; target={target_ber} was relaxed because no attack reached it"
    if psnr_constraint_relaxed:
        reason += "; min_psnr was relaxed because it removed every candidate"

    return {
        "selected_candidate_id": best["candidate_id"],
        "selected_attack": best["attack"],
        "selected_attack_value": best["attack_value"],
        "selected_psnr": best["psnr"],
        "selected_lpips": best.get("lpips"),
        "selected_quality_metric": quality_metric,
        "selected_quality_value": float(best[quality_metric]),
        "weighted_success_probability": best["weighted_success_probability"],
        "weighted_success_score": best["weighted_success_score"],
        "weighted_expected_ber": best["weighted_expected_ber"],
        "weighted_conservative_ber": best["weighted_conservative_ber"],
        "weighted_effective_ber": best_score["weighted_effective_ber"],
        "normalized_ber_score": best_score["normalized_ber"],
        "normalized_quality_score": best_score["normalized_quality"],
        "normalized_psnr_score": best_score["normalized_quality"] if quality_metric == "psnr" else None,
        "ideal_distance": best_score["ideal_distance"],
        "decision_ber_weight": ber_weight,
        "decision_quality_weight": quality_weight,
        "decision_psnr_weight": quality_weight if quality_metric == "psnr" else None,
        "weighted_ber_target": target_ber,
        "decision_rule": f"normalized_weighted_ber_{quality_metric}_pareto_knee",
        "pareto_frontier": [
            {
                "candidate_id": item["candidate"]["candidate_id"],
                "weighted_effective_ber": item["weighted_effective_ber"],
                "psnr": item["candidate"]["psnr"],
                "lpips": item["candidate"].get("lpips"),
                "normalized_ber_score": item["normalized_ber"],
                "normalized_quality_score": item["normalized_quality"],
                "normalized_psnr_score": (
                    item["normalized_quality"] if quality_metric == "psnr" else None
                ),
                "ideal_distance": item["ideal_distance"],
            }
            for item in sorted(scored_frontier, key=lambda value: value["ideal_distance"])
        ],
        "decoder_results": best["decoder_results"],
        "reason": reason,
    }


def save_weighted_ber_psnr_decision_plot(
    path: Path,
    global_candidates: list[dict[str, Any]],
    selection: dict[str, Any],
    quality_metric: str = "psnr",
) -> None:
    """Plot physical attack candidates and the selected Pareto knee."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [candidate for candidate in global_candidates if not is_noop_decision_candidate(candidate)]
    frontier_by_id = {
        str(item["candidate_id"]): item for item in selection.get("pareto_frontier", [])
    }
    attacks = sorted({str(candidate["attack"]) for candidate in candidates})
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {attack: color_cycle[index % len(color_cycle)] for index, attack in enumerate(attacks)}

    def x_value(candidate: dict[str, Any]) -> float:
        return float(candidate[quality_metric])

    def y_value(candidate: dict[str, Any]) -> float:
        if "weighted_effective_ber" in candidate:
            return float(candidate["weighted_effective_ber"])
        return float(candidate["weighted_conservative_ber"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for attack in attacks:
        attack_candidates = [candidate for candidate in candidates if str(candidate["attack"]) == attack]
        ax.scatter(
            [x_value(candidate) for candidate in attack_candidates],
            [y_value(candidate) for candidate in attack_candidates],
            label=attack,
            color=colors[attack],
            alpha=0.75,
            s=42,
        )

    if frontier_by_id:
        frontier = sorted(frontier_by_id.values(), key=x_value)
        ax.plot(
            [x_value(item) for item in frontier],
            [y_value(item) for item in frontier],
            color="black",
            linewidth=1.6,
            linestyle="--",
            label="Pareto frontier",
        )
        for item in frontier:
            ax.annotate(
                str(item["candidate_id"]),
                (x_value(item), y_value(item)),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    ax.scatter(
        [float(selection[f"selected_{quality_metric}"])],
        [float(selection["weighted_effective_ber"])],
        marker="*",
        s=220,
        color="red",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label=f"selected: {selection['selected_candidate_id']}",
    )
    if quality_metric == "lpips":
        ax.set_xlabel("LPIPS to encoded image — lower is better")
    else:
        ax.set_xlabel("PSNR to encoded image (dB) — higher is better")
    ax.set_ylabel("Decoder-weighted conservative BER — higher is better")
    ax.set_title(f"Weighted BER–{quality_metric.upper()} Pareto decision")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def is_noop_decision_candidate(candidate: dict[str, Any]) -> bool:
    return str(candidate.get("attack")) == "tps" and math.isclose(
        float(candidate.get("attack_value", 0.0)), 0.0, abs_tol=1e-12
    )


def save_decision_csv(path: Path, selections: list[dict[str, Any]]) -> None:
    fieldnames = [
        "selected_attack",
        "selected_attack_value",
        "selected_psnr",
        "selected_lpips",
        "selected_quality_metric",
        "selected_quality_value",
        "weighted_success_probability",
        "weighted_success_score",
        "weighted_expected_ber",
        "weighted_conservative_ber",
        "weighted_effective_ber",
        "normalized_ber_score",
        "normalized_quality_score",
        "normalized_psnr_score",
        "ideal_distance",
        "decision_ber_weight",
        "decision_quality_weight",
        "decision_psnr_weight",
        "weighted_ber_target",
        "decision_rule",
        "selected_candidate_id",
        "reason",
    ]
    save_csv(path, selections, fieldnames)


def run_decision_module(
    args: argparse.Namespace,
    output_dir: Path,
    attack_payloads: list[dict[str, Any]],
    decoder_weights: dict[str, float],
    decoder_weights_by_attack: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    decision_dir = output_dir / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_decision_candidates(attack_payloads, args.decision_ber_precision)
    all_weight_sets = list((decoder_weights_by_attack or {}).values()) or [decoder_weights]
    weighted_decoder_names = {
        canonical_method(name)
        for weights in all_weight_sets
        for name in weights
    }
    decision_candidates = [
        candidate
        for candidate in candidates
        if canonical_method(str(candidate["decoder"])) in weighted_decoder_names
    ]
    global_candidates = build_global_attack_candidates(
        decision_candidates,
        decoder_weights,
        args.decision_success_ber,
        args.decision_std_penalty,
        decoder_weights_by_attack=decoder_weights_by_attack,
    )
    rule_selection = choose_global_attack_locally(
        global_candidates,
        args.decision_min_psnr,
        args.decision_target_ber,
        ber_weight=args.decision_ber_weight,
        quality_metric=args.decision_quality_metric,
    )
    decision_input = {
        "ber_precision": args.decision_ber_precision,
        "decoder_weights": decoder_weights,
        "decoder_weights_by_attack": decoder_weights_by_attack or {},
        "selection_policy": {
            "primary_goal": (
                "Remove dominated attacks on the decoder-weighted BER versus "
                f"{args.decision_quality_metric.upper()} plane, normalize the Pareto frontier "
                "within this image's candidates, and select the point nearest to the upper-right ideal."
            ),
            "min_psnr": args.decision_min_psnr,
            "weighted_ber_target": args.decision_target_ber,
            "ber_weight": args.decision_ber_weight,
            "quality_metric": args.decision_quality_metric,
            "quality_weight": 1.0 - args.decision_ber_weight,
            "std_penalty": args.decision_std_penalty,
        },
        "per_decoder_candidates": decision_candidates,
        "global_candidates": global_candidates,
        "rule_selection": rule_selection,
    }
    (decision_dir / "decision_input.json").write_text(
        json.dumps(decision_input, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    decision_plot = decision_dir / f"weighted_ber_vs_{args.decision_quality_metric}.png"
    save_weighted_ber_psnr_decision_plot(
        decision_plot,
        global_candidates,
        rule_selection,
        quality_metric=args.decision_quality_metric,
    )
    decision_payload = {
        "decision_source": f"local_weighted_ber_{args.decision_quality_metric}_pareto_knee",
        "ber_precision": args.decision_ber_precision,
        "decoder_weights": decoder_weights,
        "decoder_weights_by_attack": decoder_weights_by_attack or {},
        "weighted_ber_target": args.decision_target_ber,
        "decision_ber_weight": args.decision_ber_weight,
        "decision_quality_metric": args.decision_quality_metric,
        "decision_quality_weight": 1.0 - args.decision_ber_weight,
        "decision_std_penalty": args.decision_std_penalty,
        "decision_min_psnr": args.decision_min_psnr,
        "decision_target_ber": args.decision_target_ber,
        "decision_plot": str(decision_plot),
        "rule_checked_selection": rule_selection,
        "recommended_selection": rule_selection,
    }
    (decision_dir / "decision.json").write_text(
        json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_decision_csv(decision_dir / "decision_summary.csv", [rule_selection])
    print(f"Rule decision results written to {decision_dir / 'decision.json'}")
    print("Final rule-based selection:")
    print(json.dumps(rule_selection, ensure_ascii=False, indent=2))
    return rule_selection


def run_final_verification(
    args: argparse.Namespace,
    output_dir: Path,
    watermarked: torch.Tensor,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Independently verify the chosen attack with the known true decoder."""
    if not args.true_decoder:
        raise ValueError("Final verification requires --true-decoder.")

    verification_dir = output_dir / "verification"
    images_dir = verification_dir / "attacked_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    true_decoder = canonical_method(args.true_decoder)
    decoder = get_cached_decoder(true_decoder, args)
    decoder.cache_clean_bits(watermarked)

    selected_attack = str(selection["selected_attack"])
    selected_value = selection["selected_attack_value"]
    trial_count = args.verification_trials if selected_attack == "tps" else 1
    vae_model = None
    adversarial_image = None
    verification_source = "fresh_attack_application"

    if selected_attack == "vae":
        vae_model = get_cached_vae_model(args.vae_model_name, int(selected_value), args.device)
    elif selected_attack == "adversarial":
        adversarial_path = output_dir / "adversarial" / "adversarial.png"
        if not adversarial_path.is_file():
            raise FileNotFoundError(f"Selected adversarial artifact does not exist: {adversarial_path}")
        adversarial_image, _ = load_image_tensor(adversarial_path, args.device, args.input_size)
        verification_source = "saved_selected_adversarial_artifact_redecoded_by_true_decoder"

    rows = []
    with torch.no_grad():
        for trial in range(trial_count):
            seed = attack_seed(args.seed + 10_000_000, selected_attack, selected_value, trial)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if selected_attack == "tps":
                scale = float(selected_value)
                if math.isclose(scale, 0.0, abs_tol=1e-12):
                    attacked = watermarked.detach().clone()
                else:
                    attack = build_tps_attack(scale=scale, p=args.tps_p).to(device=args.device)
                    attacked = attack(watermarked).clamp(-1.0, 1.0)
            elif selected_attack == "jpeg":
                attacked = apply_kjpeg_attack(watermarked, quality=int(selected_value))
            elif selected_attack == "vae":
                attacked = apply_vae_attack(
                    watermarked,
                    vae_model,
                    input_size=getattr(
                        args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
                    ),
                )
            elif selected_attack == "adversarial":
                attacked = adversarial_image.detach().clone()
            else:
                raise ValueError(f"Unsupported selected attack for final verification: {selected_attack}")

            image_path = images_dir / f"{selected_attack}_{selected_value}_trial_{trial:02d}.png"
            Image.fromarray(tensor_image_to_uint8(attacked)).save(image_path)
            psnr = psnr_between_neg1_tensors(attacked, watermarked)
            lpips_value = (
                lpips_between_neg1_tensors(attacked, watermarked)
                if getattr(args, "decision_quality_metric", "psnr") == "lpips"
                else None
            )
            metric = evaluate_decoder(decoder, attacked)
            row = {
                "trial": trial,
                "seed": seed,
                "true_decoder": true_decoder,
                "selected_attack": selected_attack,
                "selected_attack_value": selected_value,
                "bit_error_rate": float(metric["bit_error_rate"]),
                "bit_acc": float(metric["bit_acc"]),
                "psnr": psnr,
                "lpips": lpips_value,
                "attacked_image": str(image_path),
            }
            rows.append(row)
            print(
                f"[final verification {trial + 1}/{trial_count}] true_decoder={true_decoder} "
                f"attack={selected_attack}:{selected_value} ber={row['bit_error_rate']:.6f} "
                f"psnr={psnr:.2f} "
                + (f"lpips={lpips_value:.4f}" if lpips_value is not None else ""),
                flush=True,
            )

    ber_values = np.asarray([row["bit_error_rate"] for row in rows], dtype=np.float64)
    bit_acc_values = np.asarray([row["bit_acc"] for row in rows], dtype=np.float64)
    psnr_values = np.asarray([row["psnr"] for row in rows], dtype=np.float64)
    lpips_values = np.asarray(
        [row["lpips"] for row in rows if row.get("lpips") is not None], dtype=np.float64
    )
    verification = {
        "phase": "post_decision_true_decoder_verification",
        "vae_protocol_version": VAE_PROTOCOL_VERSION,
        "vae_input_size": getattr(
            args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
        ),
        "verification_source": verification_source,
        "true_decoder": true_decoder,
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_attack": selected_attack,
        "selected_attack_value": selected_value,
        "trials": trial_count,
        "ber_mean": float(ber_values.mean()),
        "ber_std": float(ber_values.std(ddof=1)) if trial_count > 1 else 0.0,
        "bit_acc_mean": float(bit_acc_values.mean()),
        "bit_acc_std": float(bit_acc_values.std(ddof=1)) if trial_count > 1 else 0.0,
        "psnr_mean": float(psnr_values.mean()),
        "psnr_std": float(psnr_values.std(ddof=1)) if trial_count > 1 else 0.0,
        "lpips_mean": float(lpips_values.mean()) if lpips_values.size else None,
        "lpips_std": (
            float(lpips_values.std(ddof=1)) if lpips_values.size > 1 else 0.0
        ) if lpips_values.size else None,
        "trial_results": rows,
    }
    (verification_dir / "final_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_csv(
        verification_dir / "final_verification.csv",
        rows,
        [
            "trial",
            "seed",
            "true_decoder",
            "selected_attack",
            "selected_attack_value",
            "bit_error_rate",
            "bit_acc",
            "psnr",
            "lpips",
            "attacked_image",
        ],
    )
    final_result = {"decision": selection, "verification": verification}
    decision_dir = output_dir / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "final_result.json").write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Final true-decoder verification:")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return verification


def rows_for_psnr_error_plot(
    summary: list[dict[str, Any]],
    x_key: str = "scale",
    baseline_value: float = 0.0,
) -> list[dict[str, Any]]:
    return [row for row in summary if float(row[x_key]) != baseline_value]


def smooth_curve_points(
    x_values: list[float],
    y_values: list[float],
    samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]
    unique_x, unique_indices = np.unique(x_array, return_index=True)
    x_array = unique_x
    y_array = y_array[unique_indices]
    if len(x_array) < 2:
        return x_array, y_array
    sample_count = max(samples, len(x_array))
    smooth_x = np.linspace(float(x_array[0]), float(x_array[-1]), sample_count)
    if len(x_array) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            smooth_y = PchipInterpolator(x_array, y_array)(smooth_x)
            return smooth_x, smooth_y
        except Exception:
            pass
    smooth_y = np.interp(smooth_x, x_array, y_array)
    return smooth_x, smooth_y


def darken_color(color, factor: float = 0.65) -> tuple[float, float, float]:
    rgb = color[:3]
    return tuple(max(0.0, min(1.0, float(channel) * factor)) for channel in rgb)


def save_curves(
    output_dir: Path,
    summary: list[dict[str, Any]],
    group_key: str | None = None,
    x_key: str = "scale",
    x_label: str = "TPS scale",
    filename_suffix: str = "tps_scale",
    psnr_error_baseline_value: float | None = 0.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_specs = [
        ("psnr", "PSNR to encoded image (dB)", f"psnr_vs_{filename_suffix}.png"),
        ("bit_acc", "Bit accuracy vs clean decoded bits", f"bit_acc_vs_{filename_suffix}.png"),
        ("decoded_delta_from_clean_mean", "Mean decoder probability delta", f"decoder_delta_vs_{filename_suffix}.png"),
    ]
    for metric, ylabel, filename in curve_specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if group_key is None:
            scales = [row[x_key] for row in summary]
            means = [row[f"{metric}_mean"] for row in summary]
            stds = [row[f"{metric}_std"] for row in summary]
            ax.errorbar(scales, means, yerr=stds, marker="o", capsize=3)
        else:
            groups = sorted({row[group_key] for row in summary})
            for group in groups:
                group_rows = [row for row in summary if row[group_key] == group]
                group_rows.sort(key=lambda row: float(row[x_key]))
                scales = [row[x_key] for row in group_rows]
                means = [row[f"{metric}_mean"] for row in group_rows]
                stds = [row[f"{metric}_std"] for row in group_rows]
                ax.errorbar(scales, means, yerr=stds, marker="o", capsize=3, label=str(group))
            ax.legend(title=group_key)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    if group_key is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if psnr_error_baseline_value is None:
            psnr_error_rows = summary
        else:
            psnr_error_rows = rows_for_psnr_error_plot(summary, x_key=x_key, baseline_value=psnr_error_baseline_value)
        groups = sorted({row[group_key] for row in psnr_error_rows})
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for index, group in enumerate(groups):
            base_color = color_cycle[index % len(color_cycle)]
            point_color = darken_color(plt.matplotlib.colors.to_rgb(base_color))
            group_rows = [row for row in psnr_error_rows if row[group_key] == group]
            group_rows.sort(key=lambda row: float(row["psnr_mean"]))
            psnr_means = [row["psnr_mean"] for row in group_rows]
            one_minus_acc_means = [1.0 - float(row["bit_acc_mean"]) for row in group_rows]
            bit_acc_stds = [row["bit_acc_std"] for row in group_rows]
            smooth_x, smooth_y = smooth_curve_points(psnr_means, one_minus_acc_means)
            ax.plot(smooth_x, smooth_y, linewidth=2.0, color=base_color, label=str(group))
            ax.errorbar(
                psnr_means,
                one_minus_acc_means,
                yerr=bit_acc_stds,
                fmt="o",
                capsize=3,
                color=base_color,
                ecolor=base_color,
                markerfacecolor=point_color,
                markeredgecolor=point_color,
            )
        ax.set_xlabel("PSNR to encoded image (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, alpha=0.3)
        ax.legend(title=group_key)
        fig.tight_layout()
        fig.savefig(output_dir / "ber_vs_psnr.png", dpi=180)
        plt.close(fig)


def format_scale(scale: float) -> str:
    return f"{scale:.3f}"


def evaluate_decoder(decoder, image: torch.Tensor) -> dict[str, Any]:
    decoded = decoder.decode(image)
    ber = decoder.bit_error_rate(decoded)
    metric: dict[str, Any] = {
        "bit_error_rate": ber,
        "bit_error_distance_to_random": abs(ber - 0.5),
        "bit_acc": 1.0 - ber,
        "decoded_shape": json.dumps(list(decoded.shape)),
    }
    add_decoded_margin_diagnostics(decoder, decoded, metric)
    return metric


@dataclass(frozen=True)
class SequentialAction:
    """One discrete action in a sequential attack path."""

    attack: str
    value: float | int | str

    @property
    def candidate_id(self) -> str:
        if self.attack == "tps":
            value_text = format(float(self.value), ".6g")
        else:
            value_text = str(self.value)
        return f"{self.attack}:{value_text}"

    def to_dict(self) -> dict[str, Any]:
        parameter = {
            "tps": "scale",
            "jpeg": "quality",
            "vae": "quality",
            "adversarial": "steps",
        }.get(self.attack, "value")
        return {"attack": self.attack, parameter: self.value}


@dataclass
class SequentialState:
    """A searched sequence and its Monte Carlo terminal images."""

    state_id: str
    parent_id: str | None
    depth: int
    actions: tuple[SequentialAction, ...]
    trial_images: list[torch.Tensor] = field(repr=False)
    trial_seed_paths: list[list[int]] = field(default_factory=list)
    robust_ber: float = 0.0
    ber_mean: float = 0.0
    ber_std: float = 0.0
    between_family_std: float = 0.0
    robust_lpips: float = 0.0
    lpips_mean: float = 0.0
    lpips_std: float = 0.0
    psnr_mean: float = 126.02059991327963
    psnr_std: float = 0.0
    per_family_ber: dict[str, float] = field(default_factory=dict)
    trial_records: list[dict[str, Any]] = field(default_factory=list)
    artifact_path: str | None = None
    exceeds_single_frontier: bool = False
    ber_advantage: float | None = None
    lpips_advantage: float | None = None

    @property
    def last_attack(self) -> str:
        return self.actions[-1].attack if self.actions else "clean"


def decoder_family_name(name: str) -> str:
    canonical = canonical_method(name)
    return "fin" if canonical in FIN_FAMILY else canonical


def aggregate_sequential_ber(
    trial_decoder_bers: list[dict[str, float]],
    decoder_weights: dict[str, float],
    lambda_family: float,
    lambda_ber: float,
) -> dict[str, Any]:
    """Aggregate decoder BER by architecture family and penalize disagreement/noise."""
    if not trial_decoder_bers:
        raise ValueError("Sequential BER aggregation requires at least one trial.")
    normalized_decoder_weights = normalize_decoder_weights(decoder_weights)
    family_members: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for decoder, weight in normalized_decoder_weights.items():
        family_members[decoder_family_name(decoder)].append((decoder, weight))

    family_weights = {
        family: sum(weight for _decoder, weight in members)
        for family, members in family_members.items()
    }
    family_weights = normalize_decoder_weights(family_weights)
    trial_family_bers: list[dict[str, float]] = []
    trial_weighted_bers: list[float] = []
    for trial in trial_decoder_bers:
        family_bers: dict[str, float] = {}
        for family, members in family_members.items():
            present = [(decoder, weight) for decoder, weight in members if decoder in trial]
            if not present:
                continue
            present_total = sum(weight for _decoder, weight in present)
            family_bers[family] = sum(
                weight * float(trial[decoder]) for decoder, weight in present
            ) / present_total
        if not family_bers:
            raise ValueError("No decoder BER values matched the configured sequential decoder weights.")
        present_family_total = sum(family_weights[family] for family in family_bers)
        weighted_ber = sum(
            family_weights[family] * ber for family, ber in family_bers.items()
        ) / present_family_total
        trial_family_bers.append(family_bers)
        trial_weighted_bers.append(weighted_ber)

    family_means = {
        family: float(np.mean([trial[family] for trial in trial_family_bers if family in trial]))
        for family in family_members
        if any(family in trial for trial in trial_family_bers)
    }
    present_family_total = sum(family_weights[family] for family in family_means)
    weighted_family_mean = sum(
        family_weights[family] * ber for family, ber in family_means.items()
    ) / present_family_total
    between_family_variance = sum(
        family_weights[family] * (ber - weighted_family_mean) ** 2
        for family, ber in family_means.items()
    ) / present_family_total
    between_family_std = math.sqrt(max(between_family_variance, 0.0))
    trial_array = np.asarray(trial_weighted_bers, dtype=np.float64)
    ber_mean = float(trial_array.mean())
    ber_std = float(trial_array.std(ddof=1)) if len(trial_array) > 1 else 0.0
    robust_ber = max(
        0.0,
        ber_mean - lambda_family * between_family_std - lambda_ber * ber_std,
    )
    return {
        "robust_ber": robust_ber,
        "ber_mean": ber_mean,
        "ber_std": ber_std,
        "between_family_std": between_family_std,
        "per_family_ber": family_means,
        "trial_family_bers": trial_family_bers,
        "trial_weighted_bers": trial_weighted_bers,
    }


def evaluate_sequential_images(
    images: list[torch.Tensor],
    watermarked: torch.Tensor,
    decoders: list[Any],
    decoder_weights: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate complete sequence endpoints relative to the original encoded image."""
    trial_decoder_bers: list[dict[str, float]] = []
    trial_records: list[dict[str, Any]] = []
    for trial, stored_image in enumerate(images):
        image = stored_image.to(args.device)
        lpips_value = lpips_between_neg1_tensors(image, watermarked)
        psnr = psnr_between_neg1_tensors(image, watermarked)
        decoder_bers: dict[str, float] = {}
        with torch.no_grad():
            for decoder in decoders:
                decoder_bers[decoder.name] = float(evaluate_decoder(decoder, image)["bit_error_rate"])
        trial_decoder_bers.append(decoder_bers)
        trial_records.append(
            {
                "trial": trial,
                "lpips": lpips_value,
                "psnr": psnr,
                "decoder_bers": decoder_bers,
            }
        )
        del image

    ber_summary = aggregate_sequential_ber(
        trial_decoder_bers,
        decoder_weights,
        args.sequential_lambda_family,
        args.sequential_lambda_ber,
    )
    lpips_values = np.asarray([record["lpips"] for record in trial_records], dtype=np.float64)
    psnr_values = np.asarray([record["psnr"] for record in trial_records], dtype=np.float64)
    lpips_mean = float(lpips_values.mean())
    lpips_std = float(lpips_values.std(ddof=1)) if len(lpips_values) > 1 else 0.0
    psnr_mean = float(psnr_values.mean())
    psnr_std = float(psnr_values.std(ddof=1)) if len(psnr_values) > 1 else 0.0
    for record, weighted_ber, family_bers in zip(
        trial_records,
        ber_summary["trial_weighted_bers"],
        ber_summary["trial_family_bers"],
    ):
        record["weighted_family_ber"] = weighted_ber
        record["family_bers"] = family_bers
    return {
        **ber_summary,
        "robust_lpips": lpips_mean + args.sequential_lambda_lpips * lpips_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "trial_records": trial_records,
    }


def sequential_pareto_frontier(
    states: list[SequentialState],
    tolerance: float = 1e-12,
) -> list[SequentialState]:
    """Return states not dominated on robust BER (high) and robust LPIPS (low)."""
    frontier: list[SequentialState] = []
    for candidate in states:
        dominated = False
        for other in states:
            if other is candidate:
                continue
            ber_not_worse = other.robust_ber >= candidate.robust_ber - tolerance
            lpips_not_worse = other.robust_lpips <= candidate.robust_lpips + tolerance
            strictly_better = (
                other.robust_ber > candidate.robust_ber + tolerance
                or other.robust_lpips < candidate.robust_lpips - tolerance
            )
            if ber_not_worse and lpips_not_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda state: (state.robust_lpips, -state.robust_ber, state.state_id))


def sequential_hypervolume(states: list[SequentialState], lpips_reference: float | None = None) -> float:
    """Two-dimensional hypervolume for BER high / LPIPS low candidates."""
    if not states:
        return 0.0
    frontier = sequential_pareto_frontier(states)
    max_lpips = max(state.robust_lpips for state in frontier)
    min_lpips = min(state.robust_lpips for state in frontier)
    reference = lpips_reference
    if reference is None:
        reference = max_lpips + max(max_lpips - min_lpips, 1e-3)
    points = [
        (max(float(state.robust_ber), 0.0), max(float(reference - state.robust_lpips), 0.0))
        for state in frontier
    ]
    x_values = sorted({0.0, *(point[0] for point in points)})
    area = 0.0
    for left, right in zip(x_values[:-1], x_values[1:]):
        height = max((quality for ber, quality in points if ber >= right), default=0.0)
        area += (right - left) * height
    return area


def select_three_branch_beam(states: list[SequentialState]) -> list[SequentialState]:
    """Keep the minimum-LPIPS, maximum-BER, and largest-HV-contribution branches."""
    frontier = sequential_pareto_frontier(states)
    if len(frontier) <= 3:
        return frontier
    selected: list[SequentialState] = []
    quality = min(
        frontier,
        key=lambda state: (state.robust_lpips, -state.robust_ber, state.depth, state.state_id),
    )
    attack = min(
        frontier,
        key=lambda state: (-state.robust_ber, state.robust_lpips, state.depth, state.state_id),
    )
    selected.append(quality)
    if attack.state_id != quality.state_id:
        selected.append(attack)

    reference = max(state.robust_lpips for state in frontier) + max(
        max(state.robust_lpips for state in frontier) - min(state.robust_lpips for state in frontier),
        1e-3,
    )
    total_hv = sequential_hypervolume(frontier, reference)
    contribution = {
        state.state_id: total_hv
        - sequential_hypervolume(
            [other for other in frontier if other.state_id != state.state_id], reference
        )
        for state in frontier
    }
    while len(selected) < 3:
        existing_attacks = {state.last_attack for state in selected}
        remaining = [
            state for state in frontier if state.state_id not in {item.state_id for item in selected}
        ]
        if not remaining:
            break
        chosen = min(
            remaining,
            key=lambda state: (
                -contribution[state.state_id],
                -(state.last_attack not in existing_attacks),
                state.ber_std + state.lpips_std,
                state.depth,
                state.state_id,
            ),
        )
        selected.append(chosen)
    return selected


def select_greedy_branch(states: list[SequentialState]) -> list[SequentialState]:
    """Keep only the locally best branch by robust BER, quality, and stability."""
    if not states:
        return []
    return [
        min(
            states,
            key=lambda state: (
                -state.robust_ber,
                state.robust_lpips,
                state.ber_std + state.lpips_std,
                state.state_id,
            ),
        )
    ]


def compare_sequence_to_single_frontier(
    state: SequentialState,
    single_frontier: list[SequentialState],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Measure whether a multi-step state lies outside the depth-one frontier."""
    if not single_frontier:
        raise ValueError("A single-attack frontier is required for sequence comparison.")
    no_worse_quality = [
        item for item in single_frontier if item.robust_lpips <= state.robust_lpips + tolerance
    ]
    no_worse_ber = [
        item for item in single_frontier if item.robust_ber >= state.robust_ber - tolerance
    ]
    best_single_ber = max((item.robust_ber for item in no_worse_quality), default=0.0)
    best_single_lpips = min(
        (item.robust_lpips for item in no_worse_ber),
        default=max(item.robust_lpips for item in single_frontier),
    )
    dominated_or_equal = any(
        item.robust_ber >= state.robust_ber - tolerance
        and item.robust_lpips <= state.robust_lpips + tolerance
        for item in single_frontier
    )
    ber_advantage = state.robust_ber - best_single_ber
    lpips_advantage = best_single_lpips - state.robust_lpips
    exceeds = not dominated_or_equal and (
        ber_advantage > tolerance or lpips_advantage > tolerance
    )
    return {
        "exceeds_single_frontier": exceeds,
        "best_single_ber_at_lpips": best_single_ber,
        "best_single_lpips_at_ber": best_single_lpips,
        "ber_advantage": ber_advantage,
        "lpips_advantage": lpips_advantage,
    }


def choose_sequential_result(
    all_states: list[SequentialState],
    single_frontier: list[SequentialState],
    selection_lpips_limit: float | None,
) -> tuple[SequentialState, str]:
    """Select maximum robust BER, optionally after a strict final-stage LPIPS filter."""
    for state in all_states:
        if state.depth < 2:
            continue
        comparison = compare_sequence_to_single_frontier(state, single_frontier)
        state.exceeds_single_frontier = bool(comparison["exceeds_single_frontier"])
        state.ber_advantage = float(comparison["ber_advantage"])
        state.lpips_advantage = float(comparison["lpips_advantage"])

    feasible_pool = (
        list(all_states)
        if selection_lpips_limit is None
        else [
            state for state in all_states if state.robust_lpips < selection_lpips_limit
        ]
    )
    if not feasible_pool:
        if selection_lpips_limit is None:
            raise RuntimeError("Sequential search produced no candidate for final selection.")
        raise RuntimeError(
            "Sequential search produced no candidate with robust LPIPS "
            f"< {selection_lpips_limit:.6f}."
        )

    return min(
        feasible_pool,
        key=lambda state: (
            -state.robust_ber,
            state.robust_lpips,
            state.depth,
            state.ber_std + state.lpips_std,
            state.state_id,
        ),
    ), (
        "unconstrained_maximum_robust_ber"
        if selection_lpips_limit is None
        else "lpips_feasible_pool_maximum_robust_ber"
    )


def sequential_state_payload(state: SequentialState, include_trials: bool = True) -> dict[str, Any]:
    payload = {
        "state_id": state.state_id,
        "parent_id": state.parent_id,
        "depth": state.depth,
        "sequence": [action.to_dict() for action in state.actions],
        "trial_seed_paths": state.trial_seed_paths,
        "robust_ber": state.robust_ber,
        "ber_mean": state.ber_mean,
        "ber_std": state.ber_std,
        "between_family_std": state.between_family_std,
        "per_family_ber": state.per_family_ber,
        "robust_lpips": state.robust_lpips,
        "lpips_mean": state.lpips_mean,
        "lpips_std": state.lpips_std,
        "psnr_mean": state.psnr_mean,
        "psnr_std": state.psnr_std,
        "artifact_path": state.artifact_path,
        "exceeds_single_frontier": state.exceeds_single_frontier,
        "ber_advantage": state.ber_advantage,
        "lpips_advantage": state.lpips_advantage,
    }
    if include_trials:
        payload["trial_records"] = state.trial_records
    return payload


def read_top_methods_from_ranking_csv(path: Path, top_k: int) -> list[str]:
    """Select Top-K decoders, allowing only the highest-ranked FIN variant."""
    methods = []
    selected_fin_method: str | None = None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("status") == "ok"]
    rows.sort(key=lambda row: int(row.get("rank") or 999999))
    for row in rows:
        method = canonical_method(row["method"])
        # FIN_JPEG and FIN_HEAVY are variants of the same decoder family. The
        # entropy ranking is ordered by descending self-calibrated confidence,
        # so only its first (highest-confidence) FIN member gets a Top-K slot.
        if method in FIN_FAMILY:
            if selected_fin_method is not None:
                continue
            selected_fin_method = method
        if method not in methods:
            methods.append(method)
        if len(methods) >= top_k:
            break
    if len(methods) < top_k:
        raise RuntimeError(f"Only found {len(methods)} valid ranked decoder(s) in {path}; need {top_k}.")
    return methods


def read_random_methods_from_ranking_csv(
    path: Path,
    top_k: int,
    selection_seed: int,
    sample_key: str,
    exclude_methods: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Uniformly sample decoder-family slots from valid ranking rows.

    Entropy scores and ranks are deliberately ignored. FIN variants share one
    family slot; if that slot is drawn, its concrete variant is also sampled
    uniformly. A SHA-256-derived seed makes the draw stable per image without
    perturbing the attack/TPS RNG stream.
    """
    excluded = {canonical_method(method) for method in (exclude_methods or [])}
    if excluded & FIN_FAMILY:
        excluded.update(FIN_FAMILY)
    family_variants: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            method = canonical_method(row.get("method", ""))
            if not method or method in excluded:
                continue
            family = "fin" if method in FIN_FAMILY else method
            family_variants[family].add(method)

    families = sorted(family_variants)
    if len(families) < top_k:
        raise RuntimeError(
            f"Only found {len(families)} eligible decoder family slot(s) in {path}; need {top_k}."
        )
    seed_material = f"{int(selection_seed)}\0{sample_key}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(derived_seed)
    sampled_families = rng.sample(families, top_k)
    methods = [rng.choice(sorted(family_variants[family])) for family in sampled_families]
    metadata = {
        "candidate_pool_families": families,
        "candidate_pool_variants": {
            family: sorted(family_variants[family]) for family in families
        },
        "randomly_selected_families": sampled_families,
        "randomly_selected_methods": methods,
        "candidate_selection_seed": int(selection_seed),
        "candidate_selection_sample_key": sample_key,
        "candidate_selection_derived_seed": derived_seed,
    }
    return methods, metadata


def read_decoder_weights_from_ranking_csv(path: Path, methods: list[str]) -> dict[str, float]:
    wanted = {canonical_method(method) for method in methods}
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = canonical_method(row.get("method", ""))
            if row.get("status") != "ok" or method not in wanted:
                continue
            score = as_float(row.get("self_calibrated_score"))
            if score is not None and math.isfinite(score) and score >= 0.0:
                scores[method] = score
    missing = sorted(wanted - scores.keys())
    if missing:
        raise RuntimeError(f"Ranking CSV has no valid self_calibrated_score for: {', '.join(missing)}")
    return normalize_decoder_weights(scores)


def resolve_method_weights(
    methods: list[str],
    explicit_weights: str | None,
    ranking_csv: Path | None,
    reserved_weights: dict[str, float] | None = None,
) -> tuple[dict[str, float], str]:
    canonical_methods = [canonical_method(method) for method in methods]
    if explicit_weights:
        parsed = parse_decoder_weights(explicit_weights)
        missing = [method for method in canonical_methods if method not in parsed]
        extra = [method for method in parsed if method not in canonical_methods]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"unknown={extra}")
            raise ValueError("--decoder-weights must exactly match evaluated decoders: " + ", ".join(details))
        return normalize_decoder_weights({method: parsed[method] for method in canonical_methods}), "command_line"

    reserved = {
        canonical_method(method): float(weight)
        for method, weight in (reserved_weights or {}).items()
        if canonical_method(method) in canonical_methods
    }
    if any(not math.isfinite(weight) or weight < 0.0 for weight in reserved.values()):
        raise ValueError("Reserved decoder weights must be finite and non-negative.")
    unreserved_methods = [method for method in canonical_methods if method not in reserved]
    if not unreserved_methods:
        return normalize_decoder_weights(reserved), "reserved_weights_only"
    reserved_total = sum(reserved.values())
    if reserved_total >= 1.0:
        raise ValueError("Reserved decoder weights must sum to less than 1 when other decoders are present.")

    if ranking_csv is not None:
        base_weights = read_decoder_weights_from_ranking_csv(ranking_csv, unreserved_methods)
        source = "entropy_self_calibrated_score"
    else:
        base_weights = normalize_decoder_weights({method: 1.0 for method in unreserved_methods})
        source = "uniform_default"
    remaining_weight = 1.0 - reserved_total
    weights = {method: weight * remaining_weight for method, weight in base_weights.items()}
    weights.update(reserved)
    if reserved:
        source += "_with_reserved_weights"
    return weights, source


def add_forced_candidate_methods(
    methods: list[str],
    force_methods: list[str] | None,
    exclude_methods: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Append forced decoder candidates while preserving selection order and uniqueness."""
    excluded = {canonical_method(method) for method in (exclude_methods or [])}
    selected: list[str] = []
    for method in methods:
        canonical = canonical_method(method)
        if canonical not in excluded and canonical not in selected:
            selected.append(canonical)

    added: list[str] = []
    for method in force_methods or []:
        canonical = canonical_method(method)
        if canonical in excluded or canonical in selected:
            continue
        selected.append(canonical)
        added.append(canonical)
    return selected, added


def run_entropy_ranking(args: argparse.Namespace, ranking_dir: Path) -> Path:
    ranking_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(ENTROPY_SCRIPT),
        "--watermarked-image",
        args.watermarked_image,
        "--output-dir",
        str(ranking_dir),
        "--device",
        str(args.device),
        "--reference-output-root",
        str(ranking_dir / "reference_encodings"),
    ]
    if args.skip_reference_encoder:
        command.append("--skip-reference-encoder")
    for method in args.skip_entropy_method or []:
        command.extend(["--skip-method", method])
    if args.input_size is not None:
        command.extend(["--input-size", str(args.input_size)])

    pass_through_options = [
        "mbrs_model_path",
        "mbrs_h",
        "mbrs_w",
        "mbrs_message_length",
        "hidden_options_file",
        "hidden_checkpoint_file",
        "pimog_image_size",
        "pimog_embedding_epoch",
        "pimog_distortion",
        "pimog_model_save_dir",
        "pimog_model_name",
        "stegastamp_model_path",
        "stegastamp_image_size",
        "stegastamp_message_length",
        "cin_options_file",
        "cin_checkpoint",
        "fin_noise_type",
        "fin_fed_checkpoint",
        "fin_inl_checkpoint",
        "fin_heavy_fed_checkpoint",
        "fin_heavy_inl_checkpoint",
        "fin_jpeg_fed_checkpoint",
        "trustmark_model_type",
        "trustmark_encoding_type",
        "invismark_ckpt",
        "rosteals_config",
        "rosteals_weight",
        "rosteals_image_size",
        "lightweightmark_model_path",
        "lightweightmark_mode",
        "lightweightmark_message_length",
        "lightweightmark_h",
        "lightweightmark_w",
        "videoseal_model_name",
        "videoseal_short_edge",
        "chunkyseal_model_name",
        "chunkyseal_short_edge",
    ]
    for option in pass_through_options:
        value = getattr(args, option, None)
        if value is not None:
            command.extend([f"--{option.replace('_', '-')}", str(value)])
    boolean_options = [
        "mbrs_with_diffusion",
        "videoseal_lowres_attenuation",
        "chunkyseal_lowres_attenuation",
    ]
    for option in boolean_options:
        if getattr(args, option, False):
            command.append(f"--{option.replace('_', '-')}")
    if getattr(args, "videoseal_message_length", None):
        command.extend(["--videoseal-message-length", str(args.videoseal_message_length)])
    if getattr(args, "chunkyseal_message_length", None):
        command.extend(["--chunkyseal-message-length", str(args.chunkyseal_message_length)])

    log_path = ranking_dir / "entropy_ranking_run.log"
    print("Running entropy ranking:")
    print(" ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return ranking_dir / "entropy_ranking.csv"


def select_eval_methods(args: argparse.Namespace, output_dir: Path) -> tuple[list[str], dict[str, Any]]:
    if args.eval_method:
        methods, forced_methods = add_forced_candidate_methods(
            args.eval_method, args.force_method, args.exclude_method
        )
        weights, weight_source = resolve_method_weights(
            methods, args.decoder_weights, None
        )
        return methods, {
            "source": "manual_eval_method",
            "methods": methods,
            "forced_candidate_methods": forced_methods,
            "decoder_importance_weights": weights,
            "weight_source": weight_source,
        }
    if args.ranking_csv:
        ranking_csv = Path(args.ranking_csv)
    else:
        ranking_csv = run_entropy_ranking(args, output_dir / "entropy_ranking")
    selection_mode = getattr(args, "candidate_selection", "entropy_topk")
    random_metadata: dict[str, Any] = {}
    if selection_mode == "random":
        ranked_methods, random_metadata = read_random_methods_from_ranking_csv(
            ranking_csv,
            args.candidate_top_k,
            args.candidate_selection_seed,
            str(args.watermarked_image),
            args.exclude_method,
        )
    else:
        ranked_methods = read_top_methods_from_ranking_csv(ranking_csv, args.candidate_top_k)
    methods, forced_methods = add_forced_candidate_methods(
        ranked_methods, args.force_method, args.exclude_method
    )
    random_weighting = getattr(args, "random_candidate_weighting", "uniform")
    requested_weighting = getattr(args, "candidate_weighting", "auto")
    if requested_weighting == "auto":
        candidate_weighting = random_weighting if selection_mode == "random" else "entropy"
    else:
        candidate_weighting = requested_weighting
    weights, weight_source = resolve_method_weights(
        methods,
        args.decoder_weights,
        None if candidate_weighting == "uniform" else ranking_csv,
    )
    selection_payload = {
        "source": "random_candidate_pool" if selection_mode == "random" else "entropy_ranking",
        "candidate_selection": selection_mode,
        "ranking_csv": str(ranking_csv),
        "methods": methods,
        "ranked_methods": ranked_methods,
        "forced_candidate_methods": forced_methods,
        "decoder_importance_weights": weights,
        "weight_source": weight_source,
        "candidate_weighting": candidate_weighting,
        "random_candidate_weighting": random_weighting if selection_mode == "random" else None,
    }
    selection_payload.update(random_metadata)
    return methods, selection_payload


def build_argparser() -> argparse.ArgumentParser:
    decoder_parser = build_decoder_argparser()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TPS, JPEG, VAE reconstruction, and one joint adversarial attack against candidate decoders."
        ),
        parents=[decoder_parser],
        conflict_handler="resolve",
    )
    parser.set_defaults(
        watermarked_image=DEFAULT_IMAGE,
        output_dir=DEFAULT_OUTPUT_DIR,
        eval_method=None,
        ranking_csv=None,
        input_size=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        force_method=None,
    )
    parser.add_argument(
        "--scales",
        default="0.02,0.05,0.08,0.1,0.15,0.2,0.25,0.3",
        help="Comma-separated TPS scale values.",
    )
    parser.add_argument(
        "--jpeg-qualities",
        default="90,80,70,60,50,40,30",
        help="Comma-separated JPEG quality factors for the KJpeg attack.",
    )
    parser.add_argument(
        "--vae-qualities",
        default="6,5,4,3,2,1",
        help="Comma-separated CompressAI VAE quality levels.",
    )
    parser.add_argument(
        "--vae-model-name",
        default="cheng2020-anchor",
        choices=["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean", "mbt2018", "cheng2020-anchor"],
        help="CompressAI model used for the VAE regeneration attack.",
    )
    parser.add_argument(
        "--vae-input-size",
        type=int,
        default=OFFICIAL_VAE_INPUT_SIZE,
        choices=[OFFICIAL_VAE_INPUT_SIZE],
        help=(
            "Spatial input size used by the official WatermarkAttacker VAE "
            "protocol (fixed at 512)."
        ),
    )
    parser.add_argument("--trials", type=int, default=10, help="Random TPS samples per nonzero scale.")
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=3,
        help=(
            "Number of selected decoders to evaluate before adding explicitly forced candidates. "
            "FIN variants share one candidate-family slot."
        ),
    )
    parser.add_argument(
        "--candidate-selection",
        choices=["entropy_topk", "random"],
        default="entropy_topk",
        help=(
            "Candidate decoder selection policy (default: entropy_topk). random uniformly "
            "samples family-deduplicated valid decoders using a stable per-image seed."
        ),
    )
    parser.add_argument(
        "--candidate-selection-seed",
        type=int,
        default=0,
        help="Global seed for reproducible per-image random candidate selection (default: 0).",
    )
    parser.add_argument(
        "--candidate-weighting",
        choices=["auto", "uniform", "entropy"],
        default="auto",
        help=(
            "Weights for selected candidates. uniform gives every selected decoder equal weight; "
            "entropy normalizes self-calibrated entropy-ranking scores; auto preserves the legacy "
            "behavior (entropy for entropy_topk, --random-candidate-weighting for random)."
        ),
    )
    parser.add_argument(
        "--random-candidate-weighting",
        choices=["uniform", "entropy"],
        default="uniform",
        help=(
            "Weights for randomly selected candidates (default: uniform). entropy is an "
            "optional ablation that reuses self-calibrated entropy scores after random sampling."
        ),
    )
    parser.add_argument(
        "--decoder-weights",
        default=None,
        help=(
            "Optional decoder importance weights, for example "
            "'lightweightmark=0.50,fin=0.30,cin=0.20'. They must match all evaluated decoders. "
            "Without this option, entropy-selected decoders use normalized self-calibrated scores; "
            "randomly or manually selected decoders use uniform weights."
        ),
    )
    parser.add_argument(
        "--skip-entropy-method",
        action="append",
        default=[],
        help="Method to skip during automatic entropy ranking. Repeatable.",
    )
    parser.add_argument(
        "--skip-reference-encoder",
        action="store_true",
        help="Forwarded to decoder_entropy_ranking.py.",
    )
    parser.add_argument(
        "--tps-p",
        type=float,
        default=1.0,
        help="Kornia RandomThinPlateSpline probability. Use 1.0 for deterministic scale sweeps.",
    )
    parser.add_argument(
        "--save-attacked-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save every attacked image used for decoder evaluation.",
    )
    parser.add_argument(
        "--enable-adversarial-attack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run one joint gradient-based attack using attack_topk_decoders.py defaults as the fourth attack "
            "family. Use --no-enable-adversarial-attack to skip the expensive optimization."
        ),
    )
    parser.add_argument("--adversarial-gradnorm-quality-scale", type=float, default=180.0)
    parser.add_argument("--adversarial-max-perturbation", type=float, default=0.0)
    parser.add_argument("--adversarial-watermark-weight", type=float, default=10.0)
    parser.add_argument("--adversarial-target-lpips", type=float, default=0.05)
    parser.add_argument("--adversarial-lpips-net", default="alex")
    parser.add_argument("--adversarial-selection-min-step", type=int, default=0)
    parser.add_argument("--adversarial-selection-min-psnr", type=float, default=0.0)
    parser.add_argument(
        "--adversarial-quality-constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-decision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the deterministic weighted-BER/PSNR decision module after all attack sweeps.",
    )
    parser.add_argument(
        "--true-decoder",
        default=None,
        help="Known watermark decoder loaded only after local-rule selection for final verification.",
    )
    parser.add_argument(
        "--verification-trials",
        type=int,
        default=5,
        help="Independent final-verification trials for a selected stochastic TPS attack (default: 5).",
    )
    parser.add_argument(
        "--decision-ber-precision",
        type=int,
        default=4,
        help="Decimal places used to decide whether BER values are the same.",
    )
    parser.add_argument(
        "--decision-min-psnr",
        type=float,
        default=None,
        help=(
            "Optional minimum PSNR for decision candidates. If every point is filtered out "
            "for a decoder, the full curve is used."
        ),
    )
    parser.add_argument(
        "--decision-target-ber",
        type=float,
        default=None,
        help=(
            "Optional explicit weighted conservative BER target. Among attacks reaching it, "
            "the rule chooses the highest PSNR."
        ),
    )
    parser.add_argument(
        "--decision-ber-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of normalized weighted BER when measuring distance to the Pareto upper-right ideal; "
            "PSNR receives 1-weight (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--decision-quality-metric",
        choices=["psnr", "lpips"],
        default="psnr",
        help=(
            "Image-quality axis used by the local Pareto decision: PSNR is maximized; "
            "LPIPS is minimized (default: psnr). Both metrics remain recorded when LPIPS is selected."
        ),
    )
    parser.add_argument(
        "--decision-success-ber",
        type=float,
        default=0.1,
        help=(
            "Per-decoder BER threshold retained for diagnostic success fields and final batch reporting; "
            "the Pareto decision itself uses continuous decoder-weighted BER (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--decision-std-penalty",
        type=float,
        default=1.0,
        help="Subtract this many BER standard deviations when judging attack success (default: 1.0).",
    )
    parser.add_argument(
        "--enable-sequential-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Search multi-step attacks on the robust-BER/LPIPS Pareto frontier. When enabled, "
            "the depth-one search states replace the legacy single-attack sweep and decision."
        ),
    )
    parser.add_argument(
        "--sequential-max-steps",
        type=int,
        default=4,
        help=(
            "Safety ceiling on attacks in one searched sequence (default: 4); Pareto convergence "
            "may stop expansion earlier."
        ),
    )
    parser.add_argument(
        "--sequential-search-strategy",
        choices=["pareto_beam", "greedy"],
        default="pareto_beam",
        help=(
            "Branch-retention policy (default: pareto_beam). pareto_beam keeps three "
            "complementary Pareto branches; greedy keeps only the locally highest robust-BER "
            "branch, breaking ties by lower robust LPIPS and lower uncertainty."
        ),
    )
    parser.add_argument(
        "--sequential-beam-width",
        type=int,
        default=3,
        help=(
            "States retained per depth: 3 for pareto_beam or 1 for greedy (default: 3)."
        ),
    )
    parser.add_argument(
        "--sequential-tps-scales",
        default="0.00025,0.0005,0.001,0.002",
        help="Comma-separated incremental TPS scales used by sequential search.",
    )
    parser.add_argument(
        "--sequential-jpeg-qualities",
        default="90,80,70,60,50,40,30",
        help="Comma-separated incremental JPEG qualities used by sequential search.",
    )
    parser.add_argument(
        "--sequential-vae-qualities",
        default="6,5,4,3,2,1",
        help="Comma-separated incremental VAE qualities used by sequential search.",
    )
    parser.add_argument(
        "--sequential-adversarial-steps",
        default="50,100,200",
        help="Comma-separated micro-optimization step counts used as sequential adversarial actions.",
    )
    parser.add_argument(
        "--sequential-tps-search-trials",
        type=int,
        default=3,
        help="Monte Carlo trajectories retained while searching a sequence containing TPS (default: 3).",
    )
    parser.add_argument(
        "--sequential-verification-trials",
        type=int,
        default=20,
        help="Fresh full-sequence replays when the selected sequence contains TPS (default: 20).",
    )
    parser.add_argument(
        "--sequential-replay-verification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replay the selected attack sequence during true-decoder verification. "
            "Disable to evaluate only the fixed selected.png artifact (default: enabled)."
        ),
    )
    parser.add_argument(
        "--sequential-lambda-family",
        type=float,
        default=0.5,
        help="Penalty for disagreement between decoder-family BER values (default: 0.5).",
    )
    parser.add_argument(
        "--sequential-lambda-ber",
        type=float,
        default=1.0,
        help="Penalty for BER variation between stochastic sequence trajectories (default: 1.0).",
    )
    parser.add_argument(
        "--sequential-lambda-lpips",
        type=float,
        default=0.5,
        help="Penalty added to mean LPIPS for stochastic sequence trajectories (default: 0.5).",
    )
    parser.add_argument(
        "--sequential-search-lpips-limit",
        "--sequential-lpips-limit",
        dest="sequential_lpips_limit",
        type=float,
        default=None,
        help=(
            "Absolute upper bound on robust LPIPS before candidates enter the sequential "
            "Pareto frontier. Disabled by default. --sequential-lpips-limit is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--sequential-selection-lpips-limit",
        type=float,
        default=0.05,
        help=(
            "Strict robust-LPIPS upper bound applied only to final sequential selection "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--sequential-ber-target",
        type=float,
        default=0.1,
        help=(
            "Robust proxy-BER target used for search artifacts and reporting (default: 0.1). "
            "Final selection maximizes robust BER, optionally under "
            "--sequential-selection-lpips-limit."
        ),
    )
    parser.add_argument(
        "--sequential-enable-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop adding sequence depths after both Pareto hypervolume and maximum robust BER "
            "remain below their minimum gains for the configured patience (default: disabled)."
        ),
    )
    parser.add_argument(
        "--sequential-convergence-patience",
        type=int,
        default=2,
        help="Consecutive stagnant sequence depths required for convergence (default: 2).",
    )
    parser.add_argument(
        "--sequential-min-relative-hv-gain",
        type=float,
        default=0.001,
        help="Minimum relative archive hypervolume gain per depth (default: 0.001).",
    )
    parser.add_argument(
        "--sequential-min-ber-gain",
        type=float,
        default=0.002,
        help="Minimum maximum-robust-BER gain per depth (default: 0.002).",
    )
    parser.add_argument(
        "--sequential-adversarial-convergence-patience",
        type=int,
        default=15,
        help="Consecutive adversarial iterations without meaningful loss improvement (default: 15).",
    )
    parser.add_argument(
        "--sequential-adversarial-convergence-min-delta",
        type=float,
        default=1e-4,
        help="Minimum adversarial loss improvement that resets convergence patience (default: 1e-4).",
    )
    parser.add_argument(
        "--sequential-adversarial-convergence-warmup",
        type=int,
        default=10,
        help="Adversarial iterations completed before plateau stopping is allowed (default: 10).",
    )
    parser.add_argument(
        "--batch-manifest",
        default=None,
        help=(
            "TSV manifest with image_path, output_dir, and optionally true_decoder columns. "
            "All rows run sequentially in one process with model reuse."
        ),
    )
    parser.add_argument(
        "--batch-rerun-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry rows containing an existing .failed marker (default: true).",
    )
    parser.add_argument(
        "--batch-exclude-true-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply each manifest row's true_decoder as its candidate exclusion.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[float], list[int], list[int]]:
    scales = parse_scales(args.scales)
    jpeg_qualities = parse_int_list(args.jpeg_qualities, minimum=1, maximum=100, label="JPEG quality")
    vae_qualities = parse_int_list(args.vae_qualities, minimum=1, maximum=6, label="VAE quality")
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.verification_trials <= 0:
        raise ValueError("--verification-trials must be positive.")
    if args.tps_p < 0.0 or args.tps_p > 1.0:
        raise ValueError("--tps-p must be in [0, 1].")
    if args.enable_adversarial_attack:
        if args.steps <= 0:
            raise ValueError("--steps must be positive for the adversarial attack.")
        if args.lr <= 0.0:
            raise ValueError("--lr must be positive for the adversarial attack.")
        if args.adversarial_max_perturbation < 0.0:
            raise ValueError("--adversarial-max-perturbation must be non-negative.")
        if args.log_interval <= 0:
            raise ValueError("--log-interval must be positive.")
    if args.candidate_top_k <= 0:
        raise ValueError("--candidate-top-k must be positive.")
    if args.decision_ber_precision < 0:
        raise ValueError("--decision-ber-precision must be non-negative.")
    if args.decision_min_psnr is not None and args.decision_min_psnr < 0.0:
        raise ValueError("--decision-min-psnr must be non-negative.")
    if args.decision_target_ber is not None and (args.decision_target_ber < 0.0 or args.decision_target_ber > 1.0):
        raise ValueError("--decision-target-ber must be in [0, 1].")
    if args.decision_ber_weight < 0.0 or args.decision_ber_weight > 1.0:
        raise ValueError("--decision-ber-weight must be in [0, 1].")
    if args.decision_success_ber <= 0.0 or args.decision_success_ber > 1.0:
        raise ValueError("--decision-success-ber must be in (0, 1].")
    if args.decision_std_penalty < 0.0:
        raise ValueError("--decision-std-penalty must be non-negative.")
    if args.sequential_max_steps <= 0:
        raise ValueError("--sequential-max-steps must be positive.")
    if args.sequential_beam_width <= 0:
        raise ValueError("--sequential-beam-width must be positive.")
    if args.enable_sequential_search:
        required_beam_width = 1 if args.sequential_search_strategy == "greedy" else 3
        if args.sequential_beam_width != required_beam_width:
            raise ValueError(
                f"Sequential strategy {args.sequential_search_strategy!r} requires "
                f"--sequential-beam-width {required_beam_width}."
            )
    if args.sequential_tps_search_trials <= 0:
        raise ValueError("--sequential-tps-search-trials must be positive.")
    if args.sequential_verification_trials <= 0:
        raise ValueError("--sequential-verification-trials must be positive.")
    if args.sequential_lambda_family < 0.0:
        raise ValueError("--sequential-lambda-family must be non-negative.")
    if args.sequential_lambda_ber < 0.0:
        raise ValueError("--sequential-lambda-ber must be non-negative.")
    if args.sequential_lambda_lpips < 0.0:
        raise ValueError("--sequential-lambda-lpips must be non-negative.")
    if args.sequential_lpips_limit is not None and (
        not math.isfinite(args.sequential_lpips_limit)
        or args.sequential_lpips_limit < 0.0
    ):
        raise ValueError("--sequential-search-lpips-limit must be non-negative and finite.")
    if args.sequential_selection_lpips_limit is not None and (
        not math.isfinite(args.sequential_selection_lpips_limit)
        or args.sequential_selection_lpips_limit <= 0.0
    ):
        raise ValueError("--sequential-selection-lpips-limit must be positive and finite.")
    if args.sequential_ber_target < 0.0 or args.sequential_ber_target > 1.0:
        raise ValueError("--sequential-ber-target must be in [0, 1].")
    if args.sequential_convergence_patience <= 0:
        raise ValueError("--sequential-convergence-patience must be positive.")
    if args.sequential_min_relative_hv_gain < 0.0:
        raise ValueError("--sequential-min-relative-hv-gain must be non-negative.")
    if args.sequential_min_ber_gain < 0.0:
        raise ValueError("--sequential-min-ber-gain must be non-negative.")
    if args.sequential_adversarial_convergence_patience <= 0:
        raise ValueError("--sequential-adversarial-convergence-patience must be positive.")
    if args.sequential_adversarial_convergence_min_delta < 0.0:
        raise ValueError("--sequential-adversarial-convergence-min-delta must be non-negative.")
    if args.sequential_adversarial_convergence_warmup < 0:
        raise ValueError("--sequential-adversarial-convergence-warmup must be non-negative.")
    parse_scales(args.sequential_tps_scales)
    parse_int_list(
        args.sequential_jpeg_qualities, minimum=1, maximum=100, label="sequential JPEG quality"
    )
    parse_int_list(
        args.sequential_vae_qualities, minimum=1, maximum=6, label="sequential VAE quality"
    )
    parse_int_list(
        args.sequential_adversarial_steps, minimum=1, label="sequential adversarial steps"
    )
    if args.decoder_weights:
        parse_decoder_weights(args.decoder_weights)
    return scales, jpeg_qualities, vae_qualities


def attack_seed(base_seed: int, attack_name: str, value: float | int | str, trial: int) -> int:
    offsets = {"tps": 0, "jpeg": 1_000_000, "vae": 2_000_000, "adversarial": 3_000_000}
    offset = offsets.get(attack_name, 3_000_000)
    try:
        value_offset = round(float(value) * 10_000)
    except (TypeError, ValueError):
        value_offset = sum((index + 1) * ord(character) for index, character in enumerate(str(value)))
    return int(base_seed + offset + value_offset + trial)


def run_attack_sweep(
    *,
    args: argparse.Namespace,
    attack_name: str,
    values: list[float] | list[int],
    value_key: str,
    value_label: str,
    filename_suffix: str,
    decoders: list[Any],
    watermarked: torch.Tensor,
    output_dir: Path,
    method_selection: dict[str, Any],
    original_size,
    input_size,
) -> dict[str, Any]:
    images_dir = output_dir / "attacked_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_attacked_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if attack_name == "tps":
        repeat_counts = [1 if float(value) == 0.0 else args.trials for value in values]
    else:
        repeat_counts = [1 for _ in values]
    total = sum(repeat_counts) * len(decoders)
    completed = 0
    with torch.no_grad():
        for value, repeat_count in zip(values, repeat_counts):
            for trial in range(repeat_count):
                seed = attack_seed(args.seed, attack_name, value, trial)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                if attack_name == "tps":
                    if float(value) == 0.0:
                        attacked = watermarked.detach().clone()
                    else:
                        attack = build_tps_attack(scale=float(value), p=args.tps_p).to(device=args.device)
                        attacked = attack(watermarked).clamp(-1.0, 1.0)
                elif attack_name == "jpeg":
                    attacked = apply_kjpeg_attack(watermarked, quality=int(value))
                elif attack_name == "vae":
                    vae_model = get_cached_vae_model(args.vae_model_name, int(value), args.device)
                    attacked = apply_vae_attack(
                        watermarked,
                        vae_model,
                        input_size=getattr(
                            args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported attack: {attack_name}")

                image_path = ""
                if args.save_attacked_images:
                    if value_key == "scale":
                        value_text = format_scale(float(value))
                    else:
                        value_text = str(value)
                    filename = f"{value_key}_{value_text}_trial_{trial:02d}.png"
                    image_path = str(images_dir / filename)
                    Image.fromarray(tensor_image_to_uint8(attacked)).save(image_path)

                psnr = psnr_between_neg1_tensors(attacked, watermarked)
                lpips_value = (
                    lpips_between_neg1_tensors(attacked, watermarked)
                    if getattr(args, "decision_quality_metric", "psnr") == "lpips"
                    else None
                )
                for decoder in decoders:
                    metric = evaluate_decoder(decoder, attacked)
                    row = {
                        "attack": attack_name,
                        "attack_value": value,
                        "scale": value if value_key == "scale" else "",
                        "quality": value if value_key == "quality" else "",
                        "vae_quality": value if value_key == "vae_quality" else "",
                        value_key: value,
                        "trial": trial,
                        "seed": seed,
                        "image_path": image_path,
                        "decoder": decoder.name,
                        "psnr": psnr,
                        "lpips": lpips_value,
                    }
                    row.update(metric)
                    rows.append(row)
                    completed += 1
                    print(
                        f"[{attack_name} {completed}/{total}] {value_key}={value} trial={trial} "
                        f"decoder={decoder.name} psnr={psnr:.2f} "
                        + (f"lpips={lpips_value:.4f} " if lpips_value is not None else "")
                        + f"bit_acc={row['bit_acc']:.4f} "
                        f"delta={row.get('decoded_delta_from_clean_mean', 0.0):.6f}",
                        flush=True,
                    )

    summary = summarize_rows_by(rows, ["decoder", value_key])
    save_csv(output_dir / "metrics.csv", rows, METRIC_COLUMNS)
    save_summary_csv(output_dir / "summary.csv", summary, ["decoder", value_key])
    save_curves(
        output_dir,
        summary,
        group_key="decoder",
        x_key=value_key,
        x_label=value_label,
        filename_suffix=filename_suffix,
        psnr_error_baseline_value=0.0 if attack_name == "tps" else None,
    )

    curves = [
        str(output_dir / f"psnr_vs_{filename_suffix}.png"),
        str(output_dir / f"bit_acc_vs_{filename_suffix}.png"),
        str(output_dir / "ber_vs_psnr.png"),
        str(output_dir / f"decoder_delta_vs_{filename_suffix}.png"),
    ]
    payload = {
        "attack": attack_name,
        "watermarked_image": args.watermarked_image,
        "original_size": list(original_size),
        "input_size": input_size,
        "method_selection": method_selection,
        value_key + "s": values,
        "trials": args.trials if attack_name == "tps" else 1,
        "tps_p": args.tps_p if attack_name == "tps" else None,
        "vae_model_name": args.vae_model_name if attack_name == "vae" else None,
        "vae_input_size": (
            getattr(args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE)
            if attack_name == "vae"
            else None
        ),
        "seed": args.seed,
        "metrics_csv": str(output_dir / "metrics.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "curves": curves,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{attack_name} results written to {output_dir}")
    return payload


def run_adversarial_attack(
    *,
    args: argparse.Namespace,
    decoders: list[Any],
    decoder_weights: dict[str, float],
    evaluation_decoders: list[Any] | None = None,
    evaluation_decoder_weights: dict[str, float] | None = None,
    attack_method_selection: dict[str, Any] | None = None,
    watermarked: torch.Tensor,
    output_dir: Path,
    method_selection: dict[str, Any],
    original_size,
    input_size,
) -> dict[str, Any]:
    """Evaluate the single default joint adversarial attack as a fourth attack family."""
    output_dir.mkdir(parents=True, exist_ok=True)
    attacked, attack_config = apply_adversarial_attack(
        watermarked,
        decoders,
        decoder_weights,
        args,
        output_dir,
    )
    validation_decoders = evaluation_decoders or decoders
    validation_decoder_weights = evaluation_decoder_weights or decoder_weights
    image_path = output_dir / "adversarial.png"
    Image.fromarray(tensor_image_to_uint8(attacked)).save(image_path)
    psnr = psnr_between_neg1_tensors(attacked, watermarked)
    lpips_value = (
        lpips_between_neg1_tensors(attacked, watermarked)
        if getattr(args, "decision_quality_metric", "psnr") == "lpips"
        else None
    )
    rows = []
    with torch.no_grad():
        for decoder in validation_decoders:
            metric = evaluate_decoder(decoder, attacked)
            row = {
                "attack": "adversarial",
                "attack_value": "default",
                "scale": "",
                "quality": "",
                "vae_quality": "",
                "setting": "default",
                "trial": 0,
                "seed": args.seed,
                "image_path": str(image_path),
                "decoder": decoder.name,
                "psnr": psnr,
                "lpips": lpips_value,
            }
            row.update(metric)
            rows.append(row)
            print(
                f"[adversarial] setting=default decoder={decoder.name} "
                f"psnr={psnr:.2f} "
                + (f"lpips={lpips_value:.4f} " if lpips_value is not None else "")
                + f"bit_acc={row['bit_acc']:.4f} "
                f"delta={row.get('decoded_delta_from_clean_mean', 0.0):.6f}",
                flush=True,
            )

    summary = summarize_rows_by(rows, ["decoder", "setting"])
    save_csv(output_dir / "metrics.csv", rows, METRIC_COLUMNS)
    save_summary_csv(output_dir / "summary.csv", summary, ["decoder", "setting"])
    payload = {
        "attack": "adversarial",
        "watermarked_image": args.watermarked_image,
        "original_size": list(original_size),
        "input_size": input_size,
        "method_selection": method_selection,
        "attack_method_selection": attack_method_selection or method_selection,
        "validation_decoder_weights": validation_decoder_weights,
        "settings": ["default"],
        "trials": 1,
        "seed": args.seed,
        "attack_config": attack_config,
        "attacked_image": str(image_path),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "summary_csv": str(output_dir / "summary.csv"),
        "curves": [],
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adversarial results written to {output_dir}")
    return payload


def build_sequential_actions(args: argparse.Namespace) -> list[SequentialAction]:
    actions = [
        *(SequentialAction("tps", value) for value in parse_scales(args.sequential_tps_scales)),
        *(
            SequentialAction("jpeg", value)
            for value in parse_int_list(
                args.sequential_jpeg_qualities,
                minimum=1,
                maximum=100,
                label="sequential JPEG quality",
            )
        ),
        *(
            SequentialAction("vae", value)
            for value in parse_int_list(
                args.sequential_vae_qualities,
                minimum=1,
                maximum=6,
                label="sequential VAE quality",
            )
        ),
    ]
    if args.enable_adversarial_attack:
        actions.extend(
            SequentialAction("adversarial", value)
            for value in parse_int_list(
                args.sequential_adversarial_steps,
                minimum=1,
                label="sequential adversarial steps",
            )
        )
    return actions


def build_single_attack_baseline_actions(
    args: argparse.Namespace,
    sequential_actions: list[SequentialAction],
) -> list[SequentialAction]:
    """Union incremental actions with the legacy full single-attack parameter sweep."""
    baseline = list(sequential_actions)
    baseline.extend(
        SequentialAction("tps", value)
        for value in parse_scales(getattr(args, "scales", args.sequential_tps_scales))
        if not math.isclose(float(value), 0.0, abs_tol=1e-12)
    )
    baseline.extend(
        SequentialAction("jpeg", value)
        for value in parse_int_list(
            getattr(args, "jpeg_qualities", args.sequential_jpeg_qualities),
            minimum=1,
            maximum=100,
            label="JPEG quality",
        )
    )
    baseline.extend(
        SequentialAction("vae", value)
        for value in parse_int_list(
            getattr(args, "vae_qualities", args.sequential_vae_qualities),
            minimum=1,
            maximum=6,
            label="VAE quality",
        )
    )
    if args.enable_adversarial_attack:
        baseline.append(SequentialAction("adversarial", int(getattr(args, "steps", 500))))
    unique: dict[str, SequentialAction] = {}
    for action in baseline:
        unique.setdefault(action.candidate_id, action)
    return list(unique.values())


def sequential_action_allowed(state: SequentialState, action: SequentialAction) -> bool:
    counts = defaultdict(int)
    for previous in state.actions:
        counts[previous.attack] += 1
    return counts[action.attack] < SEQUENTIAL_ATTACK_LIMITS.get(action.attack, 1)


def sequential_action_seed(
    base_seed: int,
    state_id: str,
    action: SequentialAction,
    trial: int,
    replay_offset: int = 0,
) -> int:
    text_value = f"{state_id}|{action.candidate_id}"
    stable_offset = sum((index + 1) * ord(character) for index, character in enumerate(text_value))
    return int(base_seed + replay_offset + stable_offset * 101 + trial)


def snapshot_decoder_attack_state(decoders: list[Any]) -> list[dict[str, Any]]:
    attribute_names = (
        "clean_bits",
        "clean_decoded",
    )
    return [
        {name: getattr(decoder, name, None) for name in attribute_names}
        for decoder in decoders
    ]


def restore_decoder_attack_state(decoders: list[Any], snapshots: list[dict[str, Any]]) -> None:
    for decoder, snapshot in zip(decoders, snapshots):
        for name, value in snapshot.items():
            setattr(decoder, name, value)


def apply_sequential_action_once(
    image: torch.Tensor,
    action: SequentialAction,
    seed: int,
    args: argparse.Namespace,
    adversarial_decoders: list[Any],
    adversarial_decoder_weights: dict[str, float],
    action_output_dir: Path,
) -> torch.Tensor:
    """Apply one action while preserving x0 decoder caches across adversarial optimization."""
    set_seed(seed)
    image = image.to(args.device)
    if action.attack == "tps":
        attack = build_tps_attack(float(action.value), args.tps_p).to(device=args.device)
        return attack(image).clamp(-1.0, 1.0).detach()
    if action.attack == "jpeg":
        return apply_kjpeg_attack(image, int(action.value)).detach()
    if action.attack == "vae":
        model = get_cached_vae_model(args.vae_model_name, int(action.value), args.device)
        with torch.no_grad():
            return apply_vae_attack(
                image,
                model,
                input_size=getattr(
                    args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
                ),
            ).detach()
    if action.attack == "adversarial":
        action_args = copy.copy(args)
        action_args.steps = int(action.value)
        action_args.log_interval = max(int(action.value), 1)
        action_args.adversarial_convergence_patience = getattr(
            args, "sequential_adversarial_convergence_patience", 15
        )
        action_args.adversarial_convergence_min_delta = getattr(
            args, "sequential_adversarial_convergence_min_delta", 1e-4
        )
        action_args.adversarial_convergence_warmup = getattr(
            args, "sequential_adversarial_convergence_warmup", 10
        )
        snapshots = snapshot_decoder_attack_state(adversarial_decoders)
        try:
            attacked, metadata = apply_adversarial_attack(
                image,
                adversarial_decoders,
                adversarial_decoder_weights,
                action_args,
                action_output_dir,
            )
            (action_output_dir / "sequential_adversarial_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return attacked.detach()
        finally:
            restore_decoder_attack_state(adversarial_decoders, snapshots)
    raise ValueError(f"Unsupported sequential action: {action.attack}")


def expand_sequential_state(
    state: SequentialState,
    action: SequentialAction,
    watermarked: torch.Tensor,
    evaluation_decoders: list[Any],
    evaluation_decoder_weights: dict[str, float],
    adversarial_decoders: list[Any],
    adversarial_decoder_weights: dict[str, float],
    args: argparse.Namespace,
    work_dir: Path,
) -> SequentialState:
    has_previous_tps = any(previous.attack == "tps" for previous in state.actions)
    if action.attack == "tps" and not has_previous_tps and len(state.trial_images) == 1:
        source_indices = [0] * args.sequential_tps_search_trials
    else:
        source_indices = list(range(len(state.trial_images)))

    state_id = f"{state.state_id}__{action.candidate_id}"
    safe_state_id = state_id.replace(":", "_").replace(".", "p")
    images: list[torch.Tensor] = []
    seed_paths: list[list[int]] = []
    for output_trial, source_index in enumerate(source_indices):
        seed = sequential_action_seed(args.seed, state.state_id, action, output_trial)
        source = state.trial_images[source_index]
        action_dir = work_dir / "adversarial_actions" / safe_state_id / f"trial_{output_trial:02d}"
        attacked = apply_sequential_action_once(
            source,
            action,
            seed,
            args,
            adversarial_decoders,
            adversarial_decoder_weights,
            action_dir,
        )
        images.append(attacked.detach().cpu())
        previous_seed_path = (
            state.trial_seed_paths[source_index] if state.trial_seed_paths else []
        )
        seed_paths.append([*previous_seed_path, seed])
        del attacked

    metrics = evaluate_sequential_images(
        images,
        watermarked,
        evaluation_decoders,
        evaluation_decoder_weights,
        args,
    )
    return SequentialState(
        state_id=state_id,
        parent_id=state.state_id,
        depth=state.depth + 1,
        actions=(*state.actions, action),
        trial_images=images,
        trial_seed_paths=seed_paths,
        robust_ber=metrics["robust_ber"],
        ber_mean=metrics["ber_mean"],
        ber_std=metrics["ber_std"],
        between_family_std=metrics["between_family_std"],
        robust_lpips=metrics["robust_lpips"],
        lpips_mean=metrics["lpips_mean"],
        lpips_std=metrics["lpips_std"],
        psnr_mean=metrics["psnr_mean"],
        psnr_std=metrics["psnr_std"],
        per_family_ber=metrics["per_family_ber"],
        trial_records=metrics["trial_records"],
    )


def representative_trial_index(state: SequentialState, ber_target: float) -> int:
    if not state.trial_records:
        return 0
    reaching = [
        record
        for record in state.trial_records
        if float(record.get("weighted_family_ber", 0.0)) >= ber_target
    ]
    pool = reaching or state.trial_records
    chosen = min(
        pool,
        key=lambda record: (
            float(record["lpips"]) if reaching else -float(record.get("weighted_family_ber", 0.0)),
            -float(record.get("weighted_family_ber", 0.0)),
            int(record["trial"]),
        ),
    )
    return int(chosen["trial"])


def save_sequential_state_artifact(
    state: SequentialState,
    artifacts_dir: Path,
    ber_target: float,
) -> None:
    if state.artifact_path is not None or not state.trial_images:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_state_id = state.state_id.replace(":", "_").replace(".", "p")
    path = artifacts_dir / f"{safe_state_id}.png"
    trial = min(representative_trial_index(state, ber_target), len(state.trial_images) - 1)
    Image.fromarray(tensor_image_to_uint8(state.trial_images[trial])).save(path)
    state.artifact_path = str(path)


def save_sequential_search_plot(
    path: Path,
    single_frontier: list[SequentialState],
    sequence_frontier: list[SequentialState],
    selected: SequentialState,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if single_frontier:
        ordered = sorted(single_frontier, key=lambda state: state.robust_lpips)
        ax.plot(
            [state.robust_lpips for state in ordered],
            [state.robust_ber for state in ordered],
            "o--",
            label="single-attack frontier",
        )
    if sequence_frontier:
        ordered = sorted(sequence_frontier, key=lambda state: state.robust_lpips)
        ax.plot(
            [state.robust_lpips for state in ordered],
            [state.robust_ber for state in ordered],
            "s-",
            label="multi-step frontier",
        )
    ax.scatter(
        [selected.robust_lpips],
        [selected.robust_ber],
        marker="*",
        s=220,
        color="red",
        edgecolor="black",
        label=f"selected: {selected.state_id}",
        zorder=5,
    )
    ax.set_xlabel("Robust LPIPS to encoded image — lower is better")
    ax.set_ylabel("Robust proxy BER — higher is better")
    ax.set_title("Sequential BER–LPIPS Pareto search")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_sequential_search(
    args: argparse.Namespace,
    output_dir: Path,
    watermarked: torch.Tensor,
    evaluation_decoders: list[Any],
    evaluation_decoder_weights: dict[str, float],
    adversarial_decoders: list[Any],
    adversarial_decoder_weights: dict[str, float],
) -> dict[str, Any]:
    sequential_dir = output_dir / "sequential"
    artifacts_dir = sequential_dir / "artifacts" / "states"
    sequential_dir.mkdir(parents=True, exist_ok=True)
    actions = build_sequential_actions(args)
    search_strategy = getattr(args, "sequential_search_strategy", "pareto_beam")
    baseline_actions = build_single_attack_baseline_actions(args, actions)
    incremental_action_ids = {action.candidate_id for action in actions}
    root = SequentialState(
        state_id="root",
        parent_id=None,
        depth=0,
        actions=(),
        trial_images=[watermarked.detach().cpu()],
        trial_seed_paths=[[]],
    )
    beam = [root]
    all_states: list[SequentialState] = []
    depth_frontiers: dict[int, list[str]] = {}
    convergence_history: list[dict[str, Any]] = []
    previous_hypervolume: float | None = None
    previous_max_ber: float | None = None
    stagnant_depths = 0
    search_converged = False
    search_stop_reason = "max_steps_reached"

    for depth in range(1, args.sequential_max_steps + 1):
        expanded: list[SequentialState] = []
        depth_actions = baseline_actions if depth == 1 else actions
        total_actions = sum(
            1
            for state in beam
            for action in depth_actions
            if sequential_action_allowed(state, action)
        )
        completed = 0
        for state in beam:
            for action in depth_actions:
                if not sequential_action_allowed(state, action):
                    continue
                completed += 1
                print(
                    f"[sequential depth {depth}] candidate {completed}/{total_actions} "
                    f"parent={state.state_id} action={action.candidate_id}",
                    flush=True,
                )
                candidate = expand_sequential_state(
                    state,
                    action,
                    watermarked,
                    evaluation_decoders,
                    evaluation_decoder_weights,
                    adversarial_decoders,
                    adversarial_decoder_weights,
                    args,
                    sequential_dir,
                )
                if (
                    args.sequential_lpips_limit is not None
                    and candidate.robust_lpips > args.sequential_lpips_limit
                ):
                    print(
                        f"  rejected LPIPS={candidate.robust_lpips:.6f} "
                        f"> limit={args.sequential_lpips_limit:.6f}",
                        flush=True,
                    )
                    continue
                save_sequential_state_artifact(
                    candidate, artifacts_dir, args.sequential_ber_target
                )
                expanded.append(candidate)
                all_states.append(candidate)
                print(
                    f"  robust_ber={candidate.robust_ber:.6f} "
                    f"robust_lpips={candidate.robust_lpips:.6f} psnr={candidate.psnr_mean:.2f}",
                    flush=True,
                )
        if not expanded:
            print(f"[sequential depth {depth}] no eligible candidates; stopping", flush=True)
            search_stop_reason = "no_eligible_candidates"
            break
        frontier = sequential_pareto_frontier(expanded)
        beam_pool = (
            [
                state
                for state in expanded
                if state.actions[-1].candidate_id in incremental_action_ids
            ]
            if depth == 1
            else expanded
        )
        beam_frontier = sequential_pareto_frontier(beam_pool)
        if search_strategy == "greedy":
            beam = select_greedy_branch(beam_pool)
        elif search_strategy == "pareto_beam":
            beam = select_three_branch_beam(beam_frontier)
        else:
            raise ValueError(f"Unsupported sequential search strategy: {search_strategy}")
        depth_frontiers[depth] = [state.state_id for state in frontier]
        keep_ids = {state.state_id for state in beam}
        for state in expanded:
            if state.state_id not in keep_ids:
                state.trial_images.clear()
        print(
            f"[sequential depth {depth}] full_frontier={len(frontier)} "
            f"beam_frontier={len(beam_frontier)} retained={len(beam)} "
            + ", ".join(state.state_id for state in beam),
            flush=True,
        )
        archive_frontier = sequential_pareto_frontier(all_states)
        lpips_reference = (
            float(args.sequential_lpips_limit)
            if args.sequential_lpips_limit is not None
            else max(1.0, max(state.robust_lpips for state in archive_frontier) + 1e-3)
        )
        archive_hypervolume = sequential_hypervolume(
            archive_frontier, lpips_reference=lpips_reference
        )
        archive_max_ber = max(state.robust_ber for state in all_states)
        if previous_hypervolume is None:
            relative_hv_gain = None
            max_ber_gain = None
            stagnant = False
        else:
            relative_hv_gain = (
                archive_hypervolume - previous_hypervolume
            ) / max(abs(previous_hypervolume), 1e-12)
            max_ber_gain = archive_max_ber - float(previous_max_ber)
            stagnant = (
                relative_hv_gain < args.sequential_min_relative_hv_gain
                and max_ber_gain < args.sequential_min_ber_gain
            )
        stagnant_depths = stagnant_depths + 1 if stagnant else 0
        convergence_record = {
            "depth": depth,
            "archive_hypervolume": archive_hypervolume,
            "archive_max_robust_ber": archive_max_ber,
            "relative_hypervolume_gain": relative_hv_gain,
            "maximum_ber_gain": max_ber_gain,
            "stagnant": stagnant,
            "stagnant_depths": stagnant_depths,
            "lpips_reference": lpips_reference,
        }
        convergence_history.append(convergence_record)
        print(
            f"[sequential convergence depth {depth}] hv={archive_hypervolume:.8f} "
            f"relative_hv_gain={relative_hv_gain if relative_hv_gain is not None else 'baseline'} "
            f"max_ber={archive_max_ber:.6f} "
            f"ber_gain={max_ber_gain if max_ber_gain is not None else 'baseline'} "
            f"stagnant_depths={stagnant_depths}/{args.sequential_convergence_patience}",
            flush=True,
        )
        previous_hypervolume = archive_hypervolume
        previous_max_ber = archive_max_ber
        if (
            getattr(args, "sequential_enable_convergence", True)
            and stagnant_depths >= args.sequential_convergence_patience
        ):
            search_converged = True
            search_stop_reason = "pareto_hypervolume_and_maximum_ber_plateau"
            print(
                f"[sequential] converged after depth {depth}: {search_stop_reason}",
                flush=True,
            )
            break

    depth_one = [state for state in all_states if state.depth == 1]
    if not depth_one:
        raise RuntimeError("Sequential search produced no single-attack baseline candidates.")
    single_frontier = sequential_pareto_frontier(depth_one)
    multi_states = [state for state in all_states if state.depth >= 2]
    sequence_frontier = sequential_pareto_frontier(multi_states) if multi_states else []
    selected, selection_source = choose_sequential_result(
        all_states,
        single_frontier,
        args.sequential_selection_lpips_limit,
    )
    if selected.artifact_path is None:
        save_sequential_state_artifact(selected, artifacts_dir, args.sequential_ber_target)
    selected_artifact = sequential_dir / "artifacts" / "selected.png"
    selected_artifact.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(selected.artifact_path) as image:
        image.save(selected_artifact)

    all_rows = []
    for state in all_states:
        payload = sequential_state_payload(state, include_trials=False)
        all_rows.append(
            {
                **payload,
                "sequence": json.dumps(payload["sequence"], ensure_ascii=False),
                "trial_seed_paths": json.dumps(payload["trial_seed_paths"]),
                "per_family_ber": json.dumps(payload["per_family_ber"], ensure_ascii=False),
            }
        )
    save_csv(
        sequential_dir / "all_states.csv",
        all_rows,
        [
            "state_id",
            "parent_id",
            "depth",
            "sequence",
            "trial_seed_paths",
            "robust_ber",
            "ber_mean",
            "ber_std",
            "between_family_std",
            "per_family_ber",
            "robust_lpips",
            "lpips_mean",
            "lpips_std",
            "psnr_mean",
            "psnr_std",
            "artifact_path",
            "exceeds_single_frontier",
            "ber_advantage",
            "lpips_advantage",
        ],
    )
    (sequential_dir / "search_tree.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "max_steps": args.sequential_max_steps,
                    "search_strategy": search_strategy,
                    "beam_width": args.sequential_beam_width,
                    "attack_occurrence_limits": dict(SEQUENTIAL_ATTACK_LIMITS),
                    "vae_model_name": getattr(
                        args, "vae_model_name", "cheng2020-anchor"
                    ),
                    "vae_input_size": getattr(
                        args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
                    ),
                    "actions": [action.to_dict() for action in actions],
                    "single_attack_baseline_actions": [
                        action.to_dict() for action in baseline_actions
                    ],
                    "tps_search_trials": args.sequential_tps_search_trials,
                    "lambda_family": args.sequential_lambda_family,
                    "lambda_ber": args.sequential_lambda_ber,
                    "lambda_lpips": args.sequential_lambda_lpips,
                    "search_lpips_limit": args.sequential_lpips_limit,
                    "lpips_limit": args.sequential_lpips_limit,
                    "selection_lpips_limit": args.sequential_selection_lpips_limit,
                    "ber_target": args.sequential_ber_target,
                    "convergence_enabled": getattr(
                        args, "sequential_enable_convergence", True
                    ),
                    "convergence_patience": args.sequential_convergence_patience,
                    "minimum_relative_hypervolume_gain": (
                        args.sequential_min_relative_hv_gain
                    ),
                    "minimum_maximum_ber_gain": args.sequential_min_ber_gain,
                    "method_selection": getattr(args, "active_method_selection", {}),
                },
                "depth_frontiers": depth_frontiers,
                "convergence": {
                    "converged": search_converged,
                    "stop_reason": search_stop_reason,
                    "executed_depths": len(convergence_history),
                    "history": convergence_history,
                },
                "states": [sequential_state_payload(state) for state in all_states],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sequential_dir / "one_step_lpips_frontier.json").write_text(
        json.dumps(
            [sequential_state_payload(state) for state in single_frontier],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (sequential_dir / "sequence_lpips_frontier.json").write_text(
        json.dumps(
            [sequential_state_payload(state) for state in sequence_frontier],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selection_mode = f"sequential_ber_lpips_{search_strategy}"
    best_payload = {
        "selection_mode": selection_mode,
        "search_strategy": search_strategy,
        "selection_source": selection_source,
        "selection_lpips_limit": args.sequential_selection_lpips_limit,
        "beam_width": args.sequential_beam_width,
        "search_lpips_limit": args.sequential_lpips_limit,
        "search_converged": search_converged,
        "search_stop_reason": search_stop_reason,
        "executed_depths": len(convergence_history),
        "convergence_history": convergence_history,
        "method_selection": getattr(args, "active_method_selection", {}),
        "selected_artifact": str(selected_artifact),
        "selected_state": sequential_state_payload(selected),
    }
    (sequential_dir / "best_sequence.json").write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_sequential_search_plot(
        sequential_dir / "ber_vs_lpips_frontier.png",
        single_frontier,
        sequence_frontier,
        selected,
    )
    print("Sequential selection:")
    print(json.dumps(best_payload, ensure_ascii=False, indent=2), flush=True)
    return {
        "selected_state": selected,
        "selection_source": selection_source,
        "selected_artifact": str(selected_artifact),
        "single_frontier": single_frontier,
        "sequence_frontier": sequence_frontier,
        "best_payload": best_payload,
    }


def replay_sequential_actions(
    watermarked: torch.Tensor,
    actions: tuple[SequentialAction, ...],
    replay_trial: int,
    args: argparse.Namespace,
    adversarial_decoders: list[Any],
    adversarial_decoder_weights: dict[str, float],
    output_dir: Path,
) -> tuple[torch.Tensor, list[int]]:
    image = watermarked.detach().clone()
    state_id = "root"
    seeds: list[int] = []
    for step, action in enumerate(actions, start=1):
        seed = sequential_action_seed(
            args.seed,
            state_id,
            action,
            step,
            replay_offset=10_000_000 + replay_trial * 100_000,
        )
        image = apply_sequential_action_once(
            image,
            action,
            seed,
            args,
            adversarial_decoders,
            adversarial_decoder_weights,
            output_dir / f"step_{step:02d}_{action.attack}",
        )
        seeds.append(seed)
        state_id = f"{state_id}__{action.candidate_id}"
    return image.detach(), seeds


def run_sequential_verification(
    args: argparse.Namespace,
    output_dir: Path,
    watermarked: torch.Tensor,
    search_result: dict[str, Any],
    adversarial_decoders: list[Any],
    adversarial_decoder_weights: dict[str, float],
) -> dict[str, Any]:
    if not args.true_decoder:
        raise ValueError("Sequential final verification requires --true-decoder.")
    selected: SequentialState = search_result["selected_state"]
    verification_dir = output_dir / "verification"
    sequence_verification_dir = output_dir / "sequential" / "verification"
    images_dir = verification_dir / "attacked_images"
    replay_work_dir = sequence_verification_dir / "replay_work"
    images_dir.mkdir(parents=True, exist_ok=True)
    replay_enabled = getattr(args, "sequential_replay_verification", True)
    if replay_enabled:
        replay_work_dir.mkdir(parents=True, exist_ok=True)

    true_decoder = canonical_method(args.true_decoder)
    decoder = get_cached_decoder(true_decoder, args)
    decoder.cache_clean_bits(watermarked)
    sequence_payload = [action.to_dict() for action in selected.actions]

    fixed_artifact, _ = load_image_tensor(
        search_result["selected_artifact"], args.device, args.input_size
    )
    fixed_metric = evaluate_decoder(decoder, fixed_artifact)
    fixed_result = {
        "artifact": search_result["selected_artifact"],
        "bit_error_rate": float(fixed_metric["bit_error_rate"]),
        "bit_acc": float(fixed_metric["bit_acc"]),
        "lpips": lpips_between_neg1_tensors(fixed_artifact, watermarked),
        "psnr": psnr_between_neg1_tensors(fixed_artifact, watermarked),
    }

    selected_attack = "sequential" if selected.depth > 1 else selected.actions[0].attack
    if replay_enabled:
        contains_stochastic_attack = any(action.attack == "tps" for action in selected.actions)
        trial_count = (
            args.sequential_verification_trials if contains_stochastic_attack else 1
        )
        rows: list[dict[str, Any]] = []
        for trial in range(trial_count):
            attacked, seeds = replay_sequential_actions(
                watermarked,
                selected.actions,
                trial,
                args,
                adversarial_decoders,
                adversarial_decoder_weights,
                replay_work_dir / f"trial_{trial:02d}",
            )
            image_path = images_dir / f"sequential_trial_{trial:02d}.png"
            Image.fromarray(tensor_image_to_uint8(attacked)).save(image_path)
            metric = evaluate_decoder(decoder, attacked)
            row = {
                "trial": trial,
                "seeds": json.dumps(seeds),
                "true_decoder": true_decoder,
                "selected_attack": selected_attack,
                "selected_attack_value": json.dumps(sequence_payload, ensure_ascii=False),
                "sequence": json.dumps(sequence_payload, ensure_ascii=False),
                "bit_error_rate": float(metric["bit_error_rate"]),
                "bit_acc": float(metric["bit_acc"]),
                "psnr": psnr_between_neg1_tensors(attacked, watermarked),
                "lpips": lpips_between_neg1_tensors(attacked, watermarked),
                "attacked_image": str(image_path),
            }
            rows.append(row)
            print(
                f"[sequential verification {trial + 1}/{trial_count}] "
                f"true_decoder={true_decoder} ber={row['bit_error_rate']:.6f} "
                f"lpips={row['lpips']:.6f} psnr={row['psnr']:.2f}",
                flush=True,
            )
            del attacked
        verification_source = "fixed_selected_artifact_and_fresh_full_sequence_replays"
    else:
        trial_count = 1
        rows = [
            {
                "trial": 0,
                "seeds": json.dumps([]),
                "true_decoder": true_decoder,
                "selected_attack": selected_attack,
                "selected_attack_value": json.dumps(sequence_payload, ensure_ascii=False),
                "sequence": json.dumps(sequence_payload, ensure_ascii=False),
                "bit_error_rate": fixed_result["bit_error_rate"],
                "bit_acc": fixed_result["bit_acc"],
                "psnr": fixed_result["psnr"],
                "lpips": fixed_result["lpips"],
                "attacked_image": fixed_result["artifact"],
            }
        ]
        verification_source = "fixed_selected_artifact_only"
        print(
            "[sequential fixed-artifact verification] "
            f"true_decoder={true_decoder} ber={fixed_result['bit_error_rate']:.6f} "
            f"lpips={fixed_result['lpips']:.6f} psnr={fixed_result['psnr']:.2f}",
            flush=True,
        )

    ber_values = np.asarray([row["bit_error_rate"] for row in rows], dtype=np.float64)
    bit_acc_values = np.asarray([row["bit_acc"] for row in rows], dtype=np.float64)
    lpips_values = np.asarray([row["lpips"] for row in rows], dtype=np.float64)
    psnr_values = np.asarray([row["psnr"] for row in rows], dtype=np.float64)
    ber_mean = float(ber_values.mean())
    ber_std = float(ber_values.std(ddof=1)) if len(ber_values) > 1 else 0.0
    verification = {
        "phase": "post_sequential_decision_true_decoder_verification",
        "vae_protocol_version": VAE_PROTOCOL_VERSION,
        "vae_input_size": getattr(
            args, "vae_input_size", OFFICIAL_VAE_INPUT_SIZE
        ),
        "verification_source": verification_source,
        "selection_mode": search_result.get("best_payload", {}).get(
            "selection_mode", "sequential_ber_lpips_pareto_beam"
        ),
        "selection_source": search_result["selection_source"],
        "method_selection": search_result.get("best_payload", {}).get(
            "method_selection", {}
        ),
        "true_decoder": true_decoder,
        "selected_candidate_id": selected.state_id,
        "selected_attack": selected_attack,
        "selected_attack_value": sequence_payload,
        "selected_sequence": sequence_payload,
        "selected_depth": selected.depth,
        "exceeds_single_frontier": selected.exceeds_single_frontier,
        "proxy_robust_ber": selected.robust_ber,
        "proxy_robust_lpips": selected.robust_lpips,
        "ber_advantage": selected.ber_advantage,
        "lpips_advantage": selected.lpips_advantage,
        "fixed_artifact_result": fixed_result,
        "trials": trial_count,
        "ber_mean": ber_mean,
        "ber_std": ber_std,
        "ber_lower_confidence_bound": max(0.0, ber_mean - ber_std),
        "success_ber_threshold": args.decision_success_ber,
        "success_probability": float(
            np.mean(ber_values >= args.decision_success_ber)
        ),
        "bit_acc_mean": float(bit_acc_values.mean()),
        "bit_acc_std": float(bit_acc_values.std(ddof=1)) if len(bit_acc_values) > 1 else 0.0,
        "lpips_mean": float(lpips_values.mean()),
        "lpips_std": float(lpips_values.std(ddof=1)) if len(lpips_values) > 1 else 0.0,
        "psnr_mean": float(psnr_values.mean()),
        "psnr_std": float(psnr_values.std(ddof=1)) if len(psnr_values) > 1 else 0.0,
        "trial_results": rows,
    }
    verification_dir.mkdir(parents=True, exist_ok=True)
    sequence_verification_dir.mkdir(parents=True, exist_ok=True)
    verification_json = json.dumps(verification, ensure_ascii=False, indent=2)
    (verification_dir / "final_verification.json").write_text(
        verification_json, encoding="utf-8"
    )
    (sequence_verification_dir / "stochastic_verification.json").write_text(
        verification_json, encoding="utf-8"
    )
    (sequence_verification_dir / "fixed_artifact_verification.json").write_text(
        json.dumps(fixed_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "trial",
        "seeds",
        "true_decoder",
        "selected_attack",
        "selected_attack_value",
        "sequence",
        "bit_error_rate",
        "bit_acc",
        "psnr",
        "lpips",
        "attacked_image",
    ]
    save_csv(verification_dir / "final_verification.csv", rows, fields)
    save_csv(sequence_verification_dir / "trial_results.csv", rows, fields)
    decision_dir = output_dir / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "final_result.json").write_text(
        json.dumps(
            {"decision": search_result["best_payload"], "verification": verification},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Final sequential true-decoder verification:")
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)
    return verification


def run_one_evaluation(
    args: argparse.Namespace,
    scales: list[float],
    jpeg_qualities: list[int],
    vae_qualities: list[int],
) -> None:
    """Run one image while retaining process-wide model caches for later rows."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods, method_selection = select_eval_methods(args, output_dir)
    if method_selection.get("ranking_csv"):
        # Keep the ranking path in result metadata for in-process adversarial actions.
        args.ranking_csv = method_selection["ranking_csv"]
    standard_methods = list(methods)
    if not standard_methods:
        raise RuntimeError("TPS/JPEG/VAE require at least one entropy-ranked candidate decoder.")
    # Keep the full entropy-top-k pool for candidate evaluation/selection, but
    # optimize the adversarial perturbation against only the top-ranked proxy.
    # This separates focused attack gradients from the top-k validation signal.
    adversarial_methods = list(methods[:1])
    print(f"TPS/JPEG/VAE decoders: {', '.join(standard_methods)}")
    if args.enable_adversarial_attack:
        print(f"Adversarial decoders: {', '.join(adversarial_methods)}")

    watermarked, original_size = load_image_tensor(args.watermarked_image, args.device, args.input_size)
    active_methods = list(dict.fromkeys(
        standard_methods
        + (adversarial_methods if args.enable_adversarial_attack else [])
    ))
    decoders_by_method = {}
    for method in active_methods:
        decoder = get_cached_decoder(method, args)
        decoder.cache_clean_bits(watermarked)
        decoders_by_method[canonical_method(method)] = decoder

    standard_decoders = [decoders_by_method[canonical_method(method)] for method in standard_methods]
    adversarial_decoders = (
        [decoders_by_method[canonical_method(method)] for method in adversarial_methods]
        if args.enable_adversarial_attack
        else standard_decoders
    )
    method_weights = method_selection["decoder_importance_weights"]
    if getattr(args, "decoder_weights", None):
        parsed_standard_weights = parse_decoder_weights(args.decoder_weights)
        standard_method_weights = normalize_decoder_weights(
            {
                canonical_method(method): parsed_standard_weights[canonical_method(method)]
                for method in standard_methods
            }
        )
        standard_weight_source = "command_line_subset"
    elif method_selection.get("candidate_weighting") == "uniform":
        standard_method_weights = normalize_decoder_weights(
            {canonical_method(method): 1.0 for method in standard_methods}
        )
        standard_weight_source = "candidate_uniform"
    elif method_selection.get("ranking_csv"):
        standard_method_weights = read_decoder_weights_from_ranking_csv(
            Path(method_selection["ranking_csv"]), standard_methods
        )
        standard_weight_source = "entropy_self_calibrated_score"
    else:
        standard_method_weights = normalize_decoder_weights(
            {canonical_method(method): 1.0 for method in standard_methods}
        )
        standard_weight_source = "uniform_default"
    standard_decoder_weights = normalize_decoder_weights(
        {
            decoder.name: standard_method_weights[canonical_method(method)]
            for method, decoder in zip(standard_methods, standard_decoders)
        }
    )
    adversarial_decoder_weights = (
        normalize_decoder_weights(
            {
                decoder.name: method_weights[canonical_method(method)]
                for method, decoder in zip(adversarial_methods, adversarial_decoders)
            }
        )
        if args.enable_adversarial_attack
        else standard_decoder_weights
    )
    standard_method_selection = copy.deepcopy(method_selection)
    standard_method_selection.update(
        {
            "methods": standard_methods,
            "decoder_scope": "topk_validation",
            "decoder_importance_weights": standard_decoder_weights,
            "evaluated_decoder_weights": standard_decoder_weights,
            "weight_source": standard_weight_source,
        }
    )
    adversarial_method_selection = copy.deepcopy(method_selection)
    adversarial_method_selection.update(
        {
            "methods": adversarial_methods,
            "decoder_scope": "top1_adversarial_attack",
            "evaluated_decoder_weights": adversarial_decoder_weights,
        }
    )
    print(
        "TPS/JPEG/VAE decoder weights: "
        + ", ".join(f"{decoder}={weight:.4f}" for decoder, weight in standard_decoder_weights.items())
    )
    if args.enable_adversarial_attack:
        print(
            "Adversarial decoder weights: "
            + ", ".join(
                f"{decoder}={weight:.4f}" for decoder, weight in adversarial_decoder_weights.items()
            )
        )

    if getattr(args, "enable_sequential_search", False):
        args.active_method_selection = standard_method_selection
        search_result = run_sequential_search(
            args,
            output_dir,
            watermarked,
            standard_decoders,
            standard_decoder_weights,
            adversarial_decoders,
            adversarial_decoder_weights,
        )
        if args.true_decoder:
            run_sequential_verification(
                args,
                output_dir,
                watermarked,
                search_result,
                adversarial_decoders,
                adversarial_decoder_weights,
            )
        return

    sweep_specs = (
        ("tps", scales, "scale", "TPS scale", "tps_scale"),
        ("jpeg", jpeg_qualities, "quality", "JPEG quality", "jpeg_quality"),
        ("vae", vae_qualities, "vae_quality", "VAE quality", "vae_quality"),
    )
    attack_payloads = [
        run_attack_sweep(
            args=args,
            attack_name=attack_name,
            values=values,
            value_key=value_key,
            value_label=value_label,
            filename_suffix=filename_suffix,
            decoders=standard_decoders,
            watermarked=watermarked,
            output_dir=output_dir / attack_name,
            method_selection=standard_method_selection,
            original_size=original_size,
            input_size=args.input_size,
        )
        for attack_name, values, value_key, value_label, filename_suffix in sweep_specs
    ]
    if args.enable_adversarial_attack:
        attack_payloads.append(
            run_adversarial_attack(
                args=args,
                decoders=adversarial_decoders,
                decoder_weights=adversarial_decoder_weights,
                evaluation_decoders=standard_decoders,
                evaluation_decoder_weights=standard_decoder_weights,
                attack_method_selection=adversarial_method_selection,
                watermarked=watermarked,
                output_dir=output_dir / "adversarial",
                method_selection=standard_method_selection,
                original_size=original_size,
                input_size=args.input_size,
            )
        )

    if args.enable_decision:
        decoder_weights_by_attack = {
            "tps": standard_decoder_weights,
            "jpeg": standard_decoder_weights,
            "vae": standard_decoder_weights,
        }
        if args.enable_adversarial_attack:
            # Decision/validation uses the full top-k proxy pool even though
            # the adversarial image was optimized against only top-1.
            decoder_weights_by_attack["adversarial"] = standard_decoder_weights
        recommended_selection = run_decision_module(
            args,
            output_dir,
            attack_payloads,
            standard_decoder_weights,
            decoder_weights_by_attack=decoder_weights_by_attack,
        )
        if args.true_decoder:
            run_final_verification(args, output_dir, watermarked, recommended_selection)


def read_batch_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"image_path", "output_dir"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Batch manifest is missing columns: {', '.join(sorted(missing))}")
    return rows


def prepare_batch_entropy_rankings(
    args: argparse.Namespace,
    jobs: list[dict[str, str]],
) -> dict[str, Path]:
    """Rank all batch images method-major, loading each classifier only once."""
    if args.eval_method or args.ranking_csv or not jobs:
        return {}

    import decoder_entropy_ranking as entropy

    entropy_args = copy.copy(args)
    entropy_args.method = None
    entropy_args.skip_method = list(args.skip_entropy_method or [])
    methods = entropy.selected_methods(entropy_args)
    results_by_job: dict[str, list[dict[str, Any]]] = {
        job["output_dir"]: [] for job in jobs
    }
    original_sizes: dict[str, Any] = {}
    reference_roots: dict[str, Path] = {}

    if args.skip_reference_encoder:
        for job in jobs:
            ranking_dir = Path(job["output_dir"]) / "entropy_ranking"
            reference_root = ranking_dir / "reference_encodings"
            reference_root.mkdir(parents=True, exist_ok=True)
            job_args = copy.copy(entropy_args)
            job_args.watermarked_image = job["image_path"]
            entropy.run_reference_encoders(job_args, reference_root)
            reference_roots[job["output_dir"]] = reference_root

    for method_index, method in enumerate(methods, start=1):
        print(
            f"[batch entropy model {method_index}/{len(methods)}] loading method={method}",
            flush=True,
        )
        decoder = None
        encoder = None
        build_error = None
        try:
            decoder = build_decoder(method, entropy_args)
            if not args.skip_reference_encoder:
                encoder = entropy.atk.build_encoder(method, entropy_args)
        except Exception as exc:
            build_error = exc

        for job_index, job in enumerate(jobs, start=1):
            key = job["output_dir"]
            true_decoder = canonical_method(job.get("true_decoder", ""))
            method_is_true_decoder = method == true_decoder or (
                method in FIN_FAMILY and true_decoder in FIN_FAMILY
            )
            if (
                args.batch_exclude_true_decoder
                and method_is_true_decoder
            ):
                continue
            if build_error is not None:
                result = {
                    "method": method,
                    "status": "error",
                    "error_type": type(build_error).__name__,
                    "error": str(build_error),
                }
            else:
                image, original_size = load_image_tensor(
                    job["image_path"], args.device, args.input_size
                )
                original_sizes[key] = original_size
                result = entropy.evaluate_method(
                    method,
                    image,
                    entropy_args,
                    reference_root=reference_roots.get(key),
                    encoder=encoder,
                    decoder=decoder,
                )
                del image
            results_by_job[key].append(result)
            print(
                f"[batch entropy image {job_index}/{len(jobs)}] method={method} "
                f"status={result.get('status')} image={job['image_path']}",
                flush=True,
            )
        del decoder, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranking_paths: dict[str, Path] = {}
    for job in jobs:
        key = job["output_dir"]
        ranking_dir = Path(key) / "entropy_ranking"
        ranking_dir.mkdir(parents=True, exist_ok=True)
        ranked = entropy.rank_entropy_results(results_by_job[key])
        csv_path = ranking_dir / "entropy_ranking.csv"
        json_path = ranking_dir / "entropy_ranking.json"
        entropy.write_csv(csv_path, ranked)
        summary = {
            "input_image": job["image_path"],
            "original_size": list(original_sizes.get(key, ())),
            "input_size": args.input_size,
            "device": str(args.device),
            "methods": methods,
            "batch_model_reuse": True,
            "ranking_rule": (
                "descending self_calibrated_score; "
                "self_calibrated_score=(1-entropy_mean_bits)/(1-reference_entropy_mean_bits+eps)"
            ),
            "rankings": ranked,
        }
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        ranking_paths[key] = csv_path
    return ranking_paths


def run_batch_evaluations(
    args: argparse.Namespace,
    scales: list[float],
    jpeg_qualities: list[int],
    vae_qualities: list[int],
) -> int:
    rows = read_batch_manifest(Path(args.batch_manifest))
    jobs = []
    for row in rows:
        output_dir = Path(row["output_dir"])
        verification_path = output_dir / "verification" / "final_verification.json"
        failed_marker = output_dir / ".failed"
        if verification_path.is_file():
            try:
                verification = json.loads(
                    verification_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                verification = {}
            if verification.get("vae_protocol_version") == VAE_PROTOCOL_VERSION:
                print(f"[batch] skip completed {row['image_path']}", flush=True)
                continue
            print(
                f"[batch] rerun stale VAE protocol {row['image_path']} "
                f"(found={verification.get('vae_protocol_version')!r}, "
                f"expected={VAE_PROTOCOL_VERSION!r})",
                flush=True,
            )
        if failed_marker.is_file() and not args.batch_rerun_failed:
            print(f"[batch] skip previous failure {row['image_path']}", flush=True)
            continue
        jobs.append(row)

    print(f"[batch] pending images={len(jobs)} model reuse enabled", flush=True)
    ranking_paths = prepare_batch_entropy_rankings(args, jobs)
    failures = 0
    for index, row in enumerate(jobs, start=1):
        job_args = copy.copy(args)
        job_args.batch_manifest = None
        job_args.watermarked_image = row["image_path"]
        job_args.output_dir = row["output_dir"]
        if row.get("true_decoder"):
            job_args.true_decoder = row["true_decoder"]
        if args.batch_exclude_true_decoder:
            true_decoder = canonical_method(job_args.true_decoder)
            job_args.exclude_method = list(job_args.exclude_method or [])
            true_decoder_family = sorted(FIN_FAMILY) if true_decoder in FIN_FAMILY else [true_decoder]
            for method in true_decoder_family:
                if method not in job_args.exclude_method:
                    job_args.exclude_method.append(method)
        if ranking_paths:
            job_args.ranking_csv = str(ranking_paths[row["output_dir"]])

        output_dir = Path(job_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        failed_marker = output_dir / ".failed"
        print(
            f"[batch {index}/{len(jobs)}] image={job_args.watermarked_image} "
            f"output={job_args.output_dir}",
            flush=True,
        )
        try:
            set_seed(job_args.seed)
            run_one_evaluation(job_args, scales, jpeg_qualities, vae_qualities)
            failed_marker.unlink(missing_ok=True)
        except Exception as exc:
            failures += 1
            failed_marker.write_text(
                f"error_type={type(exc).__name__}\nerror={exc}\n", encoding="utf-8"
            )
            traceback.print_exc()
            print(f"[batch {index}/{len(jobs)}] FAILED: {exc}", file=sys.stderr, flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[batch] completed={len(jobs) - failures} failed={failures}", flush=True)
    return 1 if failures else 0


def main() -> int:
    configure_torch_cache()
    args = build_argparser().parse_args()
    scales, jpeg_qualities, vae_qualities = validate_args(args)
    args.device = torch.device(args.device)
    set_seed(args.seed)
    if args.batch_manifest:
        return run_batch_evaluations(args, scales, jpeg_qualities, vae_qualities)
    run_one_evaluation(args, scales, jpeg_qualities, vae_qualities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
