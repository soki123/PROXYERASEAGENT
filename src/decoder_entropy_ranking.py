import csv
import os
import json
import subprocess
from pathlib import Path

import torch

import attack_topk_decoders as atk


DEFAULT_METHODS = [
    "mbrs",
    "hidden",
    "pimog",
    "stegastamp",
    "cin",
    "fin_heavy",
    "fin_jpeg",
    "invismark",
    "rosteals",
    "trustmark",
    "lightweightmark",
    "videoseal",
    "chunkyseal",
]
ENTROPY_EPS = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ENCODER_SCRIPT = Path(
    os.environ.get(
        "REFERENCE_ENCODER_SCRIPT",
        str(PROJECT_ROOT / "third_party" / "encoder_classifi" / "data" / "run_all_infer.sh"),
    )
)
REFERENCE_METHOD_DIRS = {
    "cin": "CIN",
    "fin": "FIN",
    "fin_heavy": "FIN_HEAVY",
    "fin_jpeg": "FIN_JPEG",
    "hidden": "HiDDeN",
    "invismark": "InvisMark",
    "mbrs": "MBRS",
    "pimog": "PIMoG",
    "stegastamp": "StegaStamp",
    "rosteals": "RoSteALS",
    "trustmark": "trustmark",
    "lightweightmark": "LightweightMark",
    "videoseal": "VideoSeal",
    "chunkyseal": "ChunkySeal",
}


def build_argparser():
    parser = atk.build_argparser()
    parser.description = (
        "Rank watermark decoder candidates by output information entropy on one image. "
        "Lower entropy means the decoder is more confident before hard bit thresholding."
    )
    parser.set_defaults(
        watermarked_image=os.environ.get("DEFAULT_WATERMARKED_IMAGE", ""),
        output_dir=os.environ.get(
            "DEFAULT_ENTROPY_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "entropy_ranking")
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            "Decoder method to evaluate. Can be repeated. "
            f"Defaults to: {','.join(DEFAULT_METHODS)}"
        ),
    )
    parser.add_argument(
        "--skip-method",
        action="append",
        default=None,
        help="Decoder method to skip. Can be repeated.",
    )
    parser.add_argument(
        "--json-name",
        default="entropy_ranking.json",
        help="Output JSON filename under --output-dir.",
    )
    parser.add_argument(
        "--csv-name",
        default="entropy_ranking.csv",
        help="Output CSV filename under --output-dir.",
    )
    parser.add_argument(
        "--reference-output-root",
        default=None,
        help=(
            "Directory where data/run_all_infer.sh writes per-method reference encodings. "
            "Defaults to <output-dir>/reference_encodings."
        ),
    )
    parser.add_argument(
        "--skip-reference-encoder",
        action="store_true",
        help=(
            "Do not build in-process encoder wrappers; use existing "
            "--reference-output-root encoded.png files instead."
        ),
    )
    return parser


def binary_entropy(probabilities, eps=ENTROPY_EPS):
    original_dtype = probabilities.dtype
    probabilities = probabilities.double().clamp(eps, 1.0 - eps)
    entropy = -(
        probabilities * torch.log2(probabilities)
        + (1.0 - probabilities) * torch.log2(1.0 - probabilities)
    )
    return entropy.to(dtype=original_dtype)


def entropy_metrics_from_probabilities(probabilities):
    probabilities = probabilities.detach().float().flatten()
    entropy = binary_entropy(probabilities)
    margin = (probabilities - 0.5).abs()
    confidence = 0.5 + margin
    entropy_quantiles = torch.quantile(
        entropy.cpu(),
        torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=entropy.dtype),
    )
    confidence_quantiles = torch.quantile(
        confidence.cpu(),
        torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=confidence.dtype),
    )
    return {
        "decoded_count": int(probabilities.numel()),
        "entropy_mean_bits": float(entropy.mean().item()),
        "entropy_min_bits": float(entropy.min().item()),
        "entropy_q05_bits": float(entropy_quantiles[1].item()),
        "entropy_q25_bits": float(entropy_quantiles[2].item()),
        "entropy_q50_bits": float(entropy_quantiles[3].item()),
        "entropy_q75_bits": float(entropy_quantiles[4].item()),
        "entropy_q95_bits": float(entropy_quantiles[5].item()),
        "entropy_max_bits": float(entropy.max().item()),
        "confidence_mean": float(confidence.mean().item()),
        "confidence_min": float(confidence.min().item()),
        "confidence_q05": float(confidence_quantiles[1].item()),
        "confidence_q25": float(confidence_quantiles[2].item()),
        "confidence_q50": float(confidence_quantiles[3].item()),
        "confidence_q75": float(confidence_quantiles[4].item()),
        "confidence_q95": float(confidence_quantiles[5].item()),
        "confidence_max": float(confidence.max().item()),
        "probability_mean": float(probabilities.mean().item()),
        "probability_min": float(probabilities.min().item()),
        "probability_max": float(probabilities.max().item()),
        "bit_one_fraction": float((probabilities > 0.5).float().mean().item()),
        "low_entropy_fraction_le_0_25": float((entropy <= 0.25).float().mean().item()),
        "low_entropy_fraction_le_0_50": float((entropy <= 0.50).float().mean().item()),
        "low_entropy_fraction_le_0_75": float((entropy <= 0.75).float().mean().item()),
        "high_conf_fraction_ge_0_60": float((confidence >= 0.60).float().mean().item()),
        "high_conf_fraction_ge_0_70": float((confidence >= 0.70).float().mean().item()),
        "high_conf_fraction_ge_0_80": float((confidence >= 0.80).float().mean().item()),
        "high_conf_fraction_ge_0_90": float((confidence >= 0.90).float().mean().item()),
    }


def attach_self_calibrated_score(result, eps=ENTROPY_EPS):
    current_entropy = float(result["entropy_mean_bits"])
    reference_entropy = float(result.get("reference_entropy_mean_bits", current_entropy))
    current_response = 1.0 - current_entropy
    reference_response = 1.0 - reference_entropy
    result["current_entropy_response"] = current_response
    result["reference_entropy_response"] = reference_response
    result["self_calibrated_score"] = current_response / (reference_response + eps)
    return result


def rank_entropy_results(results):
    successful = [item for item in results if item.get("status") == "ok"]
    failed = [item for item in results if item.get("status") != "ok"]
    for item in successful:
        attach_self_calibrated_score(item)
    successful.sort(key=lambda item: item["self_calibrated_score"], reverse=True)
    for rank, item in enumerate(successful, start=1):
        item["rank"] = rank
    for item in failed:
        item["rank"] = ""
    return successful + failed


def selected_methods(args):
    methods = args.method if args.method else DEFAULT_METHODS
    skipped = {atk.canonical_method(method) for method in (args.skip_method or [])}
    selected = []
    seen = set()
    for method in methods:
        canonical = atk.canonical_method(method)
        if canonical in skipped or canonical in seen:
            continue
        selected.append(canonical)
        seen.add(canonical)
    return selected


def reference_encoded_path_for_method(method, reference_root):
    canonical = atk.canonical_method(method)
    dirname = REFERENCE_METHOD_DIRS.get(canonical)
    if dirname is None:
        raise ValueError(f"No reference encoder output mapping for method: {method}")
    return Path(reference_root) / dirname / "encoded.png"


def run_reference_encoders(args, reference_root):
    if args.skip_reference_encoder:
        return
    if not REFERENCE_ENCODER_SCRIPT.exists():
        raise FileNotFoundError(f"Reference encoder script not found: {REFERENCE_ENCODER_SCRIPT}")
    subprocess.run(
        [
            "bash",
            str(REFERENCE_ENCODER_SCRIPT),
            str(args.watermarked_image),
            str(reference_root),
        ],
        check=True,
    )


def prefixed_metrics(metrics, prefix):
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key.startswith("entropy_") or key.startswith("confidence_")
    }


def evaluate_method(method, image, args, reference_root=None, encoder=None, decoder=None):
    entry = {"method": method}
    try:
        decoder = decoder if decoder is not None else atk.build_decoder(method, args)
        with torch.no_grad():
            decoded = decoder.decode(image)
            probabilities = decoder.probability_values(decoded.detach())
            metrics = entropy_metrics_from_probabilities(probabilities)
            reference_metrics = prefixed_metrics(metrics, "reference")
            reference_path = None
            reference_source = "current_image"
            if encoder is not None:
                reference_image = encoder.encode(image)
                reference_decoded = decoder.decode(reference_image)
                reference_probabilities = decoder.probability_values(reference_decoded.detach())
                reference_metrics = prefixed_metrics(
                    entropy_metrics_from_probabilities(reference_probabilities),
                    "reference",
                )
                reference_source = "encoder_wrapper"
            elif reference_root is not None:
                reference_path = reference_encoded_path_for_method(method, reference_root)
                if not reference_path.exists():
                    raise FileNotFoundError(f"Reference encoded image not found: {reference_path}")
                reference_image, _ = atk.load_image_tensor(reference_path, args.device, args.input_size)
                reference_decoded = decoder.decode(reference_image)
                reference_probabilities = decoder.probability_values(reference_decoded.detach())
                reference_metrics = prefixed_metrics(
                    entropy_metrics_from_probabilities(reference_probabilities),
                    "reference",
                )
                reference_source = "reference_file"
            bits = decoder.bits(decoded).detach().float().flatten()
            raw = decoded.detach().float().flatten()
        entry.update(
            {
                "status": "ok",
                "decoder_name": decoder.name,
                "decoded_shape": list(decoded.shape),
                "wrapper_bit_one_fraction": float(bits.mean().item()),
                "raw_decoded_mean": float(raw.mean().item()),
                "raw_decoded_min": float(raw.min().item()),
                "raw_decoded_max": float(raw.max().item()),
                "reference_encoded_path": str(reference_path) if reference_path is not None else "",
                "reference_source": reference_source,
            }
        )
        entry.update(metrics)
        entry.update(reference_metrics)
        attach_self_calibrated_score(entry)
    except Exception as exc:
        entry.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return entry


def evaluate_methods(methods, image, args, reference_root=None):
      results = []
      total = len(methods)
      for index, method in enumerate(methods, start=1):
          print(f"[entropy {index}/{total}] start method={method}", flush=True)
          if getattr(args, "skip_reference_encoder", False):
              print(f"[entropy {index}/{total}] decode only with reference_root={reference_root}", flush=True)
              results.append(evaluate_method(method, image, args, reference_root=reference_root))
              print(f"[entropy {index}/{total}] done method={method} status={results[-1].get('status')}", flush=True)
              continue

          try:
              print(f"[entropy {index}/{total}] building decoder method={method}", flush=True)
              decoder = atk.build_decoder(method, args)

              print(f"[entropy {index}/{total}] building encoder method={method}", flush=True)
              encoder = atk.build_encoder(method, args)

              print(f"[entropy {index}/{total}] evaluating method={method}", flush=True)
          except Exception as exc:
              print(f"[entropy {index}/{total}] build failed method={method}: {type(exc).__name__}: {exc}", flush=True)
              results.append(
                  {
                      "method": method,
                      "status": "error",
                      "error_type": type(exc).__name__,
                      "error": str(exc),
                  }
              )
              continue

          results.append(evaluate_method(method, image, args, encoder=encoder, decoder=decoder))
          print(f"[entropy {index}/{total}] done method={method} status={results[-1].get('status')}", flush=True)

      return results


def write_csv(path, results):
    fieldnames = [
        "rank",
        "method",
        "status",
        "decoder_name",
        "decoded_shape",
        "decoded_count",
        "entropy_mean_bits",
        "reference_entropy_mean_bits",
        "current_entropy_response",
        "reference_entropy_response",
        "self_calibrated_score",
        "reference_encoded_path",
        "reference_source",
        "entropy_min_bits",
        "entropy_q05_bits",
        "entropy_q25_bits",
        "entropy_q50_bits",
        "entropy_q75_bits",
        "entropy_q95_bits",
        "entropy_max_bits",
        "confidence_mean",
        "reference_confidence_mean",
        "confidence_min",
        "confidence_q05",
        "confidence_q25",
        "confidence_q50",
        "confidence_q75",
        "confidence_q95",
        "confidence_max",
        "low_entropy_fraction_le_0_25",
        "low_entropy_fraction_le_0_50",
        "low_entropy_fraction_le_0_75",
        "high_conf_fraction_ge_0_60",
        "high_conf_fraction_ge_0_70",
        "high_conf_fraction_ge_0_80",
        "high_conf_fraction_ge_0_90",
        "probability_mean",
        "probability_min",
        "probability_max",
        "bit_one_fraction",
        "wrapper_bit_one_fraction",
        "raw_decoded_mean",
        "raw_decoded_min",
        "raw_decoded_max",
        "error_type",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {field: item.get(field, "") for field in fieldnames}
            if isinstance(row.get("decoded_shape"), list):
                row["decoded_shape"] = "x".join(str(value) for value in row["decoded_shape"])
            writer.writerow(row)


def main():
    args = build_argparser().parse_args()
    args.device = torch.device(args.device)
    atk.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cin_temp").mkdir(parents=True, exist_ok=True)

    image, original_size = atk.load_image_tensor(args.watermarked_image, args.device, args.input_size)
    reference_root = Path(args.reference_output_root) if args.reference_output_root else output_dir / "reference_encodings"
    if args.skip_reference_encoder:
        reference_root.mkdir(parents=True, exist_ok=True)
        run_reference_encoders(args, reference_root)
    results = evaluate_methods(selected_methods(args), image, args, reference_root)
    ranked = rank_entropy_results(results)

    summary = {
        "input_image": args.watermarked_image,
        "original_size": list(original_size),
        "input_size": args.input_size,
        "device": str(args.device),
        "methods": selected_methods(args),
        "reference_encoder": "python_wrapper" if not args.skip_reference_encoder else str(REFERENCE_ENCODER_SCRIPT),
        "reference_output_root": str(reference_root) if args.skip_reference_encoder else "",
        "skip_reference_encoder": args.skip_reference_encoder,
        "ranking_rule": (
            "descending self_calibrated_score; "
            "self_calibrated_score=(1-entropy_mean_bits)/(1-reference_entropy_mean_bits+eps)"
        ),
        "metrics_definition": {
            "probability": "decoder output mapped to [0,1] by attack_topk_decoders.DecoderWrapper.probability_values()",
            "binary_entropy_bits": "-p*log2(p) - (1-p)*log2(1-p), averaged over decoded positions",
            "confidence": "max(p, 1-p) = 0.5 + abs(p - 0.5)",
            "self_calibrated_score": "(1-current entropy_mean_bits)/(1-reference_entropy_mean_bits+eps)",
        },
        "rankings": ranked,
    }

    json_path = output_dir / args.json_name
    csv_path = output_dir / args.csv_name
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, ranked)

    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")
    for item in ranked:
        if item["status"] == "ok":
            print(
                f"rank={item['rank']} method={item['method']} "
                f"entropy_mean_bits={item['entropy_mean_bits']:.6f} "
                f"reference_entropy_mean_bits={item['reference_entropy_mean_bits']:.6f} "
                f"self_calibrated_score={item['self_calibrated_score']:.6f} "
                f"confidence_mean={item['confidence_mean']:.6f} "
                f"decoded_shape={item['decoded_shape']}"
            )
        else:
            print(
                f"rank= method={item['method']} status=error "
                f"{item['error_type']}: {item['error']}"
            )


if __name__ == "__main__":
    main()
