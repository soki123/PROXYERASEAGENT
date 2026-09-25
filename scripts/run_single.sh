#!/usr/bin/env bash
set -euo pipefail

# Portable single-image entry point. All paths are configurable through
# environment variables; no machine-specific paths are embedded in the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PROJECT_ROOT/src/eval_tps_decoder_curve.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export TORCH_HOME
mkdir -p "$TORCH_HOME/hub/checkpoints"

WATERMARKED_IMAGE="${WATERMARKED_IMAGE:?Set WATERMARKED_IMAGE to an encoded input image}"
TRUE_DECODER="${TRUE_DECODER:?Set TRUE_DECODER to the encoder family name}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/single}"
DEVICE="${DEVICE:-cuda}"

# Candidate selection
EXCLUDED_DECODERS=("stegastamp")
EVAL_METHODS=()
EXCLUDE_TRUE_DECODER_FROM_CANDIDATES="${EXCLUDE_TRUE_DECODER_FROM_CANDIDATES:-true}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-3}"
CANDIDATE_SELECTION="${CANDIDATE_SELECTION:-entropy_topk}"
CANDIDATE_SELECTION_SEED="${CANDIDATE_SELECTION_SEED:-0}"
CANDIDATE_WEIGHTING="${CANDIDATE_WEIGHTING:-auto}"
RANDOM_CANDIDATE_WEIGHTING="${RANDOM_CANDIDATE_WEIGHTING:-uniform}"

# Attack protocol
VERIFICATION_TRIALS="${VERIFICATION_TRIALS:-5}"
SCALES="${SCALES:-0.001,0.002,0.004,0.008,0.01,0.02,0.04}"
JPEG_QUALITIES="${JPEG_QUALITIES:-80,70,60,50,40,30}"
VAE_QUALITIES="${VAE_QUALITIES:-6,5,4,3,2,1}"
VAE_MODEL_NAME="${VAE_MODEL_NAME:-cheng2020-anchor}"
VAE_INPUT_SIZE="${VAE_INPUT_SIZE:-512}"
TRIALS="${TRIALS:-5}"

# The public profile uses only the low-frequency UnMarker and image-quality
# objectives. Decoder bit loss and reference-residual frequency loss are not
# part of this release.
ENABLE_ADVERSARIAL_ATTACK="${ENABLE_ADVERSARIAL_ATTACK:-true}"
ADVERSARIAL_GRADNORM_QUALITY_SCALE="${ADVERSARIAL_GRADNORM_QUALITY_SCALE:-180}"
ADVERSARIAL_STEPS="${ADVERSARIAL_STEPS:-300}"
ADVERSARIAL_LR="${ADVERSARIAL_LR:-0.1}"
ADVERSARIAL_MAX_PERTURBATION="${ADVERSARIAL_MAX_PERTURBATION:-0}"
ADVERSARIAL_WATERMARK_WEIGHT="${ADVERSARIAL_WATERMARK_WEIGHT:-10}"
ADVERSARIAL_QUALITY_CONSTRAINT="${ADVERSARIAL_QUALITY_CONSTRAINT:-true}"
ADVERSARIAL_TARGET_LPIPS="${ADVERSARIAL_TARGET_LPIPS:-0.05}"
ADVERSARIAL_LPIPS_NET="${ADVERSARIAL_LPIPS_NET:-alex}"
ADVERSARIAL_SELECTION_MIN_STEP="${ADVERSARIAL_SELECTION_MIN_STEP:-0}"
ADVERSARIAL_SELECTION_MIN_PSNR="${ADVERSARIAL_SELECTION_MIN_PSNR:-0}"

# Decision and sequential search
DECISION_SUCCESS_BER="${DECISION_SUCCESS_BER:-0.1}"
DECISION_BER_WEIGHT="${DECISION_BER_WEIGHT:-0.5}"
DECISION_QUALITY_METRIC="${DECISION_QUALITY_METRIC:-psnr}"
ENABLE_SEQUENTIAL_SEARCH="${ENABLE_SEQUENTIAL_SEARCH:-true}"
SEQUENTIAL_MAX_STEPS="${SEQUENTIAL_MAX_STEPS:-4}"
SEQUENTIAL_SEARCH_STRATEGY="${SEQUENTIAL_SEARCH_STRATEGY:-pareto_beam}"
SEQUENTIAL_BEAM_WIDTH="${SEQUENTIAL_BEAM_WIDTH:-3}"
SEQUENTIAL_TPS_SCALES="${SEQUENTIAL_TPS_SCALES:-0.00025,0.0005,0.001,0.002}"
SEQUENTIAL_JPEG_QUALITIES="${SEQUENTIAL_JPEG_QUALITIES:-90,80,70,60,50,40,30}"
SEQUENTIAL_VAE_QUALITIES="${SEQUENTIAL_VAE_QUALITIES:-6,5,4,3,2,1}"
SEQUENTIAL_ADVERSARIAL_STEPS="${SEQUENTIAL_ADVERSARIAL_STEPS:-50,100,200}"
SEQUENTIAL_TPS_SEARCH_TRIALS="${SEQUENTIAL_TPS_SEARCH_TRIALS:-3}"
SEQUENTIAL_VERIFICATION_TRIALS="${SEQUENTIAL_VERIFICATION_TRIALS:-20}"
SEQUENTIAL_REPLAY_VERIFICATION="${SEQUENTIAL_REPLAY_VERIFICATION:-true}"
SEQUENTIAL_LAMBDA_FAMILY="${SEQUENTIAL_LAMBDA_FAMILY:-0.5}"
SEQUENTIAL_LAMBDA_BER="${SEQUENTIAL_LAMBDA_BER:-1.0}"
SEQUENTIAL_LAMBDA_LPIPS="${SEQUENTIAL_LAMBDA_LPIPS:-0.5}"
SEQUENTIAL_SELECTION_LPIPS_LIMIT="${SEQUENTIAL_SELECTION_LPIPS_LIMIT:-0.05}"
SEQUENTIAL_BER_TARGET="${SEQUENTIAL_BER_TARGET:-0.1}"
SEQUENTIAL_ENABLE_CONVERGENCE="${SEQUENTIAL_ENABLE_CONVERGENCE:-false}"
SEQUENTIAL_CONVERGENCE_PATIENCE="${SEQUENTIAL_CONVERGENCE_PATIENCE:-2}"
SEQUENTIAL_MIN_RELATIVE_HV_GAIN="${SEQUENTIAL_MIN_RELATIVE_HV_GAIN:-0.001}"
SEQUENTIAL_MIN_BER_GAIN="${SEQUENTIAL_MIN_BER_GAIN:-0.002}"
SEQUENTIAL_ADVERSARIAL_CONVERGENCE_PATIENCE="${SEQUENTIAL_ADVERSARIAL_CONVERGENCE_PATIENCE:-15}"
SEQUENTIAL_ADVERSARIAL_CONVERGENCE_MIN_DELTA="${SEQUENTIAL_ADVERSARIAL_CONVERGENCE_MIN_DELTA:-0.0001}"
SEQUENTIAL_ADVERSARIAL_CONVERGENCE_WARMUP="${SEQUENTIAL_ADVERSARIAL_CONVERGENCE_WARMUP:-10}"

canonical_method() {
    local value
    value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "$value" in
        fin-heavy|finheavy) printf '%s\n' "fin_heavy" ;;
        fin-jpeg|finjpeg) printf '%s\n' "fin_jpeg" ;;
        lightweight_mark|lightweight) printf '%s\n' "lightweightmark" ;;
        video_seal) printf '%s\n' "videoseal" ;;
        chunky_seal|cunkyseal) printf '%s\n' "chunkyseal" ;;
        *) printf '%s\n' "$value" ;;
    esac
}

decoder_is_fin_family() {
    case "$(canonical_method "$1")" in
        fin|fin_heavy|fin_jpeg) return 0 ;;
        *) return 1 ;;
    esac
}

decoders_match_for_exclusion() {
    local left right
    left="$(canonical_method "$1")"
    right="$(canonical_method "$2")"
    [[ "$left" == "$right" ]] || { decoder_is_fin_family "$left" && decoder_is_fin_family "$right"; }
}

append_entropy_exclusion() {
    local method="$1"
    if decoder_is_fin_family "$method"; then
        CMD+=(--skip-entropy-method fin --skip-entropy-method fin_heavy --skip-entropy-method fin_jpeg
              --exclude-method fin --exclude-method fin_heavy --exclude-method fin_jpeg)
    else
        CMD+=(--skip-entropy-method "$method" --exclude-method "$method")
    fi
}

case "${EXCLUDE_TRUE_DECODER_FROM_CANDIDATES,,}" in
    true|1|yes) EXCLUDE_TRUE_DECODER_FROM_CANDIDATES=true ;;
    false|0|no) EXCLUDE_TRUE_DECODER_FROM_CANDIDATES=false ;;
    *) echo "EXCLUDE_TRUE_DECODER_FROM_CANDIDATES must be true or false." >&2; exit 2 ;;
esac

is_excluded_decoder() {
    local method="$1" excluded
    method="$(canonical_method "$method")"
    if [[ "$EXCLUDE_TRUE_DECODER_FROM_CANDIDATES" == true ]] && decoders_match_for_exclusion "$method" "$TRUE_DECODER"; then
        return 0
    fi
    for excluded in "${EXCLUDED_DECODERS[@]}"; do
        decoders_match_for_exclusion "$method" "$excluded" && return 0
    done
    return 1
}

CMD=(
    "$PYTHON_BIN" "$PYTHON_SCRIPT"
    --watermarked-image "$WATERMARKED_IMAGE"
    --true-decoder "$TRUE_DECODER"
    --verification-trials "$VERIFICATION_TRIALS"
    --scales "$SCALES"
    --jpeg-qualities "$JPEG_QUALITIES"
    --vae-qualities "$VAE_QUALITIES"
    --vae-model-name "$VAE_MODEL_NAME"
    --vae-input-size "$VAE_INPUT_SIZE"
    --trials "$TRIALS"
    --adversarial-gradnorm-quality-scale "$ADVERSARIAL_GRADNORM_QUALITY_SCALE"
    --adversarial-max-perturbation "$ADVERSARIAL_MAX_PERTURBATION"
    --adversarial-watermark-weight "$ADVERSARIAL_WATERMARK_WEIGHT"
    --adversarial-target-lpips "$ADVERSARIAL_TARGET_LPIPS"
    --adversarial-lpips-net "$ADVERSARIAL_LPIPS_NET"
    --adversarial-selection-min-step "$ADVERSARIAL_SELECTION_MIN_STEP"
    --adversarial-selection-min-psnr "$ADVERSARIAL_SELECTION_MIN_PSNR"
    --steps "$ADVERSARIAL_STEPS"
    --lr "$ADVERSARIAL_LR"
    --decision-success-ber "$DECISION_SUCCESS_BER"
    --decision-ber-weight "$DECISION_BER_WEIGHT"
    --decision-quality-metric "$DECISION_QUALITY_METRIC"
    --candidate-selection "$CANDIDATE_SELECTION"
    --candidate-selection-seed "$CANDIDATE_SELECTION_SEED"
    --candidate-weighting "$CANDIDATE_WEIGHTING"
    --random-candidate-weighting "$RANDOM_CANDIDATE_WEIGHTING"
    --device "$DEVICE"
    --output-dir "$OUTPUT_DIR"
)

case "${ADVERSARIAL_QUALITY_CONSTRAINT,,}" in
    true|1|yes) CMD+=(--adversarial-quality-constraint) ;;
    false|0|no) CMD+=(--no-adversarial-quality-constraint) ;;
    *) echo "ADVERSARIAL_QUALITY_CONSTRAINT must be true or false." >&2; exit 2 ;;
esac
case "${ENABLE_ADVERSARIAL_ATTACK,,}" in
    true|1|yes) CMD+=(--enable-adversarial-attack) ;;
    false|0|no) CMD+=(--no-enable-adversarial-attack) ;;
    *) echo "ENABLE_ADVERSARIAL_ATTACK must be true or false." >&2; exit 2 ;;
esac
case "${ENABLE_SEQUENTIAL_SEARCH,,}" in
    true|1|yes)
        CMD+=(--enable-sequential-search --sequential-max-steps "$SEQUENTIAL_MAX_STEPS"
              --sequential-search-strategy "$SEQUENTIAL_SEARCH_STRATEGY"
              --sequential-beam-width "$SEQUENTIAL_BEAM_WIDTH"
              --sequential-tps-scales "$SEQUENTIAL_TPS_SCALES"
              --sequential-jpeg-qualities "$SEQUENTIAL_JPEG_QUALITIES"
              --sequential-vae-qualities "$SEQUENTIAL_VAE_QUALITIES"
              --sequential-adversarial-steps "$SEQUENTIAL_ADVERSARIAL_STEPS"
              --sequential-tps-search-trials "$SEQUENTIAL_TPS_SEARCH_TRIALS"
              --sequential-verification-trials "$SEQUENTIAL_VERIFICATION_TRIALS"
              --sequential-lambda-family "$SEQUENTIAL_LAMBDA_FAMILY"
              --sequential-lambda-ber "$SEQUENTIAL_LAMBDA_BER"
              --sequential-lambda-lpips "$SEQUENTIAL_LAMBDA_LPIPS"
              --sequential-selection-lpips-limit "$SEQUENTIAL_SELECTION_LPIPS_LIMIT"
              --sequential-ber-target "$SEQUENTIAL_BER_TARGET"
              --sequential-convergence-patience "$SEQUENTIAL_CONVERGENCE_PATIENCE"
              --sequential-min-relative-hv-gain "$SEQUENTIAL_MIN_RELATIVE_HV_GAIN"
              --sequential-min-ber-gain "$SEQUENTIAL_MIN_BER_GAIN"
              --sequential-adversarial-convergence-patience "$SEQUENTIAL_ADVERSARIAL_CONVERGENCE_PATIENCE"
              --sequential-adversarial-convergence-min-delta "$SEQUENTIAL_ADVERSARIAL_CONVERGENCE_MIN_DELTA"
              --sequential-adversarial-convergence-warmup "$SEQUENTIAL_ADVERSARIAL_CONVERGENCE_WARMUP")
        case "${SEQUENTIAL_REPLAY_VERIFICATION,,}" in true|1|yes) CMD+=(--sequential-replay-verification) ;; false|0|no) CMD+=(--no-sequential-replay-verification) ;; *) echo "SEQUENTIAL_REPLAY_VERIFICATION must be true or false." >&2; exit 2 ;; esac
        case "${SEQUENTIAL_ENABLE_CONVERGENCE,,}" in true|1|yes) CMD+=(--sequential-enable-convergence) ;; false|0|no) CMD+=(--no-sequential-enable-convergence) ;; *) echo "SEQUENTIAL_ENABLE_CONVERGENCE must be true or false." >&2; exit 2 ;; esac
        ;;
    false|0|no) CMD+=(--no-enable-sequential-search) ;;
    *) echo "ENABLE_SEQUENTIAL_SEARCH must be true or false." >&2; exit 2 ;;
esac

if [[ -n "${BATCH_MANIFEST:-}" ]]; then
    CMD+=(--batch-manifest "$BATCH_MANIFEST")
    [[ "${BATCH_RERUN_FAILED:-1}" == 1 ]] || CMD+=(--no-batch-rerun-failed)
    [[ "$EXCLUDE_TRUE_DECODER_FROM_CANDIDATES" == true ]] && CMD+=(--batch-exclude-true-decoder) || CMD+=(--no-batch-exclude-true-decoder)
fi

if ((${#EVAL_METHODS[@]})); then
    selected_count=0
    for method in "${EVAL_METHODS[@]}"; do
        if is_excluded_decoder "$method"; then echo "Skipping excluded decoder: $method" >&2; continue; fi
        CMD+=(--eval-method "$method")
        ((selected_count += 1))
    done
    ((selected_count > 0)) || { echo "No evaluation decoders remain." >&2; exit 2; }
else
    CMD+=(--candidate-top-k "$CANDIDATE_TOP_K")
    for method in "${EXCLUDED_DECODERS[@]}"; do append_entropy_exclusion "$method"; done
    [[ "$EXCLUDE_TRUE_DECODER_FROM_CANDIDATES" == true ]] && append_entropy_exclusion "$TRUE_DECODER"
fi

echo "Running portable TPS/decoder evaluation"
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
