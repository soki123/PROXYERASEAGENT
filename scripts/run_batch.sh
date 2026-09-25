#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
SINGLE_RUN_SCRIPT="${SINGLE_RUN_SCRIPT:-$SCRIPT_DIR/run_single.sh}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/batch}"
IMAGES_PER_CATEGORY="${IMAGES_PER_CATEGORY:-10}"
IMAGE_OFFSET="${IMAGE_OFFSET:-0}"
SUCCESS_BER="${SUCCESS_BER:-0.1}"
CATEGORIES="${CATEGORIES:-}"
RERUN_FAILED="${RERUN_FAILED:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_TRUE_DECODER_FROM_CANDIDATES="${EXCLUDE_TRUE_DECODER_FROM_CANDIDATES:-true}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-3}"
CANDIDATE_SELECTION="${CANDIDATE_SELECTION:-entropy_topk}"
CANDIDATE_SELECTION_SEED="${CANDIDATE_SELECTION_SEED:-0}"
CANDIDATE_WEIGHTING="${CANDIDATE_WEIGHTING:-auto}"
RANDOM_CANDIDATE_WEIGHTING="${RANDOM_CANDIDATE_WEIGHTING:-uniform}"
# Optional center-crop preprocessing. false keeps the original input; when
# enabled, CROP_RATIO is the retained width/height fraction and the result is
# resized back before entropy ranking and attack evaluation.
CROP_ENABLED="${CROP_ENABLED:-false}"
CROP_RATIO="${CROP_RATIO:-0.98}"
CROP_INPUT_ROOT="${CROP_INPUT_ROOT:-$OUTPUT_ROOT/.crop_inputs}"
CROP_HELPER="${CROP_HELPER:-$PROJECT_ROOT/src/crop_resize_image.py}"
SEQUENTIAL_SEARCH_STRATEGY="${SEQUENTIAL_SEARCH_STRATEGY:-pareto_beam}"
SEQUENTIAL_BEAM_WIDTH="${SEQUENTIAL_BEAM_WIDTH:-}"
# Comma-separated physical GPU ids. Empty keeps the original single-process
# execution; e.g. GPU_IDS=0,1,3 launches one worker per GPU.
GPU_IDS="${GPU_IDS:-}"
DEVICE="${DEVICE:-cuda:0}"
strategy_from_cli=0
beam_width_from_cli=0

while (( $# > 0 )); do
    case "$1" in
        --sequential-search-strategy)
            if (( $# < 2 )); then
                echo "--sequential-search-strategy requires a value." >&2
                exit 2
            fi
            SEQUENTIAL_SEARCH_STRATEGY="$2"
            strategy_from_cli=1
            shift 2
            ;;
        --sequential-search-strategy=*)
            SEQUENTIAL_SEARCH_STRATEGY="${1#*=}"
            strategy_from_cli=1
            shift
            ;;
        --sequential-beam-width)
            if (( $# < 2 )); then
                echo "--sequential-beam-width requires a value." >&2
                exit 2
            fi
            SEQUENTIAL_BEAM_WIDTH="$2"
            beam_width_from_cli=1
            shift 2
            ;;
        --sequential-beam-width=*)
            SEQUENTIAL_BEAM_WIDTH="${1#*=}"
            beam_width_from_cli=1
            shift
            ;;
        *)
            echo "Unknown batch option: $1" >&2
            exit 2
            ;;
    esac
done

if (( strategy_from_cli == 1 && beam_width_from_cli == 0 )); then
    SEQUENTIAL_BEAM_WIDTH=""
fi
if [[ -z "${SEQUENTIAL_BEAM_WIDTH:-}" ]]; then
    case "${SEQUENTIAL_SEARCH_STRATEGY,,}" in
        greedy) SEQUENTIAL_BEAM_WIDTH=1 ;;
        *) SEQUENTIAL_BEAM_WIDTH=3 ;;
    esac
fi

MANIFEST="$OUTPUT_ROOT/manifest.tsv"
FAILED_LOG="$OUTPUT_ROOT/failed_runs.tsv"

canonical_decoder() {
    case "$1" in
        CIN) printf '%s\n' "cin" ;;
        ChunkySeal) printf '%s\n' "chunkyseal" ;;
        FIN) printf '%s\n' "fin" ;;
        HiDDeN) printf '%s\n' "hidden" ;;
        InvisMark) printf '%s\n' "invismark" ;;
        LightweightMark) printf '%s\n' "lightweightmark" ;;
        MBRS) printf '%s\n' "mbrs" ;;
        PIMoG) printf '%s\n' "pimog" ;;
        RoSteALS) printf '%s\n' "rosteals" ;;
        VideoSeal) printf '%s\n' "videoseal" ;;
        trustmark|TrustMark) printf '%s\n' "trustmark" ;;
        *) return 1 ;;
    esac
}

category_is_enabled() {
    local category="$1"
    local item
    local -a requested_categories

    if [[ -z "$CATEGORIES" ]]; then
        return 0
    fi
    IFS=',' read -r -a requested_categories <<< "$CATEGORIES"
    for item in "${requested_categories[@]}"; do
        item="${item//[[:space:]]/}"
        if [[ "${item,,}" == "${category,,}" ]]; then
            return 0
        fi
    done
    return 1
}

aggregate_results() {
    if [[ ! -f "$MANIFEST" ]]; then
        return
    fi
    "$PYTHON_BIN" - "$MANIFEST" "$OUTPUT_ROOT" "$SUCCESS_BER" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
success_ber = float(sys.argv[3])

with manifest_path.open(newline="", encoding="utf-8") as handle:
    manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

per_image_rows = []
category_stats = defaultdict(lambda: {"planned": 0, "completed": 0, "failed": 0, "successes": 0})

for item in manifest_rows:
    category = item["category"]
    stats = category_stats[category]
    stats["planned"] += 1
    run_dir = Path(item["output_dir"])
    verification_path = run_dir / "verification" / "final_verification.json"
    failed_marker = run_dir / ".failed"
    row = {
        "category": category,
        "true_decoder": item["true_decoder"],
        "sample_id": item["sample_id"],
        "image_path": item["image_path"],
        "status": "pending",
        "selected_attack": "",
        "selected_attack_value": "",
        "ber_mean": "",
        "ber_std": "",
        "psnr_mean": "",
        "lpips_mean": "",
        "selection_mode": "",
        "candidate_selection": "",
        "candidate_selection_seed": "",
        "candidate_weighting": "",
        "candidate_methods": "",
        "selected_sequence": "",
        "exceeds_single_frontier": "",
        "success_ber_threshold": success_ber,
        "success": "",
        "output_dir": str(run_dir),
    }
    if verification_path.is_file():
        try:
            payload = json.loads(verification_path.read_text(encoding="utf-8"))
            ber_mean = float(payload["ber_mean"])
            selected_attack_value = payload.get("selected_attack_value", "")
            if isinstance(selected_attack_value, (list, dict)):
                selected_attack_value = json.dumps(selected_attack_value, ensure_ascii=False)
            method_selection = payload.get("method_selection") or {}
            # Newer single-run evaluators store selection metadata in the
            # decision artifact rather than repeating it in final verification.
            # Keep aggregation compatible with both layouts.
            if not method_selection:
                decision_path = run_dir / "decision" / "final_result.json"
                if decision_path.is_file():
                    try:
                        decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
                        method_selection = decision_payload.get("method_selection") or {}
                    except Exception:
                        method_selection = {}
            def optional_float(value):
                return "" if value is None or value == "" else float(value)

            row.update(
                {
                    "status": "ok",
                    "selected_attack": payload.get("selected_attack", ""),
                    "selected_attack_value": selected_attack_value,
                    "ber_mean": ber_mean,
                    "ber_std": float(payload.get("ber_std", 0.0)),
                    "psnr_mean": optional_float(payload.get("psnr_mean", math.nan)),
                    "lpips_mean": optional_float(payload.get("lpips_mean", math.nan)),
                    "selection_mode": payload.get("selection_mode", "legacy_single_attack"),
                    "candidate_selection": method_selection.get("candidate_selection", ""),
                    "candidate_selection_seed": method_selection.get("candidate_selection_seed", ""),
                    "candidate_weighting": method_selection.get(
                        "candidate_weighting",
                        method_selection.get(
                            "random_candidate_weighting", method_selection.get("weight_source", "")
                        ),
                    ),
                    "candidate_methods": json.dumps(
                        method_selection.get("methods", []), ensure_ascii=False
                    ),
                    "selected_sequence": json.dumps(
                        payload.get("selected_sequence", []), ensure_ascii=False
                    ),
                    "exceeds_single_frontier": payload.get("exceeds_single_frontier", ""),
                    "success": ber_mean >= success_ber,
                }
            )
            stats["completed"] += 1
            if ber_mean >= success_ber:
                stats["successes"] += 1
        except Exception as exc:
            row["status"] = f"invalid_result:{type(exc).__name__}"
            stats["failed"] += 1
    elif failed_marker.is_file():
        row["status"] = "failed"
        stats["failed"] += 1
    per_image_rows.append(row)

per_image_fields = [
    "category", "true_decoder", "sample_id", "image_path", "status",
    "selected_attack", "selected_attack_value", "ber_mean", "ber_std",
    "psnr_mean", "lpips_mean", "selection_mode", "candidate_selection",
    "candidate_selection_seed", "candidate_weighting", "candidate_methods", "selected_sequence",
    "exceeds_single_frontier", "success_ber_threshold", "success", "output_dir",
]
with (output_root / "per_image_results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=per_image_fields)
    writer.writeheader()
    writer.writerows(per_image_rows)

category_rows = []
for category in sorted(category_stats):
    stats = category_stats[category]
    completed = stats["completed"]
    planned = stats["planned"]
    category_rows.append(
        {
            "category": category,
            "successful_images": stats["successes"],
            "completed_images": completed,
            "failed_images": stats["failed"],
            "planned_images": planned,
            "success_rate_completed": stats["successes"] / completed if completed else "",
            "success_rate_over_planned": stats["successes"] / planned if planned else "",
            "success_ber_threshold": success_ber,
        }
    )

category_fields = [
    "category", "successful_images", "completed_images", "failed_images",
    "planned_images", "success_rate_completed", "success_rate_over_planned",
    "success_ber_threshold",
]
with (output_root / "category_success_rates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=category_fields)
    writer.writeheader()
    writer.writerows(category_rows)

overall_planned = sum(item["planned"] for item in category_stats.values())
overall_completed = sum(item["completed"] for item in category_stats.values())
overall_failed = sum(item["failed"] for item in category_stats.values())
overall_successes = sum(item["successes"] for item in category_stats.values())
summary = {
    "success_definition": f"true-decoder ber_mean >= {success_ber}",
    "overall": {
        "successful_images": overall_successes,
        "completed_images": overall_completed,
        "failed_images": overall_failed,
        "planned_images": overall_planned,
        "success_rate_completed": overall_successes / overall_completed if overall_completed else None,
        "success_rate_over_planned": overall_successes / overall_planned if overall_planned else None,
    },
    "categories": category_rows,
    "per_image_results_csv": str(output_root / "per_image_results.csv"),
    "category_success_rates_csv": str(output_root / "category_success_rates.csv"),
}
(output_root / "batch_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("\nCurrent category success rates:")
for item in category_rows:
    rate = item["success_rate_completed"]
    rate_text = f"{100.0 * rate:.2f}%" if rate != "" else "N/A"
    print(
        f"  {item['category']}: {item['successful_images']}/{item['completed_images']} "
        f"completed = {rate_text}; failed={item['failed_images']}; planned={item['planned_images']}"
    )
print(f"Summary: {output_root / 'batch_summary.json'}")
PY
}

validate_configuration() {
    if [[ ! -f "$SINGLE_RUN_SCRIPT" ]]; then
        echo "Single-run script not found: $SINGLE_RUN_SCRIPT" >&2
        exit 2
    fi
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "Data root not found: $DATA_ROOT" >&2
        exit 2
    fi
    if ! [[ "$IMAGES_PER_CATEGORY" =~ ^[0-9]+$ ]] || (( IMAGES_PER_CATEGORY <= 0 )); then
        echo "IMAGES_PER_CATEGORY must be a positive integer." >&2
        exit 2
    fi
    if ! [[ "$IMAGE_OFFSET" =~ ^[0-9]+$ ]]; then
        echo "IMAGE_OFFSET must be a non-negative integer." >&2
        exit 2
    fi
    case "${CROP_ENABLED,,}" in
        true|1|yes) CROP_ENABLED="true" ;;
        false|0|no) CROP_ENABLED="false" ;;
        *) echo "CROP_ENABLED must be true or false." >&2; exit 2 ;;
    esac
    if ! [[ "$CROP_RATIO" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
        || ! awk "BEGIN { exit !($CROP_RATIO > 0 && $CROP_RATIO <= 1) }"; then
        echo "CROP_RATIO must be in (0, 1]." >&2
        exit 2
    fi
    if [[ "$CROP_ENABLED" == "true" && ! -f "$CROP_HELPER" ]]; then
        echo "Crop helper not found: $CROP_HELPER" >&2
        exit 2
    fi
    case "${SEQUENTIAL_SEARCH_STRATEGY,,}" in
        greedy)
            if [[ "$SEQUENTIAL_BEAM_WIDTH" != "1" ]]; then
                echo "greedy requires SEQUENTIAL_BEAM_WIDTH=1." >&2
                exit 2
            fi
            ;;
        pareto_beam)
            if [[ "$SEQUENTIAL_BEAM_WIDTH" != "3" ]]; then
                echo "pareto_beam requires SEQUENTIAL_BEAM_WIDTH=3." >&2
                exit 2
            fi
            ;;
        *)
            echo "SEQUENTIAL_SEARCH_STRATEGY must be greedy or pareto_beam." >&2
            exit 2
            ;;
    esac

    if [[ -n "$GPU_IDS" ]]; then
        local gpu_id
        local -a gpu_list
        IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
        if (( ${#gpu_list[@]} == 0 )); then
            echo "GPU_IDS must be a comma-separated list of GPU ids." >&2
            exit 2
        fi
        for gpu_id in "${gpu_list[@]}"; do
            if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
                echo "GPU_IDS must contain only numeric ids: $GPU_IDS" >&2
                exit 2
            fi
        done
    fi
}

build_manifest() {
    local category true_decoder selected index image_path manifest_image_path sample_id run_dir
    local -a category_dirs category_images

    mkdir -p "$OUTPUT_ROOT"
    printf 'category\ttrue_decoder\tsample_id\timage_path\toutput_dir\n' > "$MANIFEST"
    if [[ ! -f "$FAILED_LOG" ]]; then
        printf 'category\tsample_id\texit_code\timage_path\toutput_dir\n' > "$FAILED_LOG"
    fi

    mapfile -t category_dirs < <(
        find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
    )
    planned_total=0
    for category in "${category_dirs[@]}"; do
        if [[ "$category" == _* ]] || ! category_is_enabled "$category"; then
            continue
        fi
        if ! true_decoder="$(canonical_decoder "$category")"; then
            echo "Skipping unsupported category directory: $category" >&2
            continue
        fi

        mapfile -d '' -t category_images < <(
            find "$DATA_ROOT/$category" -mindepth 2 -maxdepth 2 \
                -type f -name encoded.png -print0 | LC_ALL=C sort -z
        )
        selected=0
        for ((index = IMAGE_OFFSET; index < ${#category_images[@]} && selected < IMAGES_PER_CATEGORY; index++)); do
            image_path="${category_images[$index]}"
            sample_id="$(basename "$(dirname "$image_path")")"
            manifest_image_path="$image_path"
            if [[ "$CROP_ENABLED" == "true" ]]; then
                manifest_image_path="$CROP_INPUT_ROOT/$category/$sample_id/encoded.png"
                "$PYTHON_BIN" "$CROP_HELPER" "$image_path" "$manifest_image_path" --ratio "$CROP_RATIO"
            fi
            run_dir="$OUTPUT_ROOT/$category/$sample_id"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$category" "$true_decoder" "$sample_id" "$manifest_image_path" "$run_dir" >> "$MANIFEST"
            ((selected += 1))
            ((planned_total += 1))
        done
        if (( selected < IMAGES_PER_CATEGORY )); then
            echo "Warning: $category selected only $selected images after offset $IMAGE_OFFSET." >&2
        fi
    done
}

print_configuration() {
    echo "Batch evaluation configuration:"
    echo "  data root       : $DATA_ROOT"
    echo "  output root     : $OUTPUT_ROOT"
    echo "  images/category : $IMAGES_PER_CATEGORY"
    echo "  image offset    : $IMAGE_OFFSET"
    echo "  planned runs    : $planned_total"
    echo "  success         : true-decoder ber_mean >= $SUCCESS_BER"
    echo "  exclude target  : $EXCLUDE_TRUE_DECODER_FROM_CANDIDATES"
    echo "  candidates      : $CANDIDATE_SELECTION top_k=$CANDIDATE_TOP_K seed=$CANDIDATE_SELECTION_SEED weighting=$CANDIDATE_WEIGHTING random_weighting=$RANDOM_CANDIDATE_WEIGHTING"
    echo "  crop            : $CROP_ENABLED (ratio=$CROP_RATIO)"
    echo "  search strategy : $SEQUENTIAL_SEARCH_STRATEGY beam_width=$SEQUENTIAL_BEAM_WIDTH"
    echo "  gpu workers     : ${GPU_IDS:-single process} (device=$DEVICE)"
    echo "  categories      : ${CATEGORIES:-all supported categories}"
}

record_run_statuses() {
    local run_exit_code="$1"
    local category _true_decoder sample_id image_path run_dir
    local verification_path failed_marker

    while IFS=$'\t' read -r category _true_decoder sample_id image_path run_dir; do
        [[ "$category" == "category" ]] && continue
        verification_path="$run_dir/verification/final_verification.json"
        failed_marker="$run_dir/.failed"
        if [[ -f "$verification_path" ]]; then
            echo "  completed $category/$sample_id"
        elif [[ -f "$failed_marker" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$category" "$sample_id" "$run_exit_code" "$image_path" "$run_dir" >> "$FAILED_LOG"
            echo "  failed $category/$sample_id" >&2
        fi
    done < "$MANIFEST"
}

run_manifest_worker() {
    local gpu_id="$1"
    local worker_manifest="$2"
    local worker_log="$3"
    local _category first_true_decoder _sample_id first_image _output_dir

    IFS=$'\t' read -r _category first_true_decoder _sample_id first_image _output_dir < <(
        sed -n '2p' "$worker_manifest"
    )
    env \
        CUDA_VISIBLE_DEVICES="$gpu_id" \
        DEVICE="$DEVICE" \
        WATERMARKED_IMAGE="$first_image" \
        TRUE_DECODER="$first_true_decoder" \
        OUTPUT_DIR="$OUTPUT_ROOT" \
        BATCH_MANIFEST="$worker_manifest" \
        BATCH_RERUN_FAILED="$RERUN_FAILED" \
        DECISION_SUCCESS_BER="$SUCCESS_BER" \
        EXCLUDE_TRUE_DECODER_FROM_CANDIDATES="$EXCLUDE_TRUE_DECODER_FROM_CANDIDATES" \
        CANDIDATE_TOP_K="$CANDIDATE_TOP_K" \
        CANDIDATE_SELECTION="$CANDIDATE_SELECTION" \
        CANDIDATE_SELECTION_SEED="$CANDIDATE_SELECTION_SEED" \
        CANDIDATE_WEIGHTING="$CANDIDATE_WEIGHTING" \
        RANDOM_CANDIDATE_WEIGHTING="$RANDOM_CANDIDATE_WEIGHTING" \
        CROP_ENABLED="$CROP_ENABLED" \
        CROP_RATIO="$CROP_RATIO" \
        SEQUENTIAL_SEARCH_STRATEGY="$SEQUENTIAL_SEARCH_STRATEGY" \
        SEQUENTIAL_BEAM_WIDTH="$SEQUENTIAL_BEAM_WIDTH" \
        bash "$SINGLE_RUN_SCRIPT" > "$worker_log" 2>&1
}

run_batch_multi_gpu() {
    local -a gpu_list worker_manifests worker_logs worker_pids
    local gpu_index gpu_id worker_manifest worker_log worker_pid worker_rc
    local worker_count=0

    IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
    local worker_root="$OUTPUT_ROOT/multi_gpu"
    mkdir -p "$worker_root/manifests" "$worker_root/logs"

    for gpu_index in "${!gpu_list[@]}"; do
        gpu_id="${gpu_list[$gpu_index]}"
        worker_manifest="$worker_root/manifests/manifest_gpu_${gpu_id}.tsv"
        worker_log="$worker_root/logs/gpu_${gpu_id}.log"
        printf 'category\ttrue_decoder\tsample_id\timage_path\toutput_dir\n' > "$worker_manifest"
        worker_manifests[$gpu_index]="$worker_manifest"
        worker_logs[$gpu_index]="$worker_log"
    done

    # Round-robin assignment keeps the number of images balanced without
    # changing the original manifest or output paths.
    local row_index=0
    while IFS= read -r manifest_row; do
        gpu_index=$((row_index % ${#gpu_list[@]}))
        printf '%s\n' "$manifest_row" >> "${worker_manifests[$gpu_index]}"
        ((row_index += 1))
    done < <(tail -n +2 "$MANIFEST")

    echo "Launching $row_index images on ${#gpu_list[@]} GPUs: $GPU_IDS"
    for gpu_index in "${!gpu_list[@]}"; do
        worker_manifest="${worker_manifests[$gpu_index]}"
        worker_log="${worker_logs[$gpu_index]}"
        if [[ "$(wc -l < "$worker_manifest")" -le 1 ]]; then
            continue
        fi
        gpu_id="${gpu_list[$gpu_index]}"
        run_manifest_worker "$gpu_id" "$worker_manifest" "$worker_log" &
        worker_pids[$gpu_index]=$!
        ((worker_count += 1))
        echo "  gpu=$gpu_id manifest=$worker_manifest log=$worker_log"
    done

    local overall_rc=0
    for gpu_index in "${!worker_pids[@]}"; do
        worker_pid="${worker_pids[$gpu_index]}"
        if wait "$worker_pid"; then
            worker_rc=0
        else
            worker_rc=$?
            overall_rc=$worker_rc
        fi
        echo "  gpu=${gpu_list[$gpu_index]} exit_code=$worker_rc"
    done
    echo "Multi-GPU workers completed: $worker_count"
    return "$overall_rc"
}

run_batch() {
    local _category first_true_decoder _sample_id first_image _output_dir
    local run_exit_code

    if (( planned_total == 0 )); then
        return
    fi

    IFS=$'\t' read -r _category first_true_decoder _sample_id first_image _output_dir < <(
        sed -n '2p' "$MANIFEST"
    )
    echo "Run all $planned_total images in one process with shared classification/attack models."
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Dry run: skipping batch execution."
        return
    fi

    if [[ -n "$GPU_IDS" ]]; then
        run_batch_multi_gpu
        run_exit_code=$?
    else
        env \
        WATERMARKED_IMAGE="$first_image" \
        TRUE_DECODER="$first_true_decoder" \
        OUTPUT_DIR="$OUTPUT_ROOT" \
        BATCH_MANIFEST="$MANIFEST" \
        BATCH_RERUN_FAILED="$RERUN_FAILED" \
        DECISION_SUCCESS_BER="$SUCCESS_BER" \
        EXCLUDE_TRUE_DECODER_FROM_CANDIDATES="$EXCLUDE_TRUE_DECODER_FROM_CANDIDATES" \
        CANDIDATE_TOP_K="$CANDIDATE_TOP_K" \
        CANDIDATE_SELECTION="$CANDIDATE_SELECTION" \
        CANDIDATE_SELECTION_SEED="$CANDIDATE_SELECTION_SEED" \
        CANDIDATE_WEIGHTING="$CANDIDATE_WEIGHTING" \
        RANDOM_CANDIDATE_WEIGHTING="$RANDOM_CANDIDATE_WEIGHTING" \
        CROP_ENABLED="$CROP_ENABLED" \
        CROP_RATIO="$CROP_RATIO" \
        SEQUENTIAL_SEARCH_STRATEGY="$SEQUENTIAL_SEARCH_STRATEGY" \
        SEQUENTIAL_BEAM_WIDTH="$SEQUENTIAL_BEAM_WIDTH" \
            bash "$SINGLE_RUN_SCRIPT" 2>&1 | tee "$OUTPUT_ROOT/batch_run.log"
        run_exit_code=${PIPESTATUS[0]}
    fi
    record_run_statuses "$run_exit_code"
}

validate_configuration
build_manifest
print_configuration

# On an interrupted run, keep any partial results that were already produced.
trap 'aggregate_results || true' EXIT
run_batch
trap - EXIT
aggregate_results
