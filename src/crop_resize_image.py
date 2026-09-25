#!/usr/bin/env python3
"""Center-crop an image by a retained fraction and resize it back to its size."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--ratio", type=float, default=0.98)
    args = parser.parse_args()

    if not 0.0 < args.ratio <= 1.0:
        raise SystemExit("--ratio must be in (0, 1].")

    source = Path(args.source)
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop_width = max(1, min(width, round(width * args.ratio)))
        crop_height = max(1, min(height, round(height * args.ratio)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        resized = cropped.resize((width, height), Image.Resampling.BICUBIC)
        resized.save(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
