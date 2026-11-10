"""Create an evaluation index in registered-image filename order."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.misc.colmap_utils import read_colmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--wan", action="store_true", help="Check availability of frames appended by the existing Wan sampler")
    args = parser.parse_args()
    model = args.root / args.model_dir if args.model_dir is not None else None
    frames = read_colmap(args.root, args.root / args.images_dir, model)
    if not 0 < args.context <= args.targets <= len(frames):
        parser.error(f"Require 0 < context <= targets <= {len(frames)} registered images")
    required_frames = args.targets + (4 if args.wan and args.targets % 4 == 0 else 0)
    if required_frames > len(frames):
        parser.error(f"Wan sampling needs {required_frames} registered images after padding; found {len(frames)}")
    # Keep targets consecutive for temporal autoencoders; context spans the same interval.
    entry = {args.root.name: {
        "context": np.linspace(0, args.targets - 1, args.context).round().astype(int).tolist(),
        "target": list(range(args.targets)),
    }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; frames are sorted by registered image name.")


if __name__ == "__main__":
    main()
