"""RGB evaluation dataset backed directly by a COLMAP reconstruction."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from ..misc.colmap_utils import read_colmap, load_frame, validate_view_indices


@dataclass
class DatasetCOLMAPCfg(DatasetCfgCommon):
    name: Literal["colmap"]
    root: Path
    images_dir: Path = Path("images")
    model_dir: Path | None = None


class DatasetCOLMAP(Dataset):
    def __init__(self, cfg, stage, view_sampler, force_shuffle=False):
        self.cfg = cfg
        self.view_sampler = view_sampler
        if stage == "train":
            raise ValueError("The COLMAP adapter currently supports evaluation only.")
        if any(cfg.precomputed_latents.values()):
            raise ValueError("COLMAP input requires RGB data, not precomputed latents.")
        if cfg.view_sampler.name not in {"evaluation_video", "evaluation_video_wan"}:
            raise ValueError("COLMAP requires evaluation_video or evaluation_video_wan.")
        if len(cfg.shape) != 2 or min(cfg.shape) <= 0:
            raise ValueError("dataset.shape must contain positive height and width.")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        # Construct native pycolmap objects inside the worker, never pickle them.
        cfg = self.cfg
        images_dir = cfg.root / cfg.images_dir
        model_dir = cfg.root / cfg.model_dir if cfg.model_dir is not None else None
        frames = read_colmap(cfg.root, images_dir, model_dir)
        scene = cfg.root.name
        entries = self.view_sampler.index.get(scene)
        if not entries:
            raise ValueError(f"No evaluation index for scene {scene}")
        entry = entries[0]
        wan = cfg.view_sampler.name == "evaluation_video_wan"
        validate_view_indices(entry.context, entry.target, len(frames))
        views, _ = self.view_sampler.sample(
            num_views=len(frames), num_latents=len(frames), scene=scene
        )
        # Validate the final indices, including any frames appended by the sampler.
        validate_view_indices(views.context.tolist(), views.target.tolist(), len(frames))
        target_count = len(views.target)
        latent_count = target_count // 17 * 5 if wan else target_count // 4
        if latent_count < cfg.view_sampler.num_target_views:
            raise ValueError("Target sequence is shorter than the configured denoising window")
        sample, loaded = {"scene": scene}, {}
        for group, indices in asdict(views).items():
            if indices is None:
                continue
            if indices.numel() == 0 or (indices < 0).any() or (indices >= len(frames)).any():
                raise ValueError(f"Invalid {group} indices for {scene} ({len(frames)} registered images)")
            selected = indices.tolist()
            for i in selected:
                if i not in loaded:
                    pixels, intrinsics = load_frame(frames[i], cfg.shape)
                    loaded[i] = (torch.from_numpy(pixels), torch.from_numpy(intrinsics))
            sample[group] = {
                "latent": torch.stack([loaded[i][0] for i in selected]),
                "intrinsics": torch.stack([loaded[i][1] for i in selected]),
                "extrinsics": torch.tensor(np.stack([frames[i].extrinsics for i in selected]), dtype=torch.float32),
                "index": indices,
            }
        return sample
