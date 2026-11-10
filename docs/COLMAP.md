# Native COLMAP evaluation

Install dependencies in your SceneTok environment (including pycolmap):

```bash
pip install -r requirements.txt
```

Point `dataset.root` directly at one scene (no `test/` wrapper):

```text
my_scene/
  images/               # Original images matching COLMAP camera dimensions
  sparse/0/             # Complete COLMAP binary or text reconstruction
    cameras.bin
    images.bin
    points3D.bin
```

Models directly in `sparse/`, `colmap/`, or the scene root are also detected.
If multiple models exist, select one with `dataset.model_dir=sparse/0`.
`dataset.images_dir` and `dataset.model_dir` accept scene-relative or absolute paths.
The points3D file is part of the reconstruction format, but its points are not model inputs.

Registered images are sorted lexicographically by their full COLMAP names, including
subdirectories. Use zero-padded frame names to preserve temporal order. Unregistered
images are ignored. Image dimensions must match their COLMAP cameras; use matching
cameras if your images have already been resized or undistorted.

Each image uses its own camera intrinsics. Distorted cameras are rectified with
pycolmap, then images are resized and center-cropped to `dataset.shape`. Intrinsics
are updated and normalized; poses are OpenCV camera-to-world. Equirectangular
images require perspective extraction first. COLMAP scene scale is preserved.

## Generate the view index

Run from the repository root. For the VideoDCAE model, this selects 12 context
views across the first 32 registered frames and evaluates those 32 frames:

```bash
python -m src.scripts.create_colmap_index /data/my_scene \
  --output assets/evaluation_index/my_scene.json --context 12 --targets 32
```

The scene must have at least 32 registered images. Context/target overlap is intentional
for a smoke test; edit the JSON to use held-out views for quantitative evaluation.
Indices are zero-based positions in sorted registered-image order, not COLMAP image IDs.
For Wan use `--wan --targets 64` and `dataset/view_sampler=evaluation_video_wan`
with the matching Wan experiment/checkpoint. COLMAP uses the original sampler:
when the requested count is divisible by 4, it appends the next 4 frame indices.
For example, 64 requested frames require at least 68 registered images; 68 requested
frames require at least 72. The loader checks bounds after sampling, before loading
any selected RGB frames. The `--wan` flag checks this availability when generating
an index; it does not write the extra frames into the JSON (the sampler adds them).

No multiple-of-17 input restriction is imposed. The existing Wan renderer retains
only complete 17-frame groups, and VideoDCAE retains complete 4-frame groups.
Thus loaded targets and rendered outputs can differ in length, as in the original
pipeline. For evaluation, prefer 64 requested / 68 loaded and rendered Wan frames.
Targets after sampling must cover at least one configured denoising window.
Duplicate and out-of-range indices are rejected.
The index generator also accepts `--images-dir` and `--model-dir`.

## Run SceneTok on one GPU

Download the default RE10K + VideoDCAE combination (4 files including latent statistics).
Optional downloads are commented out in the script, with notes explaining which default
weights they replace. Uncomment the desired commands before switching model presets:

```bash
bash scripts/download_models.sh
```

After generating the index above, edit `COLMAP_ROOT`, `INDEX_PATH`, `OUTPUT_DIR`,
`MODEL` and `GPU_ID` at the top of `scripts/run_infer.sh`, then run from the repository root:

```bash
bash scripts/run_infer.sh
```

The script selects a preset and executes one Python inference command.
`MODEL=re10k_videodc` is the default (RE10K + VA-VAE + VideoDCAE).
Use `dl3dv_videodc` for DL3DV + VA-VAE + VideoDCAE, or `dl3dv_wan` for
DL3DV + VA-VAE + Wan. Presets select the experiment, checkpoint and view sampler;
autoencoder paths come from the experiment. Generate a Wan-compatible index before
switching to Wan. The COLMAP model is expected in `sparse/0`.
DL3DV presets enable the legacy focal-scaling conventions used by their pretrained
checkpoints (see `KNOWN_BUGS.md`). The COLMAP adapter honors these experiment flags;
the default RE10K preset keeps geometrically normalized intrinsics without this scaling.
Downloading SceneGen weights does not enable SceneGen inference in this script.
The equivalent command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main \
  +experiment=scenetok_va-vdc_lognorm_re10k_scratch \
  mode=test dataset=colmap dataset.root=/data/my_scene \
  dataset/view_sampler=evaluation_video \
  dataset.view_sampler.index_path=assets/evaluation_index/my_scene.json \
  trainer.devices=1 trainer.num_nodes=1 trainer.limit_test_batches=1 \
  data_loader.test.batch_size=1 data_loader.test.num_workers=0 \
  model.compressor.ckpt_path=checkpoints/va-videodc_re10k.ckpt \
  model.compressor.load_strict=false \
  model.denoiser.ckpt_path=checkpoints/va-videodc_re10k.ckpt \
  model.denoiser.load_strict=false \
  wandb.activated=false hydra.run.dir=outputs/colmap_test
```

This adapter currently supports single-scene SceneTok evaluation, not training or
SceneGen conditioning. It loads the selected RGB frames, including target ground
truth used by the existing evaluation pipeline. Camera reconstruction quality,
scene scale and domain differences can affect pretrained-model results.

## Test outputs

`mode=test` saves side-by-side PNG comparisons to `cmp/<scene>/`: ground truth
on the left, prediction on the right. Filenames use the selected target indices.
Only rendered frames are paired; any trailing targets truncated by the temporal
decoder are omitted. Separate `gt/` and `predicted/` outputs are no longer written.
Input views are still saved under `context/<scene>/`. Existing output files from
previous runs are not removed; use a new output directory for a clean result.
