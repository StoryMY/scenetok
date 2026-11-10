#!/usr/bin/env bash
# Run from the repository root; generate the scene's evaluation index first.
COLMAP_ROOT="/data/my_scene"
INDEX_PATH="assets/evaluation_index/my_scene.json"
# Default: re10k_videodc = RE10K SceneTok + VA-VAE + VideoDCAE.
# Alternatives: dl3dv_videodc or dl3dv_wan. All three use VA-VAE for context.
MODEL="re10k_videodc"
OUTPUT_DIR="outputs/my_scene_${MODEL}"
GPU_ID=0

case "$MODEL" in
    re10k_videodc)  # Default combination.
        EXPERIMENT="scenetok_va-vdc_lognorm_re10k_scratch"
        CHECKPOINT="checkpoints/va-videodc_re10k.ckpt"
        VIEW_SAMPLER="evaluation_video"
        ;;
    dl3dv_videodc)
        EXPERIMENT="scenetok_va-vdc_shift8_dl3dv_finetuned"
        CHECKPOINT="checkpoints/va-videodc_dl3dv.ckpt"
        VIEW_SAMPLER="evaluation_video"
        ;;
    dl3dv_wan)
        # Generate a matching index with --wan --targets 64 (at least 68 images).
        EXPERIMENT="scenetok_va-wan_shift4_dl3dv_finetuned"
        CHECKPOINT="checkpoints/va-wan_dl3dv.ckpt"
        VIEW_SAMPLER="evaluation_video_wan"
        ;;
    *) echo "Unknown MODEL: $MODEL" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_ID" python -m src.main \
    "+experiment=$EXPERIMENT" \
    mode=test \
    dataset=colmap \
    "dataset.root='$COLMAP_ROOT'" \
    dataset.images_dir=images \
    dataset.model_dir=sparse/0 \
    "dataset/view_sampler=$VIEW_SAMPLER" \
    "dataset.view_sampler.index_path='$INDEX_PATH'" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.limit_test_batches=1 \
    data_loader.test.batch_size=1 \
    data_loader.test.num_workers=0 \
    "model.compressor.ckpt_path=$CHECKPOINT" \
    model.compressor.load_strict=false \
    "model.denoiser.ckpt_path=$CHECKPOINT" \
    model.denoiser.load_strict=false \
    wandb.activated=false \
    "hydra.run.dir='$OUTPUT_DIR'"
