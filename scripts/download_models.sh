#!/usr/bin/env bash
# 默认下载 4 个文件；其余已公开模型的下载命令保留在下面的注释中。
# 默认推理组合：re10k_videodc（RE10K SceneTok + VA-VAE + VideoDCAE）。
# 按需取消可选组合的下载命令注释，并在 run_infer.sh 中切换 MODEL。
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

download() {
    local name="$1" url="$2"
    local destination="$CHECKPOINT_DIR/$name"
    if [[ -s "$destination" ]]; then
        printf 'Already exists: %s\n' "$destination"
        return
    fi
    # Resume partial downloads; only rename after curl completes successfully.
    curl --fail --location --retry 3 --continue-at - \
        --output "$destination.part" "$url"
    mv -- "$destination.part" "$destination"
}

# 默认组合 re10k_videodc：下面 4 个文件即可运行默认推理。
download "va-videodc_re10k.ckpt" \
    "https://nextcloud.mpi-klsb.mpg.de/index.php/s/6Y7EsosfbnpcRxj/download"
download "vavae-imagenet256-f16d32-dinov2.pt" \
    "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/vavae-imagenet256-f16d32-dinov2.pt"
download "latents_stats.pt" \
    "https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/resolve/main/latents_stats.pt"
download "F32T4C128_AE.safetensors" \
    "https://huggingface.co/hpcai-tech/Open-Sora-v2-Video-DC-AE/resolve/main/F32T4C128_AE.safetensors"

# 可选组合 dl3dv_videodc：用 va-videodc_dl3dv.ckpt 替换默认的 va-videodc_re10k.ckpt。
# VA-VAE、统计文件和 VideoDCAE 仍使用默认文件；run_infer.sh 设置 MODEL="dl3dv_videodc"。
# download "va-videodc_dl3dv.ckpt" \
#     "https://nextcloud.mpi-klsb.mpg.de/index.php/s/aYBX7atFNKkmdSE/download"
# 可选组合 dl3dv_wan：用 va-wan_dl3dv.ckpt 替换默认的 va-videodc_re10k.ckpt，
# 用 Wan2.2_VAE.pth 替换默认的 F32T4C128_AE.safetensors。
# VA-VAE 和统计文件仍使用默认文件；run_infer.sh 设置 MODEL="dl3dv_wan"。
# download "va-wan_dl3dv.ckpt" \
#     "https://nextcloud.mpi-klsb.mpg.de/index.php/s/X7yzk7QANtwawPc/download"
# download "Wan2.2_VAE.pth" \
#     "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"

# 可选 SceneGen：用 va-videodc_re10k_scene.ckpt 替换默认的 va-videodc_re10k.ckpt，
# 再额外选择一个 scenegen_shift{1,4,12}_re10k.ckpt（这三个生成模型互为替代）。
# VA-VAE、统计文件和 VideoDCAE 仍使用默认文件。
# SceneGen 需要 src.main_scene 及匹配的实验配置，不能使用当前 SceneTok run_infer.sh。
# download "va-videodc_re10k_scene.ckpt" \
#     "https://huggingface.co/datasets/asimbluemoon/SceneTok/resolve/main/va-videodc_re10k_scene.ckpt"
# download "scenegen_shift1_re10k.ckpt" \
#     "https://nextcloud.mpi-klsb.mpg.de/index.php/s/3XrHrKQR8diodAa/download"
# download "scenegen_shift4_re10k.ckpt" \
#     "https://nextcloud.mpi-klsb.mpg.de/index.php/s/GNbPMdd6gWtM56k/download"
# download "scenegen_shift12_re10k.ckpt" \
#     "https://nextcloud.mpi-klsb.mpg.de/index.php/s/oDDgXYwDXijQo6a/download"
