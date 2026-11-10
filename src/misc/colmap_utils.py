"""COLMAP camera and pose adapters, adapted from Orbis-AE."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

class ColmapCamera:
    """Delegate camera-model-specific distortion math to COLMAP."""

    def __init__(self, camera):
        import pycolmap

        self.camera = camera
        self.options = pycolmap.UndistortCameraOptions()
        self.pinhole = camera.model.name in {"PINHOLE", "SIMPLE_PINHOLE"}
        if camera.model.name == "EQUIRECTANGULAR":
            raise ValueError("EQUIRECTANGULAR input requires perspective view extraction first")
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
            raise ValueError(f"Invalid COLMAP camera {camera.camera_id}")
        if not camera.verify_params():
            raise ValueError(f"Invalid parameters for COLMAP camera {camera.camera_id}")
        if camera.focal_length_x <= 0 or camera.focal_length_y <= 0:
            raise ValueError(f"Non-positive focal length for COLMAP camera {camera.camera_id}")

    def prepare(self, image):
        import pycolmap

        expected = (self.camera.width, self.camera.height)
        if image.size != expected:
            raise ValueError(f"Image size {image.size} does not match COLMAP camera size {expected}")
        if self.pinhole:
            return image, self.camera.calibration_matrix()
        bitmap = pycolmap.Bitmap.from_array(np.asarray(image, dtype=np.uint8))
        try:
            rectified, camera = pycolmap.undistort_image(self.options, bitmap, self.camera)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Cannot rectify COLMAP camera model {self.camera.model.name}") from error
        return Image.fromarray(rectified.to_array()), camera.calibration_matrix()


def _is_model(path: Path) -> bool:
    return any(all((path / f"{name}.{suffix}").is_file() for name in ("cameras", "images", "points3D"))
               for suffix in ("bin", "txt"))


def find_colmap_model(scene_dir: Path) -> Path:
    candidates = {p.resolve() for p in (scene_dir, scene_dir / "sparse", scene_dir / "colmap") if _is_model(p)}
    sparse = scene_dir / "sparse"
    if sparse.is_dir():
        candidates.update(p.resolve() for p in sparse.iterdir() if p.is_dir() and _is_model(p))
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(f"No COLMAP model found in {scene_dir}; use dataset.model_dir to specify it")
        raise ValueError(f"Multiple COLMAP models found in {scene_dir}; select one with dataset.model_dir")
    return candidates.pop()


def read_colmap(scene_dir: Path, images_dir: Path, model_dir: Path | None):
    try:
        import pycolmap
    except ImportError as error:
        raise ImportError("COLMAP input requires: pip install -r requirements.txt") from error

    model_dir = model_dir if model_dir is not None else find_colmap_model(scene_dir)
    if not _is_model(model_dir):
        raise FileNotFoundError(f"Incomplete COLMAP model: {model_dir}")
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    images = sorted((i for i in reconstruction.images.values() if i.has_pose), key=lambda i: i.name)
    if not images:
        raise ValueError(f"COLMAP model has no registered images: {model_dir}")
    frames, cameras, names = [], {}, set()
    image_root = images_dir.resolve()
    for image in images:
        path = (image_root / image.name).resolve()
        if not path.is_relative_to(image_root) or image.name in names:
            raise ValueError(f"Invalid or duplicate COLMAP image name: {image.name}")
        names.add(image.name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing registered COLMAP image: {path}")
        if image.camera_id not in cameras:
            cameras[image.camera_id] = ColmapCamera(reconstruction.cameras[image.camera_id])
        pose = image.cam_from_world()
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = pose.rotation.matrix()
        extrinsics[:3, 3] = pose.translation
        frames.append(ColmapFrame(image.name, path, cameras[image.camera_id], np.linalg.inv(extrinsics)))
    return frames
@dataclass
class ColmapFrame:
    name: str
    path: Path
    camera: ColmapCamera
    extrinsics: np.ndarray  # OpenCV camera-to-world, as expected by SceneTok.


def load_frame(frame, shape):
    """Rectify, resize and center-crop, updating each frame's own intrinsics."""
    with Image.open(frame.path) as source:
        image, intrinsics = frame.camera.prepare(source.convert("RGB"))
    h, w = shape
    scale = max(w / image.width, h / image.height)
    width, height = max(w, round(image.width * scale)), max(h, round(image.height * scale))
    intrinsics = np.array(intrinsics, dtype=np.float32, copy=True)
    intrinsics[0] *= width / image.width
    intrinsics[1] *= height / image.height
    left, top = (width - w) // 2, (height - h) // 2
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top
    intrinsics[0] /= w
    intrinsics[1] /= h
    image = image.resize((width, height), Image.Resampling.LANCZOS).crop((left, top, left + w, top + h))
    pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
    return pixels, intrinsics


def validate_view_indices(context, target, num_frames):
    """Reject malformed indices without changing the sampler's frame selection."""
    for name, indices in (("context", context), ("target", target)):
        if not indices or any(type(i) is not int or not 0 <= i < num_frames for i in indices):
            raise ValueError(f"{name} must contain integer indices in [0, {num_frames})")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate {name} indices")
