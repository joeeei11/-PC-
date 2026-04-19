from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import fetch_olivetti_faces

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class DatasetBundle:
    features: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    image_size: tuple[int, int]
    source: str


def load_demo_dataset() -> DatasetBundle:
    dataset = fetch_olivetti_faces(shuffle=True, download_if_missing=True)
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    label_names = [f"person_{index:02d}" for index in sorted(np.unique(labels))]
    return DatasetBundle(
        features=features,
        labels=labels,
        label_names=label_names,
        image_size=(64, 64),
        source="Olivetti Faces",
    )


def export_olivetti_dataset(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    dataset = fetch_olivetti_faces(shuffle=False, download_if_missing=True)
    output_path.mkdir(parents=True, exist_ok=True)

    for index, (image, target) in enumerate(zip(dataset.images, dataset.target, strict=True)):
        class_name = f"person_{int(target):02d}"
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / f"{class_name}_{index:03d}.png"
        pil_image = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0), mode="L")
        pil_image.save(image_path)

    return output_path


def load_image_folder_dataset(root_dir: str | Path, image_size: tuple[int, int] = (64, 64)) -> DatasetBundle:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root_path}")

    class_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("至少需要两个类别目录，每个目录代表一个人。")

    features: list[np.ndarray] = []
    labels: list[int] = []
    label_names: list[str] = []

    for label_index, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if len(image_paths) < 2:
            raise ValueError(f"类别 {class_dir.name} 的图片数量不足 2 张，无法完成训练与测试划分。")

        label_names.append(class_dir.name)
        for image_path in image_paths:
            features.append(_load_image_as_vector(image_path, image_size))
            labels.append(label_index)

    feature_array = np.vstack(features).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    return DatasetBundle(
        features=feature_array,
        labels=label_array,
        label_names=label_names,
        image_size=image_size,
        source=str(root_path),
    )


def _load_image_as_vector(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.grayscale(image)
        image = ImageOps.fit(image, image_size, method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, -1)
