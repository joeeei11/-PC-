from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DatasetBundle


@dataclass(slots=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    pca_components: int = 120
    hidden_layer_sizes: tuple[int, ...] = (256, 128)
    learning_rate_init: float = 1e-3
    max_iter: int = 120
    batch_size: int = 32
    top_k: int = 3


def train_model(dataset: DatasetBundle, config: TrainingConfig) -> dict:
    _validate_dataset(dataset, config.test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=config.test_size,
        stratify=dataset.labels,
        random_state=config.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=_resolve_pca_components(x_train.shape[0], x_train.shape[1], config.pca_components), whiten=True, random_state=config.random_state)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=config.hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    batch_size=config.batch_size,
                    learning_rate_init=config.learning_rate_init,
                    max_iter=config.max_iter,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=config.random_state,
                    verbose=False,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    artifact = {
        "pipeline": pipeline,
        "label_names": dataset.label_names,
        "image_size": dataset.image_size,
        "source": dataset.source,
        "config": asdict(config),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }
    artifact["metrics"] = evaluate_model(artifact, x_test, y_test)
    return artifact


def evaluate_model(artifact: dict, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    pipeline: Pipeline = artifact["pipeline"]
    label_names: list[str] = artifact["label_names"]
    predictions = pipeline.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
        ),
    }


def save_artifact(artifact: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(model_path: str | Path) -> dict:
    return joblib.load(model_path)


def predict_image(artifact: dict, image_path: str | Path, top_k: int = 3) -> dict:
    pipeline: Pipeline = artifact["pipeline"]
    label_names: list[str] = artifact["label_names"]
    image_size: tuple[int, int] = tuple(artifact["image_size"])

    image_vector = _load_image_as_vector(image_path, image_size)
    probabilities = pipeline.predict_proba(image_vector)[0]
    predicted_index = int(np.argmax(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    return {
        "image_path": str(Path(image_path).resolve()),
        "predicted_label": label_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "top_matches": [
            {"label": label_names[index], "confidence": float(probabilities[index])}
            for index in top_indices
        ],
    }


def _validate_dataset(dataset: DatasetBundle, test_size: float) -> None:
    _, counts = np.unique(dataset.labels, return_counts=True)
    minimum_required = max(2, int(np.ceil(1 / test_size)))
    if np.any(counts < minimum_required):
        raise ValueError(
            f"每个类别至少需要 {minimum_required} 张图片，当前最少类别只有 {int(counts.min())} 张。"
        )


def _resolve_pca_components(num_samples: int, num_features: int, requested_components: int) -> int:
    max_valid = max(1, min(num_samples - 1, num_features))
    return min(requested_components, max_valid)


def _load_image_as_vector(image_path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.grayscale(image)
        image = ImageOps.fit(image, image_size, method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(1, -1)
