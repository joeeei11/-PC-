from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_recognition_nn import TrainingConfig, load_demo_dataset, load_image_folder_dataset, save_artifact, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个基于神经网络的人脸识别模型。")
    parser.add_argument("--dataset", choices=["demo", "custom"], default="demo", help="demo 会自动下载公开数据集；custom 使用本地图片目录。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "custom", help="自定义数据集目录，结构为 data_dir/person_name/*.jpg")
    parser.add_argument("--model-out", type=Path, default=ROOT / "artifacts" / "face_recognizer.joblib", help="训练后模型输出路径。")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例。")
    parser.add_argument("--pca-components", type=int, default=120, help="PCA 降维维度。")
    parser.add_argument("--hidden-layers", type=str, default="256,128", help="隐藏层结构，例如 256,128。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--max-iter", type=int, default=120, help="MLP 最大训练轮数。")
    parser.add_argument("--batch-size", type=int, default=32, help="batch 大小。")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_layers = tuple(int(part.strip()) for part in args.hidden_layers.split(",") if part.strip())
    config = TrainingConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        pca_components=args.pca_components,
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
    )

    if args.dataset == "demo":
        dataset = load_demo_dataset()
    else:
        dataset = load_image_folder_dataset(args.data_dir)

    artifact = train_model(dataset, config)
    output_path = save_artifact(artifact, args.model_out)

    summary = {
        "dataset_source": dataset.source,
        "num_classes": len(dataset.label_names),
        "num_samples": int(len(dataset.labels)),
        "train_size": artifact["train_size"],
        "test_size": artifact["test_size"],
        "accuracy": artifact["metrics"]["accuracy"],
        "model_path": str(output_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n分类报告:\n")
    print(artifact["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
