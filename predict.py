from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_recognition_nn import load_artifact, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的人脸识别模型进行预测。")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "face_recognizer.joblib", help="模型文件路径。")
    parser.add_argument("--image", type=Path, required=True, help="待识别的人脸图片路径。")
    parser.add_argument("--top-k", type=int, default=3, help="返回前 k 个最可能的人。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.model)
    result = predict_image(artifact, args.image, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
