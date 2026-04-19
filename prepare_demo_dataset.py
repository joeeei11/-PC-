from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from face_recognition_nn import export_olivetti_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并导出公开人脸数据集到本地文件夹。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "demo" / "olivetti_faces", help="导出的目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_olivetti_dataset(args.output_dir)
    print(f"数据集已导出到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
